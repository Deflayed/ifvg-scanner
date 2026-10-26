"""Free LONG-only IFVG-after-HTF-FVG-sweep scanner -> Discord.

Setup per timeframe (30m, 45m, 1H, 2H, 3H, 4H, Daily, Weekly):
  A candle is the first to take out the LOWEST swing low that formed inside a bullish
  FVG from ANY higher timeframe (since that FVG formed), then within MAX_BARS candles a bearish FVG on the same timeframe
  gets a body close above it (bullish IFVG), with no unfilled bearish FVG or volume
  imbalance left above it (same timeframe, or a higher timeframe close above price).
Weekly setups use Monthly FVGs as the higher timeframe. Only closed candles are used.

Universe: every US-listed stock (Nasdaq + NYSE + AMEX), filtered once a day by price
and average volume. Set UNIVERSE = "list" to scan only tickers.txt instead.
"""
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Settings ─────────────────────────────────────────────────────────────────
UNIVERSE = "all"         # "all" = every US-listed stock, "list" = tickers.txt
MIN_PRICE = 5.0          # skip stocks below this price
MIN_AVG_VOLUME = 500_000 # skip stocks trading fewer shares/day than this (20-day avg)
INCLUDE_ETFS = False
ALERT_TFS = ["30m", "45m", "1H", "2H", "3H", "4H", "Daily", "Weekly"]  # remove some to get fewer alerts
MIN_GAP_ATR = 0.25       # ignore FVGs smaller than this x ATR14 of their timeframe
MIN_VI_ATR = 0.05        # ignore volume imbalances smaller than this x ATR14 of their timeframe
SWING = 3                # swing strength: bars on each side
MAX_BARS = 8             # max candles from sweep to IFVG
MAX_ZONES = 5            # higher-TF FVGs kept per timeframe & side
BLOCK_HTF_ATR = 3.0      # skip if a higher-TF bearish FVG/VI sits within this many ATRs above the IFVG close (0 = off)
MAX_CHARTS = 40          # chart pictures per run; extra alerts are sent as text only
LOOKBACK_MIN = 90        # only send signals whose candle closed within this many minutes
INTRADAY_PERIOD = "60d"  # 15m history pulled each run (Yahoo max)
BATCH = 150              # tickers per Yahoo download request

TZ = ZoneInfo("America/New_York")
TF_NAMES = ["30m", "45m", "1H", "2H", "3H", "4H", "Daily", "Weekly", "Monthly"]
INTRADAY_MINUTES = [30, 45, 60, 120, 180, 240]
HERE = Path(__file__).parent
SENT_FILE = HERE / "sent.json"
DAILY_FILE = HERE / "daily.pkl"
SYMBOL_URLS = [
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
]


# ── Universe ─────────────────────────────────────────────────────────────────
def parse_symbol_file(text, sym_col):
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str).dropna(subset=[sym_col])
    df = df[~df[sym_col].str.startswith("File Creation")]
    df = df[df["Test Issue"] == "N"]
    if not INCLUDE_ETFS:
        df = df[df["ETF"] == "N"]
    junk = df["Security Name"].str.contains(r"Warrant|Units?\b|Rights?\b|Preferred|Notes due|Debentures", case=False, na=False)
    df = df[~junk & ~df[sym_col].str.contains(r"[$^=+]")]
    return [s.replace(".", "-") for s in df[sym_col]]


def all_us_symbols():
    import requests
    syms = set()
    for url, col in SYMBOL_URLS:
        syms.update(parse_symbol_file(requests.get(url, timeout=30).text, col))
    return sorted(syms)


# ── Data ─────────────────────────────────────────────────────────────────────
def _split(raw, tickers):
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if not df.empty:
            out[t] = df
    return out


def download(tickers, interval, period):
    import yfinance as yf
    out = {}
    for k in range(0, len(tickers), BATCH):
        part = tickers[k:k + BATCH]
        for attempt in range(3):
            try:
                raw = yf.download(part, interval=interval, period=period, group_by="ticker",
                                  auto_adjust=False, prepost=False, progress=False, threads=True)
                out.update(_split(raw, part))
                break
            except Exception as e:  # rate limits etc.
                print(f"batch {k}: {e!r}, retrying", file=sys.stderr)
                time.sleep(20 * (attempt + 1))
        time.sleep(1)
    for d in out.values():
        d.index = d.index.tz_convert(TZ).tz_localize(None) if d.index.tz is not None else d.index
    return out


def load_daily(today):
    """Daily bars + filtered universe, rebuilt once per day and cached."""
    if DAILY_FILE.exists():
        cache = pd.read_pickle(DAILY_FILE)
        if cache["date"] == today:
            return cache["daily"]
    if UNIVERSE == "all":
        symbols = all_us_symbols()
    else:
        lines = (HERE / "tickers.txt").read_text().splitlines()
        symbols = [t.strip().upper() for t in lines if t.strip() and not t.strip().startswith("#")]
    print(f"building daily cache for {len(symbols)} symbols", file=sys.stderr)
    # Filter on 1 month of data first, then pull 5 years (for Monthly FVGs) only for the keepers
    keep = [t for t, d in download(symbols, "1d", "1mo").items()
            if d["Close"].iloc[-1] >= MIN_PRICE and d["Volume"].tail(20).mean() >= MIN_AVG_VOLUME]
    daily = {}
    for t, d in download(keep, "1d", "5y").items():
        d.index = d.index.normalize()
        daily[t] = d[["Open", "High", "Low", "Close"]]
    pd.to_pickle({"date": today, "daily": daily}, DAILY_FILE)
    return daily


# ── Bars ─────────────────────────────────────────────────────────────────────
def _ohlc(df, keys):
    g = df.groupby(keys)
    return pd.DataFrame({"o": g["Open"].first(), "h": g["High"].max(),
                         "l": g["Low"].min(), "c": g["Close"].last()})


def intraday_bars(dfi, minutes, now):
    """Session-aligned bars (anchored at 9:30 ET like TradingView). Index = bar start."""
    idx = dfi.index
    since_open = idx.hour * 60 + idx.minute - 570
    starts = idx.normalize() + pd.to_timedelta(570 + (since_open // minutes) * minutes, unit="min")
    b = _ohlc(dfi, starts)
    session_close = b.index.normalize() + pd.Timedelta(hours=16)
    b["end"] = np.minimum(b.index + pd.Timedelta(minutes=minutes), session_close)
    return b[b["end"] <= now]


def daily_bars(dfd, now):
    b = dfd.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})[["o", "h", "l", "c"]]
    b["end"] = b.index.normalize() + pd.Timedelta(hours=16)
    return b[b["end"] <= now]


def period_bars(dfd, now, freq):
    """Weekly ("W-FRI") or monthly ("M") bars; a bar counts as closed at 16:00 on its period's last day."""
    b = _ohlc(dfd, dfd.index.to_period(freq))
    b.index = b.index.to_timestamp(how="end").normalize()
    b["end"] = b.index + pd.Timedelta(hours=16)
    return b[b["end"] <= now]


def build_tf_bars(df15, dfd, now):
    # Days covered by the 15m data are rebuilt from it, so today's daily candle is current
    first = df15.index.normalize().min()
    recent = _ohlc(df15, df15.index.normalize()).rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})
    dfd = pd.concat([dfd[dfd.index < first], recent])
    return ([intraday_bars(df15, m, now) for m in INTRADAY_MINUTES]
            + [daily_bars(dfd, now), period_bars(dfd, now, "W-FRI"), period_bars(dfd, now, "M")])


# ── Signals ──────────────────────────────────────────────────────────────────
def atr(h, l, c, n=14):
    prev = np.roll(c, 1)
    prev[0] = np.nan
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1] = tr[:n].mean()
        for i in range(n, len(tr)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def analyze(b):
    o, h, l, c = b["o"].to_numpy(), b["h"].to_numpy(), b["l"].to_numpy(), b["c"].to_numpy()
    a = atr(h, l, c)
    bears = []  # unfilled bearish FVGs on this timeframe: dicts top, bot, i0
    vis = []    # unfilled bearish volume imbalances (body gaps): dicts top, bot
    rows = []
    for i in range(len(b)):
        min_gap = a[i] * MIN_GAP_ATR
        bull_fvg = i >= 2 and l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap
        bear_fvg = i >= 2 and h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap
        # Bearish volume imbalance: this body opens fully below the previous body, wicks overlap
        vi_top = min(o[i - 1], c[i - 1]) if i >= 1 else None
        vi_bot = max(o[i], c[i])
        bear_vi = (i >= 1 and vi_top > vi_bot and l[i - 1] <= h[i]
                   and (vi_top - vi_bot) >= a[i] * MIN_VI_ATR)

        # Bullish IFVG: body close above a recent bearish FVG. Any bearish FVG still
        # unfilled above it afterwards blocks the setup (price can reject off it).
        inverted = [f for f in bears if c[i] > f["top"]]
        bears = [f for f in bears if c[i] <= f["top"] and h[i] < f["top"]]
        vis = [v for v in vis if c[i] <= v["top"] and h[i] < v["top"]]
        ifvg = max(inverted, key=lambda f: f["top"]) if inverted else None
        blocked = bool(bears) or bool(vis)
        if bear_fvg:
            bears = (bears + [dict(top=l[i - 2], bot=h[i], i0=i - 2)])[-5:]
        if bear_vi:
            vis = (vis + [dict(top=vi_top, bot=vi_bot)])[-5:]

        # Swing low at bar k, confirmed now that SWING bars after it have closed
        k = i - SWING
        pivot = None
        if k >= SWING and l[k] < l[k - SWING:k].min() and l[k] <= l[k + 1:i + 1].min():
            pivot = (l[k], k, b["end"].iloc[k])

        rows.append(dict(idx=i, h=h[i], l=l[i], c=c[i], atr=a[i], end=b["end"].iloc[i],
                         bull_fvg=bull_fvg, bull_top=l[i], bull_bot=h[i - 2] if i >= 2 else None,
                         bear_fvg=bear_fvg, bear_top=l[i - 2] if i >= 2 else None, bear_bot=h[i],
                         bear_vi=bear_vi, vi_top=vi_top, vi_bot=vi_bot,
                         pivot=pivot,
                         ifvg_bull=ifvg is not None, ifvg=ifvg, blocked=blocked))
    return rows


def find_signals(ticker, tf_bars):
    """tf_bars: one bar DataFrame per TF_NAMES entry. Replays all TFs in close-time order."""
    events = [(r["end"], i, r) for i, b in enumerate(tf_bars) for r in analyze(b)]
    events.sort(key=lambda e: (e[0], e[1]))
    zones = []  # higher-TF FVGs/VIs: dicts tf, bull, vi, top, bot, t (time it formed)
    n = len(TF_NAMES)
    cnt = [0] * n
    arm, arm_info = [-1000] * n, [None] * n
    pivots = [[] for _ in range(n)]  # confirmed swing lows per TF: (level, idx, candle end time)
    lows = [b["l"].to_numpy() for b in tf_bars]
    signals = []
    for end, i, r in events:
        cnt[i] += 1
        if i < n - 1:  # the top timeframe (Monthly) only supplies FVGs
            # Sweep: first candle to trade below the lowest swing low inside a higher-TF bullish FVG
            idx, best = r["idx"], None
            for z in zones:
                if not z["bull"] or z["vi"] or z["tf"] <= i or r["c"] < z["bot"]:
                    continue
                inside = [p for p in pivots[i] if z["bot"] <= p[0] <= z["top"] and p[2] >= z["t"]]
                if not inside:
                    continue
                level, k, _ = min(inside)
                if r["l"] < level and lows[i][k + 1:idx].min(initial=np.inf) >= level:
                    if best is None or z["tf"] > best[0]["tf"]:
                        best = (z, level, k)
            if best:
                arm[i] = cnt[i]
                arm_info[i] = dict(zone=best[0], sweep_idx=idx, swing_level=best[1], swing_idx=best[2])
            if r["ifvg_bull"] and cnt[i] - arm[i] <= MAX_BARS:
                near = r["c"] + BLOCK_HTF_ATR * r["atr"]
                htf_block = BLOCK_HTF_ATR > 0 and any(not z["bull"] and z["tf"] > i and z["top"] > r["c"] and z["bot"] <= near for z in zones)
                if not r["blocked"] and not htf_block:
                    info = arm_info[i]
                    signals.append(dict(ticker=ticker, tf=TF_NAMES[i], tf_idx=i, src=TF_NAMES[info["zone"]["tf"]],
                                        close=r["c"], end=end, idx=r["idx"], ifvg=r["ifvg"], **info))
                arm[i] = -1000

        if r["pivot"]:
            pivots[i] = (pivots[i] + [r["pivot"]])[-300:]

        # Maintain this timeframe's FVGs (used as higher-TF zones by lower timeframes)
        zones = [z for z in zones if not (z["tf"] == i and (
            (z["bull"] and r["c"] < z["bot"]) or (not z["bull"] and (r["c"] > z["top"] or r["h"] >= z["top"]))))]
        if i >= 1:
            if r["bull_fvg"]:
                zones.append(dict(tf=i, bull=True, vi=False, top=r["bull_top"], bot=r["bull_bot"], t=end))
            if r["bear_fvg"]:
                zones.append(dict(tf=i, bull=False, vi=False, top=r["bear_top"], bot=r["bear_bot"], t=end))
            if r["bear_vi"]:
                zones.append(dict(tf=i, bull=False, vi=True, top=r["vi_top"], bot=r["vi_bot"], t=end))
            for bull, vi in ((True, False), (False, False), (False, True)):
                mine = [z for z in zones if z["tf"] == i and z["bull"] == bull and z["vi"] == vi]
                for z in mine[:-MAX_ZONES]:
                    zones.remove(z)
    return signals


# ── Charts ───────────────────────────────────────────────────────────────────
def chart_png(b, s):
    """TradingView-style picture of the setup on its own timeframe. Returns PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lo = max(0, min(s["swing_idx"], s["ifvg"]["i0"], s["sweep_idx"]) - 25)
    hi = s["idx"]
    w = b.iloc[lo:hi + 1]
    x = np.arange(len(w))
    right = len(w) + 6
    bg, grid, txt = "#131722", "#2a2e39", "#d1d4dc"
    up, down = "#26a69a", "#ef5350"

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=110)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for xi, (o, h, l, c) in zip(x, w[["o", "h", "l", "c"]].to_numpy()):
        col = up if c >= o else down
        ax.vlines(xi, l, h, color=col, linewidth=1)
        ax.add_patch(Rectangle((xi - 0.35, min(o, c)), 0.7, max(abs(c - o), 1e-9), color=col))

    # Higher-TF bullish FVG
    z = s["zone"]
    zx = int(np.searchsorted(w["end"].to_numpy(), np.datetime64(z["t"])))
    zx = min(max(zx, 0), len(w) - 1)
    ax.add_patch(Rectangle((zx - 0.5, z["bot"]), right - zx, z["top"] - z["bot"], color="#2962ff", alpha=0.18, lw=0))
    ax.text(zx, z["top"], f" {s['src']} FVG", color="#5b8cff", fontsize=9, ha="left", va="bottom",
            bbox=dict(facecolor=bg, edgecolor="none", alpha=0.8, pad=1))

    # Inverted FVG
    f = s["ifvg"]
    fx = f["i0"] - lo
    ax.add_patch(Rectangle((fx - 0.5, f["bot"]), right - fx, f["top"] - f["bot"], color="#26a69a", alpha=0.25, lw=0))
    ax.text(right - 0.5, f["bot"], "IFVG", color=up, fontsize=9, ha="right", va="top")

    # Swept swing low
    sx, ex = s["swing_idx"] - lo, s["sweep_idx"] - lo
    ax.hlines(s["swing_level"], sx, ex, colors="#f0b90b", linestyles="dashed", linewidth=1.2)
    ax.text(sx, s["swing_level"], "lowest swing low in gap ", color="#f0b90b", fontsize=8, ha="right", va="center")
    ax.annotate("sweep", (ex, w["l"].iloc[ex]), xytext=(ex, w["l"].iloc[ex] - (w["h"].max() - w["l"].min()) * 0.06),
                color="#f0b90b", fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="#f0b90b"))
    ax.annotate("IFVG close", (x[-1], w["h"].iloc[-1]), xytext=(x[-1], w["h"].iloc[-1] + (w["h"].max() - w["l"].min()) * 0.08),
                color=up, fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=up))

    span = w["h"].max() - w["l"].min()
    ax.set_ylim(min(w["l"].min(), f["bot"]) - span * 0.12, max(w["h"].max(), f["top"]) + span * 0.15)
    ax.set_xlim(-1, right)
    step = max(1, len(w) // 8)
    ax.set_xticks(x[::step])
    fmt = "%b %d" if s["tf"] in ("Daily", "Weekly") else "%b %d %H:%M"
    ax.set_xticklabels([t.strftime(fmt) for t in w.index[::step]], fontsize=8)
    ax.yaxis.tick_right()
    ax.tick_params(colors=txt, labelsize=8)
    ax.grid(color=grid, linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.set_title(f"{s['ticker']}  ·  {s['tf']}  ·  bullish IFVG after sweep into {s['src']} FVG",
                 color=txt, fontsize=11, loc="left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


# ── Discord ──────────────────────────────────────────────────────────────────
def format_signal(s):
    return (f"🟢 **{s['ticker']}** | {s['tf']} bullish IFVG after swing-low sweep into "
            f"{s['src']} bullish FVG | Close: {s['close']:.2f}")


def _post(webhook, **kw):
    import requests
    for _ in range(5):
        r = requests.post(webhook, timeout=30, **kw)
        if r.status_code != 429:  # 429 = Discord rate limit
            r.raise_for_status()
            break
        time.sleep(float(r.json().get("retry_after", 2)))
    time.sleep(0.7)


def send_discord(webhook, items):
    """items: list of (text, png_bytes or None)."""
    text_only = []
    for text, png in items:
        if png is None:
            text_only.append(text)
            continue
        _post(webhook, data={"payload_json": json.dumps({"content": text})},
              files={"files[0]": ("setup.png", png, "image/png")})
    chunk = ""
    for line in text_only + [None]:
        if line is None or len(chunk) + len(line) + 1 > 1900:
            if chunk:
                _post(webhook, json={"content": chunk})
            chunk = ""
        if line is not None:
            chunk += line + "\n"


def main():
    webhook = os.environ.get("DISCORD_WEBHOOK", "")
    now = datetime.now(TZ).replace(tzinfo=None)
    sent = json.loads(SENT_FILE.read_text()) if SENT_FILE.exists() else {}

    daily = load_daily(now.date().isoformat())
    tickers = sorted(daily)
    print(f"scanning {len(tickers)} stocks", file=sys.stderr)
    intr = download(tickers, "15m", INTRADAY_PERIOD)

    new = []
    for t in tickers:
        if t not in intr:
            continue
        try:
            bars = build_tf_bars(intr[t], daily[t], now)
            signals = find_signals(t, bars)
        except Exception as e:
            print(f"{t}: {e!r}", file=sys.stderr)
            continue
        for s in signals:
            s["bars"] = bars[s["tf_idx"]]
            key = f"{s['ticker']}|{s['tf']}|{s['end']:%Y-%m-%d %H:%M}"
            if s["tf"] in ALERT_TFS and s["end"] >= now - timedelta(minutes=LOOKBACK_MIN) and key not in sent:
                new.append(s)
                sent[key] = now.isoformat()

    new.sort(key=lambda s: TF_NAMES.index(s["tf"]), reverse=True)  # higher timeframes first
    items = []
    for k, s in enumerate(new):
        png = None
        if k < MAX_CHARTS:
            try:
                png = chart_png(s["bars"], s)
            except Exception as e:
                print(f"chart {s['ticker']}: {e!r}", file=sys.stderr)
        items.append((format_signal(s), png))
    print("\n".join(t for t, _ in items) or "no new signals")
    if items and webhook:
        send_discord(webhook, items)

    cutoff = (now - timedelta(days=7)).isoformat()
    SENT_FILE.write_text(json.dumps({k: v for k, v in sent.items() if v >= cutoff}, indent=0))


if __name__ == "__main__":
    main()
