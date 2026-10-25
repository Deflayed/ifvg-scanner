"""Free LONG-only IFVG-after-HTF-FVG-sweep scanner -> Discord.

Setup per timeframe (1H, 2H, 3H, 4H, Daily, Weekly):
  A candle takes out a prior swing LOW while wicking into a bullish FVG from ANY
  higher timeframe, then within MAX_BARS candles a bearish FVG on the same timeframe
  gets a body close above it (bullish IFVG).
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
ALERT_TFS = ["1H", "2H", "3H", "4H", "Daily", "Weekly"]  # remove some to get fewer alerts
MIN_GAP_ATR = 0.25       # ignore FVGs smaller than this x ATR14 of their timeframe
SWING = 3                # swing strength: bars on each side
MAX_BARS = 8             # max candles from sweep to IFVG
MAX_ZONES = 5            # higher-TF bullish FVGs kept per timeframe
LOOKBACK_MIN = 90        # only send signals whose candle closed within this many minutes
INTRADAY_PERIOD = "90d"  # 1H history pulled each run
BATCH = 150              # tickers per Yahoo download request

TZ = ZoneInfo("America/New_York")
TF_NAMES = ["1H", "2H", "3H", "4H", "Daily", "Weekly", "Monthly"]
INTRADAY_MINUTES = [60, 120, 180, 240]
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


def build_tf_bars(df1h, dfd, now):
    # Days covered by the 1H data are rebuilt from it, so today's daily candle is current
    first = df1h.index.normalize().min()
    recent = _ohlc(df1h, df1h.index.normalize()).rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})
    dfd = pd.concat([dfd[dfd.index < first], recent])
    return ([intraday_bars(df1h, m, now) for m in INTRADAY_MINUTES]
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
    h, l, c = b["h"].to_numpy(), b["l"].to_numpy(), b["c"].to_numpy()
    a = atr(h, l, c)
    bear_tops = []
    swing_lo = None
    rows = []
    for i in range(len(b)):
        min_gap = a[i] * MIN_GAP_ATR
        bull_fvg = i >= 2 and l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap
        bear_fvg = i >= 2 and h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap

        # Bullish IFVG: body close above a recent bearish FVG
        ifvg_bull = any(c[i] > t for t in bear_tops)
        bear_tops = [t for t in bear_tops if c[i] <= t]
        if bear_fvg:
            bear_tops = (bear_tops + [l[i - 2]])[-5:]

        k = i - SWING
        if k >= SWING and l[k] < l[k - SWING:k].min() and l[k] <= l[k + 1:i + 1].min():
            swing_lo = l[k]
        sweep_low = swing_lo is not None and l[i] < swing_lo
        if sweep_low:
            swing_lo = None

        rows.append(dict(l=l[i], c=c[i], end=b["end"].iloc[i],
                         bull_fvg=bull_fvg, bull_top=l[i], bull_bot=h[i - 2] if i >= 2 else None,
                         sweep_low=sweep_low, ifvg_bull=ifvg_bull))
    return rows


def find_signals(ticker, tf_bars):
    """tf_bars: one bar DataFrame per TF_NAMES entry. Replays all TFs in close-time order."""
    events = [(r["end"], i, r) for i, b in enumerate(tf_bars) for r in analyze(b)]
    events.sort(key=lambda e: (e[0], e[1]))
    zones = []  # bullish FVGs: dicts tf, top, bot
    n = len(TF_NAMES)
    cnt = [0] * n
    arm, arm_src = [-1000] * n, [""] * n
    signals = []
    for end, i, r in events:
        cnt[i] += 1
        if i < n - 1:  # the top timeframe (Monthly) only supplies FVGs
            if r["sweep_low"]:
                hits = [z["tf"] for z in zones if z["tf"] > i and r["l"] <= z["top"] and r["c"] >= z["bot"]]
                if hits:
                    arm[i], arm_src[i] = cnt[i], TF_NAMES[max(hits)]
            if r["ifvg_bull"] and cnt[i] - arm[i] <= MAX_BARS:
                signals.append(dict(ticker=ticker, tf=TF_NAMES[i], src=arm_src[i], close=r["c"], end=end))
                arm[i] = -1000

        zones = [z for z in zones if not (z["tf"] == i and r["c"] < z["bot"])]
        if i >= 1:
            if r["bull_fvg"]:
                zones.append(dict(tf=i, top=r["bull_top"], bot=r["bull_bot"]))
            mine = [z for z in zones if z["tf"] == i]
            for z in mine[:-MAX_ZONES]:
                zones.remove(z)
    return signals


# ── Discord ──────────────────────────────────────────────────────────────────
def format_signal(s):
    return (f"🟢 **{s['ticker']}** | {s['tf']} bullish IFVG after swing-low sweep into "
            f"{s['src']} bullish FVG | Close: {s['close']:.2f}")


def send_discord(webhook, lines):
    import requests
    chunks, chunk = [], ""
    for line in lines:
        if len(chunk) + len(line) + 1 > 1900:
            chunks.append(chunk)
            chunk = ""
        chunk += line + "\n"
    chunks.append(chunk)
    for ch in chunks:
        r = requests.post(webhook, json={"content": ch}, timeout=20)
        if r.status_code == 429:  # Discord rate limit
            time.sleep(float(r.json().get("retry_after", 2)))
            r = requests.post(webhook, json={"content": ch}, timeout=20)
        r.raise_for_status()
        time.sleep(1)


def main():
    webhook = os.environ.get("DISCORD_WEBHOOK", "")
    now = datetime.now(TZ).replace(tzinfo=None)
    sent = json.loads(SENT_FILE.read_text()) if SENT_FILE.exists() else {}

    daily = load_daily(now.date().isoformat())
    tickers = sorted(daily)
    print(f"scanning {len(tickers)} stocks", file=sys.stderr)
    intr = download(tickers, "60m", INTRADAY_PERIOD)

    new = []
    for t in tickers:
        if t not in intr:
            continue
        try:
            signals = find_signals(t, build_tf_bars(intr[t], daily[t], now))
        except Exception as e:
            print(f"{t}: {e!r}", file=sys.stderr)
            continue
        for s in signals:
            key = f"{s['ticker']}|{s['tf']}|{s['end']:%Y-%m-%d %H:%M}"
            if s["tf"] in ALERT_TFS and s["end"] >= now - timedelta(minutes=LOOKBACK_MIN) and key not in sent:
                new.append(s)
                sent[key] = now.isoformat()

    new.sort(key=lambda s: TF_NAMES.index(s["tf"]), reverse=True)  # higher timeframes first
    lines = [format_signal(s) for s in new]
    print("\n".join(lines) or "no new signals")
    if lines and webhook:
        send_discord(webhook, lines)

    cutoff = (now - timedelta(days=7)).isoformat()
    SENT_FILE.write_text(json.dumps({k: v for k, v in sent.items() if v >= cutoff}, indent=0))


if __name__ == "__main__":
    main()
