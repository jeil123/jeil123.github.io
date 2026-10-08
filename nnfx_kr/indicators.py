import numpy as np
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s, n):
    return wma(2 * wma(s, max(n // 2, 1)) - wma(s, n), max(int(np.sqrt(n)), 1))


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                      (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd_hist(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow)
    return m - ema(m, sig)


def supertrend_dir(df, n=10, mult=2.0):
    """+1 상승추세 / -1 하락추세."""
    a = atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    up, dn = hl2 + mult * a, hl2 - mult * a
    fu, fl = up.copy(), dn.copy()
    d = np.ones(len(df))
    for i in range(1, len(df)):
        if np.isnan(a[i]):
            continue
        fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        d[i] = 1 if c[i] > fu[i - 1] else (-1 if c[i] < fl[i - 1] else d[i - 1])
    return pd.Series(d, index=df.index)
