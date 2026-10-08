"""데이터 로더. 컬럼: date, open, high, low, close, volume (수정주가 권장)."""
import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df[COLS].astype(float)


def fetch_pykrx(code, start, end):
    """pip install pykrx 필요. start/end: 'YYYYMMDD'. 수정주가 일봉."""
    from pykrx import stock
    df = stock.get_market_ohlcv(start, end, code, adjusted=True)
    df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low",
                            "종가": "close", "거래량": "volume"})
    df.index.name = "date"
    return df[COLS].astype(float)


def synthetic(n=1500, seed=0, start=50000.0):
    """테스트용 추세/횡보 레짐이 섞인 가짜 일봉. 실제 성과 판단에 쓰면 안 됨."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([0.0006, -0.0003, 0.0], size=n // 60 + 1), 60)[:n]
    r = regime + rng.normal(0, 0.017, n)
    close = start * np.exp(np.cumsum(r))
    open_ = np.r_[start, close[:-1]] * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.lognormal(13, 0.4, n)
    idx = pd.bdate_range("2019-01-01", periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": vol}, index=idx)
