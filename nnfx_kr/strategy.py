"""NNFX 구성: Baseline + C1 + C2 + Volume + Exit + ATR 기반 리스크 관리 (롱온리)."""
from dataclasses import dataclass, asdict
import pandas as pd
from . import indicators as ind


@dataclass
class Params:
    baseline_type: str = "ema"     # ema | hma
    baseline_len: int = 20
    c1_rsi_len: int = 14           # C1: RSI > c1_th 이면 롱 상태
    c1_th: float = 50.0
    c2_fast: int = 12              # C2: MACD 히스토그램 > 0
    c2_slow: int = 26
    c2_sig: int = 9
    vol_len: int = 20              # Volume: 거래량 > 이동평균 * vol_mult
    vol_mult: float = 1.0
    atr_len: int = 14
    max_extend: float = 1.0        # 종가가 baseline 위로 ATR*x 이상 벌어지면 진입 보류
    exit_len: int = 10             # Exit: Supertrend 하락 전환
    exit_mult: float = 2.0
    sl_mult: float = 1.5           # 손절 = 진입가 - 1.5 ATR
    tp_mult: float = 1.0           # 1차 익절(절반) = 진입가 + 1.0 ATR, 이후 손절을 본전으로
    risk: float = 0.02             # 거래당 계좌 위험 2%

    def to_dict(self):
        return asdict(self)


def prepare(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    c = df["close"]
    base = ind.ema(c, p.baseline_len) if p.baseline_type == "ema" else ind.hma(c, p.baseline_len)
    atr = ind.atr(df, p.atr_len)
    c1 = ind.rsi(c, p.c1_rsi_len) > p.c1_th
    c2 = ind.macd_hist(c, p.c2_fast, p.c2_slow, p.c2_sig) > 0
    vol = df["volume"] > df["volume"].rolling(p.vol_len).mean() * p.vol_mult
    st = ind.supertrend_dir(df, p.exit_len, p.exit_mult)

    cross_up = (c > base) & (c.shift(1) <= base.shift(1))
    c1_flip = c1 & ~c1.shift(1, fill_value=False)
    states = (c > base) & c1 & c2 & vol & ((c - base) <= atr * p.max_extend)
    entry = (cross_up | c1_flip) & states
    exit_ = (st == -1) & (st.shift(1) == 1) | (c < base) & (c.shift(1) >= base.shift(1))
    warm = max(p.baseline_len, p.c2_slow + p.c2_sig, p.vol_len, p.atr_len) * 2
    out = pd.DataFrame({"base": base, "atr": atr, "entry": entry, "exit": exit_})
    out.iloc[:warm, out.columns.get_indexer(["entry"])] = False
    return out
