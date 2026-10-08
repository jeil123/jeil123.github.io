"""KRX 시장 규칙: 호가단위, 거래비용."""
import math
from dataclasses import dataclass


def tick_size(price: float, etf: bool = False) -> int:
    """2023.1 이후 통합된 KOSPI/KOSDAQ 주식 호가단위. ETF는 2,000원 기준 1원/5원."""
    if etf:
        return 1 if price < 2000 else 5
    for limit, tick in ((2000, 1), (5000, 5), (20000, 10), (50000, 50),
                        (200000, 100), (500000, 500)):
        if price < limit:
            return tick
    return 1000


def round_up(price: float, etf: bool = False) -> float:
    t = tick_size(price, etf)
    return math.ceil(price / t - 1e-9) * t


def round_down(price: float, etf: bool = False) -> float:
    t = tick_size(price, etf)
    return math.floor(price / t + 1e-9) * t


@dataclass
class Costs:
    commission: float = 0.00015   # 매수/매도 각각 (증권사별 상이)
    sell_tax: float = 0.0018      # 매도 시 증권거래세+농특세. 세율은 해마다 바뀌니 확인 후 조정
    slippage_ticks: int = 1       # 시장가 체결 가정 시 불리한 방향 슬리피지
