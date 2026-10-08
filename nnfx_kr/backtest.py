"""일봉 이벤트 백테스터.

체결 규칙(미래참조 방지): 신호는 종가 확정 후, 주문은 다음 거래일 시가.
장중 손절/익절은 고저가로 판정하며 같은 봉에서 둘 다 닿으면 손절 우선(보수적).
"""
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .krx import Costs, round_up, round_down
from .strategy import Params, prepare


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    pending: dict = field(default_factory=dict)   # 다음 거래일 시가에 낼 주문
    position: dict = field(default_factory=dict)


def simulate(df, p: Params, equity0=10_000_000, costs: Costs = Costs(), etf=False) -> Result:
    sig = prepare(df, p)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr, entry_s, exit_s = sig["atr"].values, sig["entry"].values, sig["exit"].values
    cash, pos = float(equity0), None
    pend_entry = pend_exit = False
    pend_atr = 0.0
    trades, eq = [], []

    def sell(qty, price, date, reason):
        nonlocal cash
        proceeds = qty * price * (1 - costs.commission - costs.sell_tax)
        cash += proceeds
        pos["pnl"] += proceeds
        pos["qty"] -= qty
        if pos["qty"] == 0:
            trades.append(dict(entry_date=pos["date"], exit_date=date,
                               entry=pos["px"], exit=price, reason=reason,
                               pnl=pos["pnl"] - pos["cost"],
                               ret=(pos["pnl"] - pos["cost"]) / pos["cost"]))
            return True
        return False

    for i in range(len(df)):
        d = df.index[i]
        if pend_exit and pos:
            px = round_down(o[i], etf)
            px = max(px - costs.slippage_ticks * (px - round_down(px - 1e-9, etf) or 1), 1)
            if sell(pos["qty"], px, d, "exit"):
                pos = None
        pend_exit = False
        if pend_entry and pos is None and not math.isnan(pend_atr):
            px = round_up(o[i], etf)
            tick = px - round_down(px - 1e-9, etf) or 1
            px += costs.slippage_ticks * tick
            equity = cash
            risk_per_share = p.sl_mult * pend_atr
            qty = int(min(equity * p.risk / risk_per_share,
                          cash / (px * (1 + costs.commission))))
            if qty > 0:
                cost = qty * px * (1 + costs.commission)
                cash -= cost
                pos = dict(date=d, px=px, qty=qty, q0=qty, cost=cost, pnl=0.0,
                           sl=round_down(px - risk_per_share, etf),
                           tp=round_up(px + p.tp_mult * pend_atr, etf), tp_done=False)
        pend_entry = False

        if pos:
            if l[i] <= pos["sl"]:
                px = min(o[i], pos["sl"]) if i > 0 and d != pos["date"] else pos["sl"]
                if sell(pos["qty"], round_down(px, etf), d, "stop" if not pos["tp_done"] else "breakeven"):
                    pos = None
            elif not pos["tp_done"] and h[i] >= pos["tp"]:
                half = pos["qty"] // 2
                if half > 0:
                    sell(half, pos["tp"], d, "tp1")
                pos["tp_done"] = True
                pos["sl"] = pos["px"]          # 본전 스톱
        if pos and exit_s[i]:
            pend_exit = True
        elif pos is None and entry_s[i]:
            pend_entry, pend_atr = True, atr[i]
        eq.append(cash + (pos["qty"] * c[i] if pos else 0))

    equity = pd.Series(eq, index=df.index)
    tr = pd.DataFrame(trades)
    pending = {}
    if pend_entry:
        pending = dict(action="BUY", atr=float(pend_atr),
                       stop=round_down(c[-1] - p.sl_mult * pend_atr, etf))
    elif pend_exit:
        pending = dict(action="SELL_ALL")
    position = dict(qty=pos["qty"], entry=pos["px"], stop=pos["sl"]) if pos else {}
    return Result(tr, equity, metrics(tr, equity), pending, position)


def metrics(tr, equity):
    n = len(equity)
    ret = equity.iloc[-1] / equity.iloc[0] - 1 if n else 0
    yrs = max(n / 252, 1e-9)
    dd = (equity / equity.cummax() - 1).min() if n else 0
    r = equity.pct_change().dropna()
    s = dict(total_return=ret, cagr=(1 + ret) ** (1 / yrs) - 1 if ret > -1 else -1,
             max_dd=dd, sharpe=(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
             trades=len(tr), win_rate=0.0, profit_factor=0.0)
    if len(tr):
        w, lo = tr.loc[tr.pnl > 0, "pnl"].sum(), -tr.loc[tr.pnl <= 0, "pnl"].sum()
        s["win_rate"] = float((tr.pnl > 0).mean())
        s["profit_factor"] = float(w / lo) if lo > 0 else float("inf")
    return s
