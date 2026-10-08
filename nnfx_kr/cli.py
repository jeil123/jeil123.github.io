import argparse
import json
import pandas as pd
from . import data
from .backtest import simulate
from .forward import run as fwd
from .live import dispatch
from .optimize import grid_search, walk_forward
from .strategy import Params


def load(a):
    if a.csv:
        return data.load_csv(a.csv)
    if a.code:
        return data.fetch_pykrx(a.code, a.start, a.end)
    return data.synthetic()


def fmt(s):
    return (f"수익률 {s['total_return']:+.1%} CAGR {s['cagr']:+.1%} MDD {s['max_dd']:.1%} "
            f"샤프 {s['sharpe']:.2f} 거래 {s['trades']} 승률 {s['win_rate']:.0%} PF {s['profit_factor']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nnfx_kr")
    ap.add_argument("cmd", choices=["backtest", "optimize", "walkforward", "forward"])
    ap.add_argument("--csv"); ap.add_argument("--code"); ap.add_argument("--name", default="")
    ap.add_argument("--start", default="20150101"); ap.add_argument("--end", default="20261231")
    ap.add_argument("--params", help="JSON 파라미터 덮어쓰기")
    ap.add_argument("--forward-start", default=None)
    ap.add_argument("--etf", action="store_true")
    a = ap.parse_args(argv)
    df = load(a)
    p = Params(**json.loads(a.params)) if a.params else Params()
    kw = dict(etf=a.etf)
    if a.cmd == "backtest":
        print(fmt(simulate(df, p, **kw).stats))
    elif a.cmd == "optimize":
        cut = int(len(df) * 0.7)
        best = grid_search(df.iloc[:cut], **kw)[:5]
        for sc, bp, s in best:
            print(f"score {sc:.2f} {fmt(s)}\n  {bp.to_dict()}")
        print("검증구간(OOS):", fmt(simulate(df, best[0][1], **kw).stats))
    elif a.cmd == "walkforward":
        for r in walk_forward(df, **kw):
            print(f"fold {r['fold']} OOS {r['oos_return']:+.1%} MDD {r['oos_mdd']:.1%}")
    else:
        fs = pd.Timestamp(a.forward_start) if a.forward_start else df.index[-60]
        res, state = fwd(df, p, fs, **kw)
        print(fmt(res.stats)); print("다음 주문:", state["next_open_order"] or "없음")
        dispatch(a.code or "-", a.name or "-", state)


if __name__ == "__main__":
    main()
