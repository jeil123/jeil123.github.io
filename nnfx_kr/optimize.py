"""파라미터 그리드 최적화 + 워크포워드(학습/검증 분리). 검증 구간 성과로만 판단할 것."""
import itertools
from dataclasses import replace
from .backtest import simulate
from .strategy import Params

DEFAULT_GRID = dict(baseline_len=[10, 20, 30, 50], c1_rsi_len=[10, 14],
                    exit_mult=[1.5, 2.0, 3.0], sl_mult=[1.5, 2.0], max_extend=[1.0, 1.5])


def score(stats, min_trades=10):
    """Calmar(CAGR/|MDD|) 기반. 표본 거래수가 적으면 탈락."""
    if stats["trades"] < min_trades:
        return -1e9
    return stats["cagr"] / max(abs(stats["max_dd"]), 0.02)


def grid_search(df, grid=None, base=Params(), **kw):
    grid = grid or DEFAULT_GRID
    keys, rows = list(grid), []
    for vals in itertools.product(*grid.values()):
        p = replace(base, **dict(zip(keys, vals)))
        s = simulate(df, p, **kw).stats
        rows.append((score(s), p, s))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


def walk_forward(df, folds=4, train_frac=0.7, grid=None, **kw):
    """겹치지 않는 구간 `folds`개로 나눠 각 구간의 앞 70%에서 최적화, 뒤 30%에서 검증."""
    size, out = len(df) // folds, []
    for k in range(folds):
        seg = df.iloc[k * size:(k + 1) * size]
        cut = int(len(seg) * train_frac)
        best = grid_search(seg.iloc[:cut], grid, **kw)[0]
        # 지표 워밍업을 위해 학습 구간 꼬리를 붙여 검증하고 검증 구간 성과만 기록
        oos = simulate(seg, best[1], **kw)
        eq = oos.equity.iloc[cut:]
        out.append(dict(fold=k, params=best[1], train=best[2],
                        oos_return=eq.iloc[-1] / eq.iloc[0] - 1,
                        oos_mdd=(eq / eq.cummax() - 1).min()))
    return out
