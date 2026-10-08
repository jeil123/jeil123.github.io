import numpy as np
from nnfx_kr import data
from nnfx_kr.backtest import simulate
from nnfx_kr.krx import tick_size, round_up, round_down
from nnfx_kr.strategy import Params


def test_tick():
    assert tick_size(1999) == 1 and tick_size(4999) == 5 and tick_size(71300) == 100
    assert round_up(71001) == 71100 and round_down(71099) == 71000


def test_no_lookahead_and_sanity():
    df = data.synthetic(1500, seed=1)
    r = simulate(df, Params())
    assert r.stats["trades"] > 0
    assert (r.equity > 0).all() and r.equity.notna().all()
    # 마지막 봉 데이터를 바꿔도 이전 시점 자금곡선은 같아야 한다
    df2 = df.copy(); df2.iloc[-1, :4] *= 1.2
    r2 = simulate(df2, Params())
    assert np.allclose(r.equity.iloc[:-2], r2.equity.iloc[:-2])
    assert (r.trades.entry >= 1).all()
