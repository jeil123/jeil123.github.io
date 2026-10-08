"""포워드 테스트: 파라미터를 고정하고 forward_start 이후 데이터만으로 모의매매를 재생한다.
재생 방식이라 매일 실행해도 상태가 어긋나지 않고, 결과 로그는 JSON으로 남긴다."""
import json
from pathlib import Path
from .backtest import simulate
from .strategy import Params


def run(df, p: Params, forward_start, state_path="forward_state.json", equity0=10_000_000, **kw):
    # 워밍업을 위해 전체 데이터로 신호를 만들되, 자금곡선은 forward_start부터 새로 시작
    sub = df[df.index >= forward_start]
    warm = df[df.index < forward_start].tail(300)
    res = simulate(df.loc[warm.index.union(sub.index)], p, equity0, **kw)
    state = dict(params=p.to_dict(), forward_start=str(forward_start),
                 last_bar=str(df.index[-1].date()), stats=res.stats,
                 position=res.position, next_open_order=res.pending)
    Path(state_path).write_text(json.dumps(state, ensure_ascii=False, indent=2, default=float))
    return res, state
