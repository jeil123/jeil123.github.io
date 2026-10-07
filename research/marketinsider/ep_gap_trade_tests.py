"""ep07·08·11 갭/시초가(일봉 근사) + ep02·05·06 돌파 매매 백테스트 + 손절 유무 비교."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import ep_common as C, ep01_signals as E

data, idx = C.load_all()
frames = {t: C.feats(d, (idx["KOSDAQ"] if t.endswith(".KQ") else idx["KOSPI"])["Close"]) for t, d in data.items()}
F = list(frames.values()); allf = pd.concat(F)
P = dict(E.PARAMS); res = {}
cost_in = P["fee"] + P["slip"]; cost_out = P["fee"] + P["slip"] + P["tax"]

# ---------- ep07·ep11 갭 구간별 ----------
def gap_row(sel, label):
    s = allf[sel & allf.oc.notna()]
    if len(s) < 30: return dict(구간=label, n=len(s))
    o5 = (s.c.shift(0) * 0 + 1)  # placeholder
    return dict(구간=label, n=len(s), 시가to종가=s.oc.mean(), 시가종가_양봉률=(s.oc > 0).mean(),
                시가=고가비중(s), 갭메움률=(s.l <= s.c / (1 + s.ret)).mean(), 종가후5일=s.f5.mean(), 종가후20일=s.f20.mean())
def 고가비중(s): return float((s.o >= s.h * 0.998).mean())
# 시가→+5일/20일 종가 보유수익 (시가 기준)
allf["o_f5"] = np.nan
for t, f in frames.items():
    c = f.c; f["o_f5"] = c.shift(-5) / f.o - 1; f["o_f20"] = c.shift(-20) / f.o - 1
allf = pd.concat(list(frames.values()))
bins = [(-1, -0.07), (-0.07, -0.05), (-0.05, -0.03), (-0.03, -0.01), (-0.01, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 1)]
rows = []
for lo, hi in bins:
    s = allf[(allf.gap > lo) & (allf.gap <= hi) & allf.oc.notna()]
    if len(s) < 30: rows.append(dict(갭=f"{lo:+.0%}~{hi:+.0%}", n=len(s))); continue
    rows.append(dict(갭=f"{lo:+.0%}~{hi:+.0%}", n=len(s), 시가to종가=s.oc.mean(), 양봉률=(s.oc > 0).mean(), 시가가고가=float((s.o >= s.h * 0.998).mean()),
                     갭메움=float(((s.l <= (s.c / (1 + s.ret))) if hi > 0 else (s.h >= (s.c / (1 + s.ret)))).mean()), 시가보유_5일=s.o_f5.mean(), 시가보유_20일=s.o_f20.mean()))
_b = allf[allf.oc.notna()]
res["gap_base"] = dict(시가to종가=float(_b.oc.mean()), 시가보유_5일=float(_b.o_f5.mean()), 시가보유_20일=float(_b.o_f20.mean()), 양봉률=float((_b.oc > 0).mean()))
res["gap_bins"] = rows

# 갭하락 체크리스트(검증 가능한 3개: 갭≤-3%, 거래량 2배, 시가가 20·60일선 아래)
below = lambda f: (f.o < f.ma20.shift(1)) & (f.o < f.ma60.shift(1))
chk = []
for name, fn in [("갭≤-3% 전체", lambda f: f.gap <= -0.03),
                 ("갭≤-3% & 거래량2배↑", lambda f: (f.gap <= -0.03) & (f.vr >= 2)),
                 ("갭≤-3% & 시가<20·60일선", lambda f: (f.gap <= -0.03) & below(f)),
                 ("갭≤-3% & 거래량2배↑ & 시가<20·60일선 (3개 모두)", lambda f: (f.gap <= -0.03) & (f.vr >= 2) & below(f)),
                 ("갭≤-3% & 3개 중 1개 이하", lambda f: (f.gap <= -0.03) & (((f.vr >= 2).astype(int) + below(f).astype(int)) == 0))]:
    s = pd.concat([f[fn(f).fillna(False)] for f in F]); s = s[s.oc.notna()]
    chk.append(dict(조건=name, n=len(s), 시가to종가=float(s.oc.mean()), 시가보유_5일=float(s.o_f5.mean()), 시가보유_20일=float(s.o_f20.mean()),
                    시가보유_5일_승률=float((s.o_f5 > 0).mean())))
res["gapdown_checklist"] = chk
# 갭상승 후 다음날/5일 (시가=고가 모양)
oh = []
for name, fn in [("갭≥3%", lambda f: f.gap >= 0.03), ("갭≥3% & 시가=고가(0.2%이내)", lambda f: (f.gap >= 0.03) & (f.o >= f.h * 0.998)),
                 ("갭≥3% & 종가가 시가보다 높음(양봉)", lambda f: (f.gap >= 0.03) & (f.oc > 0)), ("갭≥7%", lambda f: f.gap >= 0.07),
                 ("갭≥7% & 시가=고가", lambda f: (f.gap >= 0.07) & (f.o >= f.h * 0.998))]:
    s = pd.concat([f[fn(f).fillna(False)] for f in F]); s = s[s.f5.notna()]
    oh.append(dict(조건=name, n=len(s), 시가to종가=float(s.oc.mean()), 종가후1일=float(s.f1.mean()), 종가후5일=float(s.f5.mean()), 종가후20일=float(s.f20.mean())))
res["gapup_shapes"] = oh

# ---------- ep08 시초가 매매(일봉 근사): 전일 거래량 후보 → 시가 매수, -3% 손절, 아니면 종가 ----------
def open_trade(fn, stop=0.03, label=""):
    rets = []; yrs = []
    for f in F:
        m = fn(f).fillna(False)
        for i in np.where(m.values)[0]:
            if i + 1 >= len(f): continue
            o, l, c = f.o.values[i + 1], f.l.values[i + 1], f.c.values[i + 1]
            if np.isnan(o) or o <= 0: continue
            px = o * (1 + cost_in)
            ex = o * (1 - stop) if l <= o * (1 - stop) else c
            rets.append(ex * (1 - cost_out) / px - 1); yrs.append(f.index[i + 1])
    r = pd.Series(rets, index=pd.DatetimeIndex(yrs))
    if len(r) == 0: return dict(조건=label, n=0)
    w, lo = r[r > 0].sum(), -r[r < 0].sum()
    return dict(조건=label, n=len(r), 평균=float(r.mean()), 승률=float((r > 0).mean()), PF=float(w / lo) if lo > 0 else None,
                IS평균=float(r[r.index < C.SPLIT].mean()), OOS평균=float(r[r.index >= C.SPLIT].mean()))
cand = lambda f: (f.vr >= 2.5) & (f.c >= 0.95 * f.ch252)
rng = np.random.default_rng(0)
ot = [open_trade(cand, label="후보(전일 거래량2.5배↑ & 신고가 5%이내)")]
ot.append(open_trade(lambda f: (cand(f)) & (f.oc > 0), label="후보 & 전일 양봉"))
ot.append(open_trade(lambda f: (f.vr >= 2.5), label="전일 거래량 2.5배↑ (위치 무관)"))
ot.append(open_trade(lambda f: pd.Series(rng.random(len(f)) < 0.01, index=f.index), label="대조군: 무작위 1%"))
ot.append(open_trade(lambda f: pd.Series(rng.random(len(f)) < 0.01, index=f.index), stop=1.0, label="대조군: 무작위 1% (손절 없음)"))
ot.append(open_trade(cand, stop=1.0, label="후보 (손절 없음)"))
res["open_trade"] = ot

# ---------- 돌파 매매 백테스트 (ep02·05·06) ----------
def run(trig_fn, level_fn, P, label, boxfail=True, rand=None):
    trades = []
    for t, f in frames.items():
        o, h, l, c = (f[k].values for k in "ohlc"); n = len(c)
        if rand is None:
            trig = trig_fn(f).fillna(False).values; lv = level_fn(f).values if level_fn else None
        else:
            trig = rand.random(n) < 0.005; lv = None
        for i in np.where(trig)[0]:
            if i + 1 >= n or np.isnan(o[i + 1]): continue
            bhv = lv[i] if (boxfail and lv is not None) else None
            if bhv is not None and np.isnan(bhv): bhv = None
            j, ex = E.sim_trade(o, h, l, c, i + 1, bhv, P)
            px = o[i + 1] * (1 + cost_in); r = ex * (1 - cost_out) / px - 1
            trades.append((f.index[i + 1], r, j - (i + 1) + 1))
    if not trades: return dict(전략=label, n=0)
    r = pd.Series([x[1] for x in trades], index=pd.DatetimeIndex([x[0] for x in trades]))
    w, lo = r[r > 0].sum(), -r[r < 0].sum()
    return dict(전략=label, n=len(r), 승률=float((r > 0).mean()), 평균=float(r.mean()), 중앙=float(r.median()), PF=float(w / lo) if lo > 0 else None,
                평균보유일=float(np.mean([x[2] for x in trades])), 최악=float(r.min()),
                IS평균=float(r[r.index < C.SPLIT].mean()), OOS평균=float(r[r.index >= C.SPLIT].mean()), OOS_PF=float(r[(r.index >= C.SPLIT) & (r > 0)].sum() / max(-r[(r.index >= C.SPLIT) & (r < 0)].sum(), 1e-9)))
Pv = dict(P); Pv.update(stop=0.07, trail=0.10, maxhold=40)
box = lambda f, n: f.h.rolling(n).max().shift(1) / f.l.rolling(n).min().shift(1) - 1
bh = lambda f, n: f.h.rolling(n).max().shift(1)
lv60 = lambda f: bh(f, 60); lv252 = lambda f: f.ch252
bt = []
bt.append(run(lambda f: f.c > bh(f, 60), lv60, Pv, "ep02④ 60일 박스 돌파(전체)"))
bt.append(run(lambda f: (f.c > bh(f, 60)) & (f.vr >= 2), lv60, Pv, "ep02④ 박스 돌파 + 거래량 2배"))
bt.append(run(lambda f: (f.c > bh(f, 60)) & (f.vr >= 2) & (f.uw < 0.2), lv60, Pv, "ep02④ 박스 돌파 + 거래량 2배 + 윗꼬리<20%"))
bt.append(run(lambda f: f.c > f.ch252, lv252, Pv, "ep05 52주 신고가(전체)"))
bt.append(run(lambda f: (f.c > f.ch252) & (f.vr >= 2), lv252, Pv, "ep05 신고가 + 거래량 2배"))
bt.append(run(lambda f: (f.c > f.ch252) & (f.vr >= 2) & (box(f, 40) <= 0.25) & (f.c.shift(1) / f.c.shift(6) - 1 <= 0.10), lv252, Pv, "ep05 신고가 + 거래량2배 + 횡보박스 + 급등아님(영상 풀조건)"))
bt.append(run(lambda f: (f.c > f.ch252) & (f.c.shift(1) / f.c.shift(6) - 1 >= 0.15), lv252, Pv, "ep05 신고가 + 5일 +15%↑ 급등 후 추격(영상이 경고)"))
bt.append(run(lambda f: ((box(f, 60) <= 0.25).shift(1) & (f.v.rolling(10).mean().shift(1) / f.v.rolling(60).mean().shift(1) >= 1.2)) & (f.c > bh(f, 60)) & (f.vr >= 2), lv60, Pv, "ep06 이상적 조합(횡보+거래량증가+돌파+거래량2배)"))
# 손절 유무 비교 (ep24·26: 사전 손절 규칙)
Pn = dict(P); Pn.update(stop=0.99, trail=0.99, maxhold=40)
fullc = lambda f: (f.c > f.ch252) & (f.vr >= 2)
bt.append(run(fullc, None, Pn, "[손절 비교] 신고가+거래량2배, 손절·추적손절 없음(40일 보유)", boxfail=False))
bt.append(run(fullc, None, Pv, "[손절 비교] 신고가+거래량2배, 손절7%·추적10%(돌파선 이탈 청산 제외)", boxfail=False))
# 대조군
ctr = [run(None, None, Pv, "대조군", boxfail=False, rand=np.random.default_rng(s)) for s in range(10)]
cdf = pd.DataFrame(ctr)
bt.append(dict(전략="대조군: 무작위 진입(10회 평균), 손절7%·추적10%·40일", n=float(cdf.n.mean()), 승률=float(cdf.승률.mean()), 평균=float(cdf.평균.mean()), 중앙=float(cdf.중앙.mean()), PF=float(cdf.PF.mean()), 평균보유일=float(cdf.평균보유일.mean()), IS평균=float(cdf.IS평균.mean()), OOS평균=float(cdf.OOS평균.mean()), OOS_PF=float(cdf.OOS_PF.mean())))
res["breakout_backtest"] = bt
json.dump(res, open("ep_gap_trade_results.json", "w"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 250)
print("전체 평균", res["gap_base"])
for k in ("gap_bins", "gapdown_checklist", "gapup_shapes", "open_trade", "breakout_backtest"):
    print("\n##", k); print(pd.DataFrame(res[k]).round(4).to_string())
