"""ep02·05·06·14~20 종목 단위 이벤트 스터디. 조건식은 결과를 보기 전에 고정."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import ep_common as C

data, idx = C.load_all()
frames = {}
for t, d in data.items():
    ib = (idx["KOSDAQ"] if t.endswith(".KQ") else idx["KOSPI"])["Close"]
    frames[t] = C.feats(d, ib)
F = list(frames.values())

def S(f, a, b):  # 안전한 비율
    return a / b

# ---------- 조건식 ----------
def box(f, n): return f.h.rolling(n).max().shift(1) / f.l.rolling(n).min().shift(1) - 1
def bh(f, n): return f.h.rolling(n).max().shift(1)
tests = {}

# ep02 매수 함정 5곳
tests["ep02-①급등추격(3일+12%·갭2%↑·거래량3배)"] = lambda f: (f.c / f.c.shift(3) - 1 >= 0.12) & (f.gap >= 0.02) & (f.vr >= 3)
tests["ep02-①급등추격+윗꼬리40%↑"] = lambda f: (f.c / f.c.shift(3) - 1 >= 0.12) & (f.gap >= 0.02) & (f.vr >= 3) & (f.uw >= 0.4)
tests["ep02-①갭상승3%↑+거래량2배+윗꼬리40%↑"] = lambda f: (f.gap >= 0.03) & (f.vr >= 2) & (f.uw >= 0.4)
tests["ep02-③고점대비-30%이하"] = lambda f: f.dd252 <= -0.30
tests["ep02-③고점대비-50%이하(반토막)"] = lambda f: f.dd252 <= -0.50
tests["ep02-④박스돌파(60일고가) 전체"] = lambda f: f.c > bh(f, 60)
tests["ep02-④박스돌파+거래량2배↑"] = lambda f: (f.c > bh(f, 60)) & (f.vr >= 2)
tests["ep02-④박스돌파+거래량<1.2배(슬쩍)"] = lambda f: (f.c > bh(f, 60)) & (f.vr < 1.2)
tests["ep02-④박스돌파+거래량2배↑+윗꼬리40%↑(거짓돌파 의심)"] = lambda f: (f.c > bh(f, 60)) & (f.vr >= 2) & (f.uw >= 0.4)
tests["ep02-④박스돌파+거래량2배↑+윗꼬리<20%"] = lambda f: (f.c > bh(f, 60)) & (f.vr >= 2) & (f.uw < 0.2)

# ep05 신고가 / ep19 (신고가 근접은 여력 제한?)
nh = lambda f: f.c > f.ch252
tests["ep05-52주 신고가 전체"] = nh
tests["ep05-신고가+거래량2배↑"] = lambda f: nh(f) & (f.vr >= 2)
tests["ep05-신고가+직전40일 박스≤25%"] = lambda f: nh(f) & (box(f, 40) <= 0.25)
tests["ep05-신고가+직전5일 급등≤10%"] = lambda f: nh(f) & (f.c.shift(1) / f.c.shift(6) - 1 <= 0.10)
tests["ep05-신고가+거래량2배+박스≤25%+급등아님"] = lambda f: nh(f) & (f.vr >= 2) & (box(f, 40) <= 0.25) & (f.c.shift(1) / f.c.shift(6) - 1 <= 0.10)
tests["ep05-신고가+며칠 급등 후(5일+15%↑) 추격"] = lambda f: nh(f) & (f.c.shift(1) / f.c.shift(6) - 1 >= 0.15)
tests["ep19-52주 신고가 근접(3%이내, 미돌파)"] = lambda f: (f.c >= 0.97 * f.ch252) & (f.c <= f.ch252)
tests["ep19-거래대금 전일비150%↑+3일연속 상승"] = lambda f: (f.tv / f.tv.shift(1) >= 1.5) & (f.ret > 0) & (f.ret.shift(1) > 0) & (f.ret.shift(2) > 0)

# ep06 폭등 전 조건 (상태)
sideways = lambda f: box(f, 60) <= 0.25
tests["ep06-①바닥 횡보(60일 박스≤25%)"] = sideways
tests["ep06-①횡보+가격제자리인데 거래량 증가(10일/60일≥1.2, 20일수익±5%)"] = lambda f: sideways(f) & (f.v.rolling(10).mean() / f.v.rolling(60).mean() >= 1.2) & (f.c / f.c.shift(20) - 1).abs().le(0.05)
tests["ep06-④이평 밀집→정배열 전환(5>20>60, 밀집 5%이내 직전)"] = lambda f: (f.ma5 > f.ma20) & (f.ma20 > f.ma60) & ~((f.ma5.shift(3) > f.ma20.shift(3)) & (f.ma20.shift(3) > f.ma60.shift(3))) & (((pd.concat([f.ma5, f.ma20, f.ma60], axis=1).shift(5).max(axis=1) - pd.concat([f.ma5, f.ma20, f.ma60], axis=1).shift(5).min(axis=1)) / f.c.shift(5)) <= 0.05)
tests["ep06-이상적 조합(횡보+거래량증가+박스돌파+거래량2배)"] = lambda f: ((sideways(f).shift(1)) & (f.v.rolling(10).mean().shift(1) / f.v.rolling(60).mean().shift(1) >= 1.2)) & (f.c > bh(f, 60)) & (f.vr >= 2)

# ep14·17 매집 (가격·거래량 단서)
dnv = lambda f: f.v.where(f.ret < 0).rolling(20, min_periods=3).mean()
upv = lambda f: f.v.where(f.ret > 0).rolling(20, min_periods=3).mean()
flat = lambda f: (box(f, 40) <= 0.20)
tests["ep14-③하락일 거래량 ≤ 상승일 80% (20일)"] = lambda f: dnv(f) <= 0.8 * upv(f)
tests["ep14-①횡보+거래량 증가 (상승·하락 비대칭 결합)"] = lambda f: flat(f) & (dnv(f) <= 0.8 * upv(f)) & (f.v.rolling(10).mean() / f.v.rolling(60).mean() >= 1.1)
def support4(f):
    low40 = f.l.rolling(40).min()
    touch = ((f.l <= low40 * 1.015) & (f.lw >= 0.4)).astype(int)
    return touch.rolling(40).sum() >= 3
tests["ep14-④가격대 반복 지지(40일 중 3회↑ 저점부근 긴 아래꼬리)"] = support4
tests["ep17-저점 상승(최근10일 저점 > 직전20일 저점 +1%)"] = lambda f: f.l.rolling(10).min() >= 1.01 * f.l.shift(10).rolling(20).min()
tests["ep17-지수 하락일(-1%↓)에 안 빠짐 2회↑(20일)"] = lambda f: ((f.bret <= -0.01) & (f.ret >= 0)).rolling(20).sum() >= 2
tests["ep17-지수 하락일에도 아래꼬리 회복(종가>시가) 2회↑"] = lambda f: ((f.bret <= -0.01) & (f.oc > 0) & (f.lw >= 0.3)).rolling(20).sum() >= 2

# ep15·16 분산(기관 매도 초기) 차트 신호
tests["ep16-①윗꼬리 반복(10일 중 3회↑, 꼬리≥50%·변동≥2%)"] = lambda f: ((f.uw >= 0.5) & (f.rng >= 0.02)).rolling(10).sum() >= 3
tests["ep16-②반등 고점 하락(최근10일 고점 < 직전20일 고점)"] = lambda f: f.h.rolling(10).max() < f.h.shift(10).rolling(20).max()
tests["ep16-③고점권 횡보+거래량 증가(60일고점 5%이내, 10일변동≤3%, 거래량1.3배)"] = lambda f: (f.c >= 0.95 * f.h.rolling(60).max()) & ((f.c / f.c.shift(10) - 1).abs() <= 0.03) & (f.v.rolling(10).mean() / f.v.rolling(60).mean() >= 1.3)
tests["ep15-②반등일 거래량↑·종가 밀림(5일 중 2회↑ 거래량1.5배·종가<시가)"] = lambda f: ((f.vr >= 1.5) & (f.h > f.o * 1.01) & (f.oc < 0)).rolling(5).sum() >= 2
tests["ep15-③지수 대비 상대약세(20일 -5%p↓, 지수 상승)"] = lambda f: ((f.c / f.c.shift(20) - 1) - (f.bret.add(1).rolling(20).apply(np.prod, raw=True) - 1) <= -0.05) & ((f.bret.add(1).rolling(20).apply(np.prod, raw=True) - 1) > 0)

# ep20 하락폭 축소(물량 교체)
neg = lambda f: f.ret.where(f.ret < 0)
tests["ep20-하락폭 축소(최근10일 평균하락 ≤ 직전10일의 60%, 고점대비 -15%↓)"] = lambda f: (neg(f).rolling(10, min_periods=3).mean().abs() <= 0.6 * neg(f).shift(10).rolling(10, min_periods=3).mean().abs()) & (f.dd252 <= -0.15)

res = {}
for k, fn in tests.items():
    try:
        res[k] = C.ev_stats(F, fn, gap=10 if "ep02-③" not in k else 20)
    except Exception as ex:
        res[k] = dict(error=str(ex))
allf = pd.concat(F)
res["_base"] = {h: float(allf[h].mean()) for h in ("f5", "f20", "f60")}
res["_base"]["ex20"] = float(allf["ex20"].mean())
# 거짓 돌파율(돌파 후 5일 내 종가가 돌파선 아래로 복귀)
def false_rate(fn, level=lambda f: bh(f, 60)):
    n = k = 0
    for f in F:
        m = C.decluster(fn(f).fillna(False).astype(bool), 5)
        lv = level(f); c = f.c
        mn = pd.concat([c.shift(-i) for i in range(1, 6)], axis=1).min(axis=1)
        sel = m & mn.notna()
        n += int(sel.sum()); k += int((mn[sel] < lv[sel]).sum())
    return n, k / max(n, 1)
res["_false_breakout"] = {name: dict(zip(("n", "rate"), false_rate(fn))) for name, fn in [
    ("박스돌파 전체", tests["ep02-④박스돌파(60일고가) 전체"]),
    ("거래량2배↑", tests["ep02-④박스돌파+거래량2배↑"]),
    ("거래량<1.2배", tests["ep02-④박스돌파+거래량<1.2배(슬쩍)"]),
    ("거래량2배↑+윗꼬리40%↑", tests["ep02-④박스돌파+거래량2배↑+윗꼬리40%↑(거짓돌파 의심)"]),
    ("거래량2배↑+윗꼬리<20%", tests["ep02-④박스돌파+거래량2배↑+윗꼬리<20%"])]}
json.dump(res, open("ep_stock_results.json", "w"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 250)
rows = []
for k, v in res.items():
    if k.startswith("_") or "error" in v or v.get("n", 0) == 0: print(k, v if "error" in v else "n=0"); continue
    rows.append(dict(조건=k[:60], n=v["n"], f20=v.get("f20_mean"), base=v.get("f20_base"), diff=v.get("f20_diff"), t=v.get("f20_t"), 승률=v.get("f20_win"), ex20=v.get("ex20_mean"), ex20_t=v.get("ex20_t"), f60diff=v.get("f60_diff"), IS=v.get("f20_diff_IS"), OOS=v.get("f20_diff_OOS")))
print(pd.DataFrame(rows).round(4).to_string())
print(res["_base"]); print(json.dumps(res["_false_breakout"], ensure_ascii=False))
