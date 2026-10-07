"""ep03·09·12·13·15~20 수급 규칙 검증. 네이버 증권 일별 투자자 순매수량(외국인·기관·개인), 외국인 보유율 사용.
주의: 네이버 '기관'에는 금융투자(증권사 자기매매·ETF LP 등)가 포함된다(ep03가 경고한 바로 그 항목)."""
import os, json, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import ep_common as C

data, idx = C.load_all()
raw = pickle.load(open(os.path.join(C.CACHE, "flows.pkl"), "rb"))
def num(s): return pd.to_numeric(s.astype(str).str.replace(r"[,%+]", "", regex=True), errors="coerce")
frames = {}
for t, d in data.items():
    code = t.split(".")[0]
    if code not in raw or not raw[code]: continue
    r = pd.DataFrame(raw[code]); r["date"] = pd.to_datetime(r["bizdate"]); r = r.drop_duplicates("date").set_index("date").sort_index()
    f = C.feats(d, (idx["KOSDAQ"] if t.endswith(".KQ") else idx["KOSPI"])["Close"])
    f["fq"] = num(r["foreignerPureBuyQuant"]).reindex(f.index); f["oq"] = num(r["organPureBuyQuant"]).reindex(f.index)
    f["iq"] = num(r["individualPureBuyQuant"]).reindex(f.index); f["hold"] = num(r["foreignerHoldRatio"]).reindex(f.index)
    f["nv"] = num(r["accumulatedTradingVolume"]).reindex(f.index)
    f = f[f.fq.notna() & f.oq.notna()]
    if len(f) < 500: continue
    frames[t] = f
F = list(frames.values())
print("종목", len(F), "기간", min(f.index[0] for f in F).date(), max(f.index[-1] for f in F).date())

def streak(pos):  # 연속 True 일수
    g = (~pos).cumsum(); return pos.groupby(g).cumsum()
def box(f, n): return f.h.rolling(n).max().shift(1) / f.l.rolling(n).min().shift(1) - 1
tests = {}
fr = lambda f: f.fq / f.nv; orr = lambda f: f.oq / f.nv
fs = lambda f: streak(f.fq > 0); os_ = lambda f: streak(f.oq > 0)
# ep12·13 외국인
tests["ep12·19 외국인 순매수 3일 연속(첫 도달일)"] = lambda f: fs(f) == 3
tests["ep13 외국인 순매수 5일 연속(첫 도달일)"] = lambda f: fs(f) == 5
tests["ep13 외국인 순매수 3일↑ + 보유율 5일간 상승"] = lambda f: (fs(f) >= 3) & (f.hold - f.hold.shift(5) > 0)
tests["ep13 외국인 순매수/거래량 ≥10%인 날 5일 중 3회↑"] = lambda f: (fr(f) >= 0.10).rolling(5).sum() >= 3
tests["ep12 체크리스트 3개↑(연속3일·보유율↑·횡보/바닥·기관동반·강도)"] = lambda f: ((fs(f) >= 3).astype(int) + (f.hold - f.hold.shift(5) > 0).astype(int) + ((box(f, 40) <= 0.25) | (f.dd252 <= -0.20)).astype(int) + (f.oq.rolling(3).sum() > 0).astype(int) + ((fr(f) >= 0.10).rolling(5).sum() >= 2).astype(int)) >= 3
tests["ep12 체크리스트 4개↑"] = lambda f: ((fs(f) >= 3).astype(int) + (f.hold - f.hold.shift(5) > 0).astype(int) + ((box(f, 40) <= 0.25) | (f.dd252 <= -0.20)).astype(int) + (f.oq.rolling(3).sum() > 0).astype(int) + ((fr(f) >= 0.10).rolling(5).sum() >= 2).astype(int)) >= 4
tests["ep12 외국인·기관 동반 순매수 3일 연속"] = lambda f: streak((f.fq > 0) & (f.oq > 0)) == 3
tests["ep12 외국인 순매수 3일↑ & 신고가 부근(고점)"] = lambda f: (fs(f) >= 3) & (f.c >= 0.95 * f.ch252)
tests["ep12 외국인 순매수 3일↑ & 횡보/바닥(고점대비 -20%↓)"] = lambda f: (fs(f) >= 3) & (f.dd252 <= -0.20)
# ep19 HTS 필터 (금액 임계: 기관 50억, 외국인 30억, 거래대금 전일비 150%, 3일 연속)
tests["ep19 HTS필터(기관≥50억·외국인≥30억·거래대금150%·3일연속 동반)"] = lambda f: (f.oq * f.c >= 5e9) & (f.fq * f.c >= 3e9) & (f.tv / f.tv.shift(1) >= 1.5) & (streak((f.fq > 0) & (f.oq > 0)) >= 3)
tests["ep19 기관 순매수 50억↑ & 외국인 30억↑ (당일)"] = lambda f: (f.oq * f.c >= 5e9) & (f.fq * f.c >= 3e9)
tests["ep19 외국인 5일 누적 순매수 ≥ 거래대금 평균의 1일치"] = lambda f: (f.fq.rolling(5).sum() * f.c) >= (f.tv.rolling(20).mean().shift(1))
# ep18 조용한 매집
tests["ep18 횡보(60일 박스≤25%) + 보유율 40일 +0.5%p↑"] = lambda f: (box(f, 60) <= 0.25) & (f.hold - f.hold.shift(40) >= 0.5)
tests["ep18 지수 하락일(-1%↓) 외국인 순매수 2회↑(20일)"] = lambda f: ((f.bret <= -0.01) & (f.fq > 0)).rolling(20).sum() >= 2
tests["ep18 횡보 + 보유율↑ + 하락일 순매수"] = lambda f: (box(f, 60) <= 0.25) & (f.hold - f.hold.shift(40) >= 0.5) & (((f.bret <= -0.01) & (f.fq > 0)).rolling(20).sum() >= 1)
# ep14·17 기관 매집
tests["ep14 기관 20일 누적 순매수 + 횡보(40일 박스≤20%)"] = lambda f: (f.oq.rolling(20).sum() > 0) & (box(f, 40) <= 0.20)
tests["ep14 기관 순매수 3일 연속"] = lambda f: os_(f) == 3
tests["ep17 기관·외국인 20일 누적 순매수 + 지수 하락일 순매수 유지"] = lambda f: (f.oq.rolling(20).sum() > 0) & (f.fq.rolling(20).sum() > 0) & (((f.bret <= -0.01) & ((f.oq + f.fq) > 0)).rolling(20).sum() >= 2)
# ep15·16 기관 매도 초기
tests["ep15 오르는 날 기관 순매도(10일 중 5회↑) + 기관 20일 누적 순매도"] = lambda f: (((f.ret > 0) & (f.oq < 0)).rolling(10).sum() >= 5) & (f.oq.rolling(20).sum() < 0)
tests["ep15 기관·외국인 20일 동반 순매도"] = lambda f: (f.oq.rolling(20).sum() < 0) & (f.fq.rolling(20).sum() < 0)
tests["ep16 기관 순매도 + 개인 순매수 20일 누적(손바뀜) + 고점권(60일고점 -10%이내)"] = lambda f: (f.oq.rolling(20).sum() < 0) & (f.iq.rolling(20).sum() > 0) & (f.c >= 0.9 * f.h.rolling(60).max())
# ep03·ep20 외국인 보유율, 외국인 사는데 하락
tests["ep03 외국인 보유율 60일 -1%p↓ (삼성전자식 이탈)"] = lambda f: (f.hold - f.hold.shift(60) <= -1.0)
tests["ep03 외국인 보유율 60일 +1%p↑"] = lambda f: (f.hold - f.hold.shift(60) >= 1.0)
tests["ep20 외국인 5일 순매수인데 주가 -5%↓"] = lambda f: (f.fq.rolling(5).sum() > 0) & (f.c / f.c.shift(5) - 1 <= -0.05)
tests["ep20 외국인 순매수 + 하락폭 축소(최근10일 평균하락 ≤ 직전10일 60%)"] = lambda f: (f.fq.rolling(10).sum() > 0) & (f.ret.where(f.ret < 0).rolling(10, min_periods=3).mean().abs() <= 0.6 * f.ret.where(f.ret < 0).shift(10).rolling(10, min_periods=3).mean().abs()) & (f.dd252 <= -0.15)
# ep09 저가 우량주(가격·수급 부분만): 52주 신저가 ±10% + 외국인 3일 순매수 + 거래량 1.5배
tests["ep09 52주 신저가 10%이내 + 외국인 3일 순매수 + 거래량 1.5배"] = lambda f: (f.c <= 1.10 * f.l.rolling(252).min()) & (fs(f) >= 3) & (f.vr >= 1.5)
tests["ep09 52주 신저가 10%이내 (가격만)"] = lambda f: (f.c <= 1.10 * f.l.rolling(252).min())

res = {}
for k, fn in tests.items():
    try: res[k] = C.ev_stats(F, fn, gap=10)
    except Exception as ex: res[k] = dict(error=str(ex))
allf = pd.concat(F); res["_base"] = {h: float(allf[h].mean()) for h in ("f5", "f20", "f60")}; res["_base"]["ex20"] = float(allf.ex20.mean())
res["_meta"] = dict(종목수=len(F), 기간=[str(min(f.index[0] for f in F).date()), str(max(f.index[-1] for f in F).date())])
json.dump(res, open("ep_flow_results.json", "w"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 250)
rows = []
for k, v in res.items():
    if k.startswith("_"): continue
    if "error" in v or v.get("n", 0) < 5: print(k, v); continue
    rows.append(dict(조건=k[:56], n=v["n"], f5=v.get("f5_mean"), f20=v.get("f20_mean"), diff20=v.get("f20_diff"), t=v.get("f20_t"), 승률=v.get("f20_win"), ex20=v.get("ex20_mean"), ex20_t=v.get("ex20_t"), IS=v.get("f20_diff_IS"), OOS=v.get("f20_diff_OOS")))
print(pd.DataFrame(rows).round(4).to_string()); print(res["_base"], res["_meta"])
