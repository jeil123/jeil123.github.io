"""ep04 시장 바닥 5신호(지수), ep10 폭락장 분할매수·VIX, ep02-⑤ 시장 과열, ep07 미국장→국내 갭 예측력."""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import ep_common as C

data, idx = C.load_all()
K, Q, SPX, VIX = idx["KOSPI"].copy(), idx["KOSDAQ"].copy(), idx["SPX"].copy(), idx["VIX"].copy()
res = {}
def fwd(d, ns=(20, 60, 120, 250)):
    for n in ns: d[f"f{n}"] = d["Close"].shift(-n) / d["Close"] - 1
    return d
K, Q, SPX = fwd(K), fwd(Q), fwd(SPX)
for d in (K, Q, SPX):
    d["ret"] = d["Close"].pct_change(); d["dd"] = d["Close"] / d["Close"].rolling(252, min_periods=60).max() - 1
    d["vol60p"] = d["Volume"].rolling(60).mean().shift(1); d["vr"] = d["Volume"] / d["vol60p"]

def ev(d, mask, gap=20, label=""):
    m = C.decluster(mask.fillna(False), gap); e = d[m]
    out = dict(조건=label, n=int(len(e)))
    for n in (20, 60, 120, 250):
        s = e[f"f{n}"].dropna(); b = d[f"f{n}"].dropna()
        if len(s): out[f"f{n}"] = float(s.mean()); out[f"b{n}"] = float(b.mean()); out[f"win{n}"] = float((s > 0).mean())
    out["dates"] = [str(x.date()) for x in e.index[:60]]
    return out

# ---------- ep04 시장 바닥 5신호 ----------
brd = []
for t, d in data.items():
    c = d["Close"]; brd.append((c <= c.rolling(60).min()).astype(float).rename(t))
breadth = pd.concat(brd, axis=1).mean(axis=1)   # 60일 신저가 종목 비율 (검증 종목군)
rows = []
for name, d in (("KOSPI", K), ("KOSDAQ", Q)):
    l20 = d["Low"].rolling(20).min().shift(1)
    b1 = (d["Low"] <= l20) & (d["Close"] > d["Open"]) & (d["Close"] / d["Low"] - 1 >= 0.02) & (d["dd"] <= -0.10)
    rows.append(ev(d, b1, 20, f"{name} ①악재에 안 밀림: 20일 신저가 터치 후 종가>시가, 저점대비 +2%↑ (낙폭 -10%↓)"))
    cap = (d["Volume"] >= 2 * d["vol60p"]) & (d["ret"] <= -0.02) & (d["dd"] <= -0.10)
    # ② 투매 후 재하락 때 거래량 고갈 (W자): 5~30거래일 전 투매일 존재, 오늘 저점이 투매일 저점 ±3% 이내, 5일 평균거래량 ≤ 투매일의 70%
    sig = pd.Series(False, index=d.index)
    cidx = np.where(cap.values)[0]; lv = d["Low"].values; vv = d["Volume"].values; v5 = d["Volume"].rolling(5).mean().values
    for i in range(len(d)):
        for j in cidx[(cidx <= i - 5) & (cidx >= i - 30)]:
            if abs(lv[i] / lv[j] - 1) <= 0.03 and v5[i] <= 0.7 * vv[j] and d["Close"].values[i] >= d["Close"].values[j] * 0.97:
                sig.iloc[i] = True; break
    rows.append(ev(d, sig, 20, f"{name} ②투매 후 재하락 시 거래량 고갈(W자 재확인)"))
    if name == "KOSPI":
        bd = breadth.reindex(d.index)
        low60 = d["Close"] <= d["Close"].rolling(60).min()
        s3 = pd.Series(False, index=d.index); last = None
        for i, dt in enumerate(d.index):
            if low60.iloc[i] and not np.isnan(bd.iloc[i]):
                if last is not None and (i - last[0]) <= 120 and (i - last[0]) >= 10 and bd.iloc[i] < 0.8 * last[1]: s3.iloc[i] = True
                last = (i, bd.iloc[i])
        rows.append(ev(d, s3, 20, "KOSPI ③시장폭 다이버전스: 지수 60일 신저가인데 신저가 종목 비율이 직전 지수 저점 대비 20%↓ (2016~)"))
        rows.append(ev(d, b1 & bd.notna() & (bd.rolling(5).max() > 0), 20, "(참고) ①과 동시에 시장폭 데이터가 있는 구간"))
        comb = (b1.rolling(10).max() > 0) & (sig.rolling(10).max() > 0)
        rows.append(ev(d, comb, 20, "KOSPI ①+② 10일 내 동시 발생"))
res["ep04_market_bottom"] = rows
res["base_KOSPI"] = {f"f{n}": float(K[f"f{n}"].mean()) for n in (20, 60, 120, 250)}
res["base_KOSDAQ"] = {f"f{n}": float(Q[f"f{n}"].mean()) for n in (20, 60, 120, 250)}

# ---------- ep10 폭락장 분할매수 (고점 대비 -20/-30/-40%) ----------
c = K["Close"]; runmax = c.cummax(); epi = []
ath_idx = np.where(c.values >= runmax.values)[0]
for a, b in zip(ath_idx[:-1], list(ath_idx[1:]) + [len(c)]):
    if b - a < 40: continue
    seg = c.iloc[a:b]; peak = seg.iloc[0]; trough = seg.min()
    if trough / peak > 0.80: continue
    tr = {}
    for lvl in (0.8, 0.7, 0.6):
        hit = np.where(seg.values <= peak * lvl)[0]
        tr[lvl] = (seg.index[hit[0]], seg.iloc[hit[0]]) if len(hit) else None
    end_val = c.iloc[b] if b < len(c) else c.iloc[-1]
    row = dict(고점일=str(seg.index[0].date()), 고점=float(peak), 저점낙폭=float(trough / peak - 1), 저점일=str(seg.idxmin().date()), 다음고점회복일=str(c.index[b].date()) if b < len(c) else "미회복")
    for lvl, v in tr.items():
        if v:
            pos = c.index.get_loc(v[0]); row[f"-{round((1-lvl)*100)}%진입일"] = str(v[0].date())
            row[f"-{round((1-lvl)*100)}%후12개월"] = float(c.iloc[min(pos + 250, len(c) - 1)] / v[1] - 1) if pos + 250 < len(c) else None
            row[f"-{round((1-lvl)*100)}%후끝까지"] = float(end_val / v[1] - 1)
    got = [v for v in tr.values() if v]
    if got:
        avg_cost = np.mean([v[1] for v in got]); row["분할평균단가/-20%단가"] = float(avg_cost / tr[0.8][1]) if tr[0.8] else None
        row["진입횟수"] = len(got); row["분할매수 끝까지수익"] = float(end_val / avg_cost - 1)
        if tr[0.8]: row["-20%일시매수 끝까지수익"] = float(end_val / tr[0.8][1] - 1)
    epi.append(row)
res["ep10_staged_buy_KOSPI"] = epi
# 최대 상승일은 최대 하락일 직후에 몰린다?
for nm, d in (("KOSPI", K), ("S&P500", SPX)):
    r = d["ret"].dropna(); up = r.nlargest(25).index; dn = r.nsmallest(25).index
    pos = {dt: i for i, dt in enumerate(r.index)}
    near = sum(any(0 < pos[u] - pos[x] <= 10 for x in dn) for u in up)
    pr = np.mean([any(0 < i - pos[x] <= 10 for x in dn) for i in range(len(r))])
    near_any = sum(any(abs(pos[u] - pos[x]) <= 10 for x in dn) for u in up)
    res.setdefault("ep10_best_days", {})[nm] = dict(상위25상승일_중_하락상위25일_직후10일내=int(near), 무작위기대=float(pr * 25), 전후10일내=int(near_any))
# VIX 정점 후 하락(공포 지수 고점 낮아짐) → 이후 S&P·KOSPI
vx = VIX["Close"]; spike = vx.rolling(20).max() >= 30
vsig = spike & (vx <= 0.8 * vx.rolling(20).max()) & (SPX["Close"].reindex(vx.index) < SPX["Close"].rolling(50).mean().reindex(vx.index))
vm = C.decluster(vsig, 30)
spx_e = SPX.reindex(vx.index)[vm]; k_e = K.reindex(vx.index, method="ffill")[vm]
res["ep10_vix_peak_passed"] = dict(n=int(vm.sum()), dates=[str(x.date()) for x in vx.index[vm]][:60],
    SPX={f"f{n}": float(spx_e[f"f{n}"].mean()) for n in (20, 60, 120)}, SPX_base={f"f{n}": float(SPX[f"f{n}"].mean()) for n in (20, 60, 120)},
    SPX_win60=float((spx_e["f60"] > 0).mean()))

# ---------- ep02-⑤ 시장 과열 (신용잔고는 데이터 없음 → 지수 위치·거래량·급등으로 근사) ----------
ov = []
for nm, d in (("KOSPI", K), ("KOSDAQ", Q), ("S&P500", SPX)):
    near = d["Close"] >= 0.98 * d["Close"].rolling(252).max()
    vol = d["Volume"].rolling(5).mean() >= 1.5 * d["vol60p"]
    surge = d["Close"] / d["Close"].shift(20) - 1 >= 0.08
    ov.append(ev(d, near & vol & surge, 20, f"{nm} 과열 근사: 52주고점 2%이내 & 5일거래량 1.5배 & 20일 +8%↑"))
    ov.append(ev(d, near & surge, 20, f"{nm} 52주고점 근접 & 20일 +8%↑"))
res["ep02_overheat"] = ov

# ---------- ep07 미국장 → 국내 갭 ----------
gaps = pd.concat([pd.Series(f.gap, name=t) for t, f in {t: C.feats(d, K["Close"]) for t, d in data.items()}.items()], axis=1)
mkt_gap = gaps.mean(axis=1).dropna()
us = SPX["ret"].dropna(); ndx = idx["NDX"]["Close"].pct_change().dropna()
def align(s, dates): return s.reindex(pd.DatetimeIndex(dates) - pd.Timedelta(days=1), method="ffill").set_axis(dates)
usr = align(us, mkt_gap.index); nr = align(ndx, mkt_gap.index)
df = pd.DataFrame(dict(gap=mkt_gap, us=usr, ndx=nr)).dropna()
o = []
for lab, sel in (("미국 S&P500 +1%↑", df.us >= 0.01), ("미국 -1%↓", df.us <= -0.01), ("미국 -0.3~+0.3%", df.us.abs() < 0.003), ("미국 +0.3~+1%", (df.us >= 0.003) & (df.us < 0.01)), ("미국 -1~-0.3%", (df.us <= -0.003) & (df.us > -0.01))):
    s = df[sel]; o.append(dict(조건=lab, n=int(len(s)), 국내갭_평균=float(s.gap.mean()), 갭상승비율=float((s.gap > 0).mean())))
big = df[df.us.abs() >= 0.003]
res["ep07_us_overnight"] = dict(bins=o, 방향적중률_미국변동0p3이상=float((np.sign(big.us) == np.sign(big.gap)).mean()), 방향적중률_전체=float((np.sign(df.us) == np.sign(df.gap)).mean()),
    n=int(len(df)), 상관계수=float(df.us.corr(df.gap)), 나스닥상관=float(df.ndx.corr(df.gap)), 기간=[str(df.index[0].date()), str(df.index[-1].date())])
# 종목 단위 적중률: 미국 변동 절대값 1% 이상인 날, 개별 종목 갭 방향이 미국 방향과 같은 비율
G = gaps.reindex(df.index); sel_up = df.us >= 0.01; sel_dn = df.us <= -0.01
res["ep07_stock_level"] = dict(미국1pct이상날_종목갭상승비율=float((G[sel_up] > 0).stack().mean()), 미국마이너스1pct날_종목갭하락비율=float((G[sel_dn] < 0).stack().mean()),
    미국1pct이상날_종목갭3pct이상비율=float((G[sel_up] >= 0.03).stack().mean()), 전체_종목갭상승비율=float((G > 0).stack().mean()),
    미국0p3이상_종목방향적중률=float(np.mean([(np.sign(G.loc[d].dropna()) == np.sign(df.us[d])).mean() for d in df.index[df.us.abs() >= 0.003]])))
print(res["ep07_stock_level"])
json.dump(res, open("ep_index_results.json", "w"), ensure_ascii=False, indent=1, default=float)
pd.set_option("display.width", 250)
for k in ("ep04_market_bottom", "ep02_overheat"):
    print("\n##", k); print(pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != "dates"} for r in res[k]]).round(3).to_string())
print("\n## ep10 staged"); print(pd.DataFrame(res["ep10_staged_buy_KOSPI"]).round(3).T.to_string())
print(res["ep10_best_days"]); print({k: v for k, v in res["ep10_vix_peak_passed"].items() if k != "dates"}); print(json.dumps(res["ep07_us_overnight"], ensure_ascii=False, indent=1))
print(res["base_KOSPI"], res["base_KOSDAQ"])
