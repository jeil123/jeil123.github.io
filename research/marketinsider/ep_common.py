"""공통 유틸: 데이터 로딩, 종목별 특성(feature) 생성, 이벤트 스터디(월 군집 t통계)."""
import os, pickle, numpy as np, pandas as pd
import ep01_signals as E

CACHE = os.environ.get("MI_CACHE", os.path.dirname(os.path.abspath(__file__)))
SPLIT = pd.Timestamp("2023-01-01")

def load_all():
    data, _ = E.load()
    data = {t: cd for t, d in data.items() if t not in E.EXCLUDE for cd in [E.clean(d)] if cd is not None}
    idx = pickle.load(open(os.path.join(CACHE, "idx_long.pkl"), "rb"))
    return data, idx

def feats(d, ib):
    """d: OHLCV, ib: 벤치마크 지수 종가(일자 인덱스)"""
    o, h, l, c, v = (d[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    f = pd.DataFrame(index=d.index)
    for k, s in zip("ohlcv", (o, h, l, c, v)): f[k] = s
    f["ret"] = c.pct_change(); f["gap"] = o / c.shift(1) - 1; f["oc"] = c / o - 1
    rg = (h - l).replace(0, np.nan)
    f["rng"] = (h - l) / c
    f["uw"] = (h - np.maximum(o, c)) / rg; f["lw"] = (np.minimum(o, c) - l) / rg
    f["vol20p"] = v.rolling(20).mean().shift(1); f["vol60p"] = v.rolling(60).mean().shift(1)
    f["vr"] = v / f["vol20p"]
    for n in (5, 20, 60): f[f"ma{n}"] = c.rolling(n).mean()
    f["hh252"] = h.rolling(252).max().shift(1); f["ch252"] = c.rolling(252).max().shift(1)
    f["dd252"] = c / h.rolling(252).max() - 1
    b = ib.reindex(d.index).ffill()
    f["bret"] = b.pct_change()
    for n in (1, 5, 20, 60):
        f[f"f{n}"] = c.shift(-n) / c - 1
        f[f"bf{n}"] = b.shift(-n) / b - 1
    f["ex20"] = f["f20"] - f["bf20"]; f["ex5"] = f["f5"] - f["bf5"]; f["ex60"] = f["f60"] - f["bf60"]
    f["tv"] = c * v  # 거래대금
    return f

def decluster(mask, gap):
    m = mask.fillna(False).values; out = np.zeros(len(m), bool); last = -10**9
    for i, x in enumerate(m):
        if x and i - last >= gap: out[i] = True; last = i
    return pd.Series(out, index=mask.index)

def clustered_t(diff):
    """diff: 날짜 인덱스 Series(이벤트수익-기본평균). 월 단위로 묶은 t통계."""
    g = diff.groupby(diff.index.to_period("M")).mean()
    if len(g) < 6: return np.nan
    return float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))

def ev_stats(frames, fn, gap=10, hz=("f5", "f20", "f60"), extra=None):
    ev = []
    for f in frames:
        m = fn(f)
        if m is None: continue
        m = m.fillna(False).astype(bool)
        if gap: m = decluster(m, gap)
        e = f[m]
        if len(e): ev.append(e)
    if not ev: return dict(n=0)
    ev = pd.concat(ev)
    allf = pd.concat(frames)
    out = dict(n=int(len(ev)))
    for h in hz:
        e = ev[h].dropna(); base = allf[h].dropna()
        if len(e) < 5: continue
        d = (ev[h] - base.mean()).dropna()
        out[f"{h}_mean"] = float(e.mean()); out[f"{h}_base"] = float(base.mean())
        out[f"{h}_diff"] = float(e.mean() - base.mean()); out[f"{h}_win"] = float((e > 0).mean())
        out[f"{h}_t"] = clustered_t(d)
    e = ev["ex20"].dropna()
    if len(e) >= 5:
        out["ex20_mean"] = float(e.mean()); out["ex20_win"] = float((e > 0).mean())
        out["ex20_t"] = clustered_t(e - allf["ex20"].dropna().mean())
    for lab, sub in (("IS", ev[ev.index < SPLIT]), ("OOS", ev[ev.index >= SPLIT])):
        s = sub["f20"].dropna()
        out[f"f20_diff_{lab}"] = float(s.mean() - allf["f20"].dropna().mean()) if len(s) >= 5 else None
        out[f"n_{lab}"] = int(len(s))
    if extra:
        for k, fnx in extra.items(): out[k] = float(fnx(ev))
    return out
