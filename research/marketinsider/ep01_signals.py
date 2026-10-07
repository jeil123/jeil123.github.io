"""
마켓인사이더 영상 l5soyjp5JJw "주가 급등 전날, 차트에는 이미 이 5가지가 보입니다"
5가지 신호를 일봉 조건식으로 구현하고 (1) 이벤트 스터디 (2) 진입 방식별 백테스트를 수행.
임계값은 결과를 보기 전에 고정(PARAMS). 민감도는 별도 점검.
"""
import os, sys, json, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("MI_CACHE", HERE)
sys.path.insert(0, HERE)
from universe import CODES

START = "2016-01-01"
PARAMS = dict(
    s1_box=0.08, s1_vol=0.6, s1_body=0.015,          # 신호1 거래량 마른 횡보
    s2_wick=0.4, s2_recov=0.6, s2_minrange=0.015, s2_n=2,   # 신호2 하락 시도 실패
    s3_vol=1.5, s3_move=0.02, s3_n=2,                 # 신호3 거래량 증가·가격 정체
    s4_spread=0.05,                                   # 신호4 이평 수렴
    s5_idx=-0.01, s5_excess=0.01, s5_n=2,             # 신호5 약세장 상대강도
    watch_days=5, box_days=10, brk_vol=1.5,           # 돌파 확인
    stop=0.07, trail=0.10, maxhold=20,                # 청산
    fee=0.00015, slip=0.001, tax=0.0020,              # 비용(편도 수수료·슬리피지, 매도세)
    split="2023-01-01", cap=10)

def load():
    pk = os.path.join(CACHE, "prices.pkl")
    if os.path.exists(pk): return pickle.load(open(pk, "rb"))
    import yfinance as yf
    syms = [c + s for c in CODES for s in (".KS", ".KQ")]
    raw = yf.download(syms, start=START, progress=False, auto_adjust=True, group_by="ticker", threads=True)
    data = {}
    for c in CODES:
        best = None
        for s in (".KS", ".KQ"):
            try: d = raw[c + s].dropna(subset=["Close"])
            except KeyError: continue
            if len(d) > 300 and (best is None or len(d) > len(best[1])): best = (c + s, d)
        if best: data[best[0]] = best[1][["Open","High","Low","Close","Volume"]].astype(float)
    idx = {}
    for n, t in (("KOSPI","^KS11"),("KOSDAQ","^KQ11")):
        idx[n] = yf.download(t, start=START, progress=False, auto_adjust=True)["Close"].squeeze()
    out = (data, idx); pickle.dump(out, open(pk, "wb")); return out

def clean(d, min_rows=300):
    """거래량 0일 제거, 일간 변동 ±31% 초과(액면분할·분할상장 등 조정 누락) 이후 구간만 사용."""
    d = d[d["Volume"] > 0].dropna()
    d = d[(d[["Open","High","Low","Close"]] > 0).all(axis=1)]
    r = d["Close"].pct_change().abs()
    bad = np.where(r.values > 0.31)[0]
    if len(bad): d = d.iloc[bad[-1]:]
    return d if len(d) >= min_rows else None

EXCLUDE = {"122630.KS"}  # 레버리지 ETF

def signals(d, bench, P=PARAMS):
    o,h,l,c,v = (d[k] for k in ["Open","High","Low","Close","Volume"])
    ret = c.pct_change()
    box5 = (h.rolling(5).max()-l.rolling(5).min())/c
    vol5, vol60 = v.rolling(5).mean(), v.rolling(60).mean()
    body5 = ((c-o).abs()/c).rolling(5).mean()
    S1 = (box5<=P["s1_box"])&(vol5<=P["s1_vol"]*vol60)&(body5<=P["s1_body"])
    rg = (h-l).replace(0,np.nan)
    lw = (np.minimum(o,c)-l)/rg; rec = (c-l)/rg
    hit = (lw>=P["s2_wick"])&(rec>=P["s2_recov"])&(((h-l)/c)>=P["s2_minrange"])
    S2 = (hit.rolling(5).sum()>=P["s2_n"])&(l.rolling(3).min()>=l.shift(3).rolling(5).min())
    vol20p = v.rolling(20).mean().shift(1)
    stall = (v>=P["s3_vol"]*vol20p)&(ret.abs()<=P["s3_move"])
    S3 = stall.rolling(10).sum()>=P["s3_n"]
    m5,m20,m60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    S4 = ((pd.concat([m5,m20,m60],axis=1).max(axis=1)-pd.concat([m5,m20,m60],axis=1).min(axis=1))/c)<=P["s4_spread"]
    bret = bench.reindex(d.index).ffill().pct_change()
    down = bret<=P["s5_idx"]
    exd = (ret-bret).where(down)
    S5 = (down.rolling(20).sum()>=P["s5_n"])&(exd.rolling(20,min_periods=P["s5_n"]).mean()>=P["s5_excess"])
    sig = pd.concat([S1,S2,S3,S4,S5],axis=1).fillna(False).astype(int); sig.columns=["S1","S2","S3","S4","S5"]
    sig["score"] = sig.sum(axis=1)
    return sig

def forward(d):
    c = d["Close"]
    f5 = c.shift(-5)/c-1; f20 = c.shift(-20)/c-1
    mx5 = pd.concat([c.shift(-k) for k in range(1,6)],axis=1).max(axis=1)/c-1
    mn5 = pd.concat([c.shift(-k) for k in range(1,6)],axis=1).min(axis=1)/c-1
    f1 = c.shift(-1)/c-1
    return pd.DataFrame({"f1":f1,"f5":f5,"f20":f20,"mx5":mx5,"mn5":mn5})

def decluster(mask, gap=10):
    out = np.zeros(len(mask),bool); last=-10**9
    for i,m in enumerate(mask):
        if m and i-last>=gap: out[i]=True; last=i
    return pd.Series(out,index=mask.index)

def event_study(frames):
    rows = []
    allf = pd.concat([f for f in frames], keys=range(len(frames)))
    cond = {"전체(기본확률)": lambda s: pd.Series(True,index=s.index)}
    for k in ["S1","S2","S3","S4","S5"]: cond[k]=(lambda k: lambda s: s[k]==1)(k)
    for k in (2,3,4,5): cond[f"점수>={k}"]=(lambda k: lambda s: s["score"]>=k)(k)
    for name,fn in cond.items():
        parts=[]
        for f in frames:
            m = fn(f).fillna(False).astype(bool)
            if name!="전체(기본확률)": m = decluster(m)
            parts.append(f[m & f["f20"].notna()])
        e = pd.concat(parts)
        if len(e)==0: continue
        rows.append(dict(조건=name, 건수=len(e),
            급등5일10pct=(e["mx5"]>=0.10).mean(), 급락5일10pct=(e["mn5"]<=-0.10).mean(),
            익일상승률=(e["f1"]>0).mean(), 평균5일=e["f5"].mean(), 평균20일=e["f20"].mean(), 중앙20일=e["f20"].median()))
    return pd.DataFrame(rows)

def sim_trade(o,h,l,c,i,boxhigh,P):
    n=len(c); cost_in=P["fee"]+P["slip"]
    px=o[i]*(1+cost_in); stop=o[i]*(1-P["stop"]); peak=c[i]; last=min(i+P["maxhold"]-1,n-1)
    for j in range(i,last+1):
        if l[j]<=stop:
            ex=min(o[j],stop) if j>i else stop; return j,ex
        peak=max(peak,c[j])
        if j==last: return j,c[j]
        if (boxhigh is not None and c[j]<boxhigh) or c[j]<peak*(1-P["trail"]):
            return j+1,o[j+1]
    return last,c[last]

def run_backtest(data,sigs,P,mode,k,rng=None,rand_n=None):
    trades=[]
    for t,d in data.items():
        o,h,l,c,v=(d[x].values for x in ["Open","High","Low","Close","Volume"]); n=len(c)
        s=sigs[t]; sc=s["score"].values>=k
        idx=d.index
        if mode=="A":
            trig=sc & ~np.r_[False,sc[:-1]]
            boxh=None
        elif mode=="B":
            wl=pd.Series(sc,index=idx).astype(int).rolling(P["watch_days"]).max().shift(1).fillna(0).values==1
            bh=pd.Series(h,index=idx).rolling(P["box_days"]).max().shift(1).values
            vv=pd.Series(v,index=idx).rolling(20).mean().shift(1).values
            trig=wl&(c>bh)&(v>=P["brk_vol"]*vv)&~np.isnan(bh)
            boxh=bh
        else: # R: 무작위 진입 (동일 청산, 대조군)
            trig=rng.random(n)<rand_n; boxh=None
        for i in np.where(trig)[0]:
            if i+1>=n or np.isnan(o[i+1]): continue
            ent=i+1; j,ex=sim_trade(o,h,l,c,ent,boxh[i] if boxh is not None else None,P)
            px=o[ent]*(1+P["fee"]+P["slip"]); net=ex*(1-P["fee"]-P["slip"]-P["tax"])/px-1
            trades.append(dict(t=t,px=px,entry=idx[ent],exit=idx[min(j,n-1)],ret=net,hold=j-ent+1,
                               vr=(v[i]/ (pd.Series(v).rolling(20).mean().shift(1).values[i]+1)) if mode=="B" else 0))
    return pd.DataFrame(trades)

def stats(tr,label=""):
    if len(tr)==0: return dict(구간=label,거래수=0)
    r=tr["ret"]; w=r[r>0].sum(); lo=-r[r<0].sum()
    return dict(구간=label,거래수=len(tr),승률=(r>0).mean(),평균수익=r.mean(),중앙수익=r.median(),
                손익비PF=(w/lo if lo>0 else np.inf),평균보유일=tr["hold"].mean(),
                최대손실=r.min(),최대이익=r.max())

def portfolio(tr,data,P,cap):
    """동시 보유 cap종목 동일가중(자산/cap), 우선순위=돌파 거래량배수. 일별 종가 평가손익 기준."""
    if len(tr)==0: return None
    dates=sorted(set().union(*[d.index for d in data.values()]))
    closes={t:d["Close"] for t,d in data.items()}
    tr=tr.sort_values(["entry","vr"],ascending=[True,False])
    pend=list(tr.itertuples()); pi=0; cash=1.0; pos=[]; curve=[]
    for dt in dates:
        keep=[]
        for p in pos:
            if p["exit"]<=dt: cash+=p["size"]*(1+p["ret"])
            else: keep.append(p)
        pos=keep
        while pi<len(pend) and pend[pi].entry<=dt:
            x=pend[pi]; pi+=1
            if x.entry==dt and len(pos)<cap:
                eqv=cash+sum(p["val"] for p in pos)
                size=min(cash,eqv/cap)
                if size>0: cash-=size; pos.append(dict(size=size,ret=x.ret,exit=x.exit,t=x.t,px=x.px,val=size))
        for p in pos:
            cs=closes[p["t"]]
            if dt in cs.index: p["val"]=p["size"]*cs.at[dt]/p["px"]
        curve.append((dt,cash+sum(p["val"] for p in pos)))
    s=pd.Series(dict(curve)); mdd=((s/s.cummax())-1).min()
    yrs=(s.index[-1]-s.index[0]).days/365.25
    return dict(최종자산배수=float(s.iloc[-1]),연복리=float(s.iloc[-1]**(1/yrs)-1),MDD=float(mdd)),s

def main():
    data,idx=load()
    data={t:cd for t,d in data.items() if t not in EXCLUDE for cd in [clean(d)] if cd is not None}
    print("종목수",len(data))
    sigs={};fwd={};frames=[]
    for t,d in data.items():
        bench=idx["KOSDAQ"] if t.endswith(".KQ") else idx["KOSPI"]
        sigs[t]=signals(d,bench); f=forward(d); fwd[t]=f
        frames.append(pd.concat([f,sigs[t]],axis=1))
    res={"params":PARAMS,"n_stocks":len(data),"start":START}
    ev=event_study(frames); res["event_study"]=ev.to_dict("records"); print(ev.round(3).to_string())
    # 구간별 이벤트(점수>=3)
    res["freq"]={k:float(np.mean([sigs[t][k].mean() for t in sigs])) for k in ["S1","S2","S3","S4","S5"]}
    print("신호 발생 빈도",res["freq"])
    P=PARAMS; split=pd.Timestamp(P["split"]); out=[]
    rng=np.random.default_rng(0)
    for mode,k in [("A",2),("A",3),("A",4),("B",2),("B",3),("B",4)]:
        tr=run_backtest(data,sigs,P,mode,k)
        row=dict(전략=f"{mode}(점수>={k})")
        for lab,sub in (("전체",tr),("IS~2022",tr[tr.entry<split] if len(tr) else tr),("OOS2023~",tr[tr.entry>=split] if len(tr) else tr)):
            s=stats(sub,lab); out.append({**row,**s})
        if len(tr):
            pf=portfolio(tr,data,P,P["cap"])
            if pf: out.append({**row,"구간":"포트폴리오(10종목 동일가중)",**pf[0]})
        print(mode,k,len(tr))
    # 대조군: 무작위 진입(전체 일수의 0.5%), 동일 청산, 20회 평균
    ctr=[]
    for r in range(20):
        tr=run_backtest(data,sigs,P,"R",0,rng=rng,rand_n=0.005)
        ctr.append(stats(tr,"무작위"))
    cdf=pd.DataFrame(ctr)
    out.append(dict(전략="대조군(무작위진입 20회평균)",구간="전체",거래수=float(cdf["거래수"].mean()),승률=float(cdf["승률"].mean()),
                    평균수익=float(cdf["평균수익"].mean()),중앙수익=float(cdf["중앙수익"].mean()),손익비PF=float(cdf["손익비PF"].mean()),평균보유일=float(cdf["평균보유일"].mean())))
    kb=idx["KOSPI"].dropna(); res["kospi_bh"]=float(kb.iloc[-1]/kb.iloc[0]); res["kospi_cagr"]=float((kb.iloc[-1]/kb.iloc[0])**(365.25/(kb.index[-1]-kb.index[0]).days)-1)
    res["backtest"]=out
    json.dump(res,open(os.path.join(HERE,"ep01_results.json"),"w"),default=float,ensure_ascii=False,indent=1)
    print(pd.DataFrame(out).round(4).to_string())
if __name__=="__main__": main()
