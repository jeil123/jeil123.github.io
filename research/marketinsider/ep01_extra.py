"""ep01 추가 검증: (1) 무작위 진입 대조군 포트폴리오 (2) 임계값 민감도 (3) 신호별 급등/급락 비율(lift) 신뢰구간"""
import json, os, numpy as np, pandas as pd
import ep01_signals as E
P0=E.PARAMS
data,idx=E.load()
data={t:cd for t,d in data.items() if t not in E.EXCLUDE for cd in [E.clean(d)] if cd is not None}
out={}
def prep(P):
    sigs={};frames=[]
    for t,d in data.items():
        b=idx["KOSDAQ"] if t.endswith(".KQ") else idx["KOSPI"]
        sigs[t]=E.signals(d,b,P); frames.append(pd.concat([E.forward(d),sigs[t]],axis=1))
    return sigs,frames
sigs,frames=prep(P0)
# (1) 대조군 포트폴리오: 무작위 진입, B전략과 동일 청산 (box 이탈 없음)
rng=np.random.default_rng(1); ctr=[]
for r in range(5):
    tr=E.run_backtest(data,sigs,P0,"R",0,rng=rng,rand_n=0.005); tr["vr"]=rng.random(len(tr))
    pf=E.portfolio(tr,data,P0,P0["cap"])[0]; ctr.append(pf)
out["control_portfolio"]={k:float(np.mean([c[k] for c in ctr])) for k in ctr[0]}
print("대조군 포트폴리오(5회평균)",out["control_portfolio"])
# (2) lift 및 95% 신뢰구간 (급등 5일내 +10%)
allf=pd.concat([f.assign(**{}) for f in frames]); base=(allf["mx5"]>=0.10)[allf["f20"].notna()].mean()
rows=[]
for name,fn in [("S1",lambda s:s.S1==1),("S2",lambda s:s.S2==1),("S3",lambda s:s.S3==1),("S4",lambda s:s.S4==1),("S5",lambda s:s.S5==1),("점수>=3",lambda s:s.score>=3)]:
    ev=pd.concat([f[E.decluster(fn(f).fillna(False).astype(bool)) & f.f20.notna()] for f in frames])
    p=(ev.mx5>=0.10).mean(); n=len(ev); se=np.sqrt(p*(1-p)/n)
    rows.append(dict(신호=name,건수=n,급등률=p,기본확률=base,lift=p/base,ci95_lo=p-1.96*se,ci95_hi=p+1.96*se))
out["lift"]=rows; print(pd.DataFrame(rows).round(3).to_string())
# (3) 민감도: 임계값을 엄격/느슨하게
sens=[]
for lab,f in (("엄격",0.75),("기준",1.0),("느슨",1.35)):
    P=dict(P0)
    for k in ("s1_box","s1_body","s2_minrange","s4_spread","s3_move"): P[k]=P0[k]*f
    P["s1_vol"]=min(0.95,P0["s1_vol"]*f); P["s3_vol"]=P0["s3_vol"]/f
    sg,fr=prep(P)
    for k in (2,3):
        evs=[]
        for f_ in fr:
            m=E.decluster((f_.score>=k)); evs.append(f_[m & f_.f20.notna()])
        ev=pd.concat(evs); 
        trB=E.run_backtest(data,sg,P,"B",k)
        sens.append(dict(설정=lab,점수=f">={k}",이벤트건수=len(ev),급등률=float((ev.mx5>=0.10).mean()),
                         급락률=float((ev.mn5<=-0.10).mean()),평균20일=float(ev.f20.mean()),
                         B거래수=len(trB),B승률=float((trB.ret>0).mean()) if len(trB) else None,B평균수익=float(trB.ret.mean()) if len(trB) else None))
out["sensitivity"]=sens; print(pd.DataFrame(sens).round(4).to_string())
json.dump(out,open("ep01_extra.json","w"),default=float,ensure_ascii=False,indent=1)
