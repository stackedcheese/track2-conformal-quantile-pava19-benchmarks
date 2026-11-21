"""Track 2. Offline chronological quantile calibration and fixed-universe sequential review."""
import argparse
from pathlib import Path
import sys
def offline(event,args):
 if event.startswith('socket.') and event != 'socket.gethostname': raise RuntimeError('OFFLINE_SOCKET_BLOCK')
sys.addaudithook(offline)
import json
import math
import numpy as np
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import mean_pinball_loss
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

LEVELS=np.arange(1,20)/20
SEED=2715215

def fixture():
    rng=np.random.default_rng(SEED)
    x=rng.normal(size=(3600,6))
    scale=.45+.6*abs(x[:,0])
    y=1.1*x[:,0]-x[:,1]+np.sin(x[:,2]*2)+scale*rng.normal(size=len(x))
    y[2700:]+=0.65+(.25+abs(x[2700:,3])*.2)*rng.normal(size=900)
    return x,y

def vendor_model(name,quantile):
    if name=='CatBoost':
        return CatBoostRegressor(loss_function=f'Quantile:alpha={quantile}',iterations=45,depth=5,
            learning_rate=.15,verbose=False,random_seed=SEED,thread_count=2,allow_writing_files=False,task_type="CPU")
    if name=='LightGBM':
        return LGBMRegressor(objective='quantile',alpha=quantile,n_estimators=45,learning_rate=.15,
            max_depth=5,random_state=SEED,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    if name=='XGBoost':
        return XGBRegressor(objective="reg:quantileerror",quantile_alpha=quantile,n_estimators=45,
            max_depth=5,learning_rate=.15,random_state=SEED,n_jobs=2,tree_method='hist',device='cpu')
    raise ValueError('unknown model')

def pava(x):
    return np.stack([isotonic_regression(row,increasing=True) for row in x])

def crps(grid,y):
    x=np.column_stack((grid[:,0],grid,grid[:,-1]))
    total=np.zeros(len(y))
    for i in range(20):
        left=i/20; right=(i+1)/20
        l=x[:,i];r=x[:,i+1]
        cross=(l!=r)&((y-l)*(y-r)<0)
        divider=np.where(cross,left+(right-left)*np.divide(y-l,r-l,out=np.zeros_like(y),where=r!=l),right)
        def loss(p):
            val=y-(l+(r-l)*(p-left)/(right-left))
            return 2*val*(p-(val<0))
        def exact(a,b):
            return (b-a)*(loss(a)+4*loss((a+b)*.5)+loss(b))/6
        total+=exact(left,divider)+exact(divider,right)
    return total

def crossing(x):
    return int(np.count_nonzero(np.any(np.diff(x,axis=1)<0,axis=1)))

def metrics(grid,y,interval):
    lo,hi=interval
    return {("crps" if crossing(grid)==0 else "integrated_pinball_nonmonotone"):float(np.mean(crps(grid,y))),"coverage":float(np.mean((lo<=y)&(y<=hi))),
      "width":float(np.mean(hi-lo)),"crossing_rows":crossing(grid),
      "mean_pinball":float(np.mean([mean_pinball_loss(y,grid[:,i],alpha=t) for i,t in enumerate(LEVELS)]))}

def simplex(values):
    values=np.asarray(values,dtype=float)
    assert np.all((values>0)&(values<1)) and values.sum()>=1
    def power(k): return values**k
    q=values**2/values.sum()
    def shin(z): return 2*q/(np.sqrt(z*z+4*(1-z)*q)+z)
    def solve(fn,lo,hi):
        assert fn(lo).sum()>=1-1e-13 and fn(hi).sum()<=1+1e-13
        for _ in range(80):
            mid=(lo+hi)/2
            if fn(mid).sum()>1:lo=mid
            else:hi=mid
        p=fn((lo+hi)/2);assert abs(p.sum()-1)<1e-11
        return p.tolist()
    return {"input":values.tolist(),"power":solve(power,1,256),"shin":solve(shin,0,1)}

def ebh(e,alpha,expected):
    if e.ndim!=1 or len(e)!=expected or np.any(~np.isfinite(e)) or np.any(e<0) or not 0<alpha<1:
        raise ValueError("COMPLETE_FIXED_UNIVERSE_REQUIRED")
    order=np.argsort(-e,kind="stable")
    k=np.flatnonzero(e[order]>=expected/(alpha*np.arange(1,expected+1)))
    return order[:k[-1]+1] if len(k) else order[:0]

def monitoring():
    rng=np.random.default_rng(SEED+25)
    m=20; alpha=.1; trials=300; horizon=100
    fdp=[];tpr=[];ville=0; latencies=[]
    for _ in range(trials):
        e=np.ones(m)
        stopped=horizon
        rejected=[]
        saw_ville=False
        for t in range(1,horizon+1):
            p=np.r_[np.full(15,.5),np.full(5,.8 if t>=25 else .5)]
            x=rng.random(m)<p
            e*=np.where(x,1.5,.5)
            saw_ville=saw_ville or e[0]>=1/alpha
            got=ebh(e,alpha,m)
            if len(got) or t==horizon:
                stopped=t;rejected=got;break
        false=int(sum(i<15 for i in rejected))
        true=len(rejected)-false
        fdp.append(false/max(1,len(rejected)));tpr.append(true/5)
        ville+=saw_ville
        if true:latencies.append(stopped-25 if stopped>=25 else 0)
    return {"seed":SEED+25,"alpha":alpha,"simulated_runs":trials,"fixed_families":m,
            "mean_fdp":float(np.mean(fdp)),"mean_true_discovery_fraction":float(np.mean(tpr)),
            "first_null_ville_crossing_rate":ville/trials,
            "median_true_alert_latency_steps":float(np.median(latencies)) if latencies else None,
            "scope":"Bernoulli simulation with preregistered factors and common bounded stopping"}

def run():
    x,y=fixture()
    baseline={}
    for name in ('CatBoost','LightGBM','XGBoost'):
        raw=[]
        for q in LEVELS:
            model=vendor_model(name,float(q))
            model.fit(x[:2000],y[:2000])
            raw.append(model.predict(x[2000:]))
        raw=np.stack(raw,axis=1)
        repaired=pava(raw)
        assert crossing(repaired)==0
        assert abs(float(crps(np.zeros((1,19)),np.asarray([3.]))[0])-3)<1e-12
        cal=repaired[:700]; actual=y[2000:2700]
        scores=np.maximum.reduce([cal[:,1]-actual,actual-cal[:,-2],np.zeros_like(actual)])
        rank=min(len(scores),math.ceil((len(scores)+1)*.90))
        cqr=float(np.partition(scores,rank-1)[rank-1])
        p=repaired[700:]; q=raw[700:]; actual=y[2700:]
        baseline[name]={"raw":metrics(q,actual,(q[:,1],q[:,-2])),
           "pava":metrics(p,actual,(p[:,1],p[:,-2])),
           "pava_cqr":metrics(p,actual,(p[:,1]-cqr,p[:,-2]+cqr)),
           "cqr_radius":cqr,"interval_correction_only":True}
    return {"schema":"SDL_V27_BENCHMARK_TRACK2_V1","status":"PASS_SYNTHETIC_SHIFT_EXECUTION",
      "scope":"Synthetic chronological train/calibration/shift splits. Protected cohorts excluded.",
      "seed":SEED,"rows":{"train":2000,"calibration":700,"shift_test":900},"models":baseline,
      "simplex":simplex([.8,.2381]),"eprocess":monitoring(),
      "historical_reference":{"past_only_rows":36040,"crossings_before":184,"crossings_after":0,"source":"Note 214 methods manifest"},
      "crps_integration":"19 piecewise linear quantiles with constant tails. CQR corrects intervals separately.",
      "claim_boundary":"Empirical shift coverage is descriptive and exchangeability-based coverage does not transfer automatically.",
      "network_calls":0}

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True)
    payload=run()
    with cli.parse_args().output.open("x",encoding="utf8",newline="\n") as fp:
        json.dump(payload,fp,sort_keys=True,indent=2,allow_nan=False);fp.write("\n")
