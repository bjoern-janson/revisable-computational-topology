from __future__ import annotations
from dataclasses import dataclass
import torch
from .config import BobConfig,WorldConfig
from .types import NODE_IDS
from .world import RotatingDependencyWorld
@dataclass(frozen=True)
class WorldAuditReport:
    analytic_latent_marginal_invariance:bool; max_abs_mean_shift:float; max_abs_cov_shift:float; analytic_cross_lag_change:float; empirical_cross_lag_change:float; analytic_self_lag_norm:float; analytic_partner_lag_norm:float; empirical_self_lag_change:float; empirical_partner_lag_change:float; min_partner_prediction_gain:float; block_count:int; block_size:int; samples_per_regime:int; block_gain_std:float; scope:str='ENVIRONMENT_OPPORTUNITY_ONLY'; bob_executed:bool=False

def _cov(x,y=None):
    x=x.double(); y=x if y is None else y.double(); xc=x-x.mean(0,keepdim=True); yc=y-y.mean(0,keepdim=True); return xc.T@yc/max(1,x.shape[0]-1)
def _ridge(x,y,v,ridge=1e-3):
    x=x.double(); y=y.double(); v=v.double(); x=torch.cat([x,torch.ones((x.shape[0],1),dtype=torch.float64)],1); v=torch.cat([v,torch.ones((v.shape[0],1),dtype=torch.float64)],1); I=torch.eye(x.shape[1],dtype=torch.float64); I[-1,-1]=0; return v@torch.linalg.solve(x.T@x+ridge*I,x.T@y)
def _partners(w,r):
    out={}
    for a,b in w.audit_pairing(r): i,j=NODE_IDS.index(a),NODE_IDS.index(b); out[i]=j; out[j]=i
    return out

def audit_world(world_config=None,bob_config=None,*,samples=4096):
    wc=world_config or WorldConfig(); bc=bob_config or BobConfig(); w=RotatingDependencyWorld(wc); r1=w.audit_transition_matrix('W1'); r3=w.audit_transition_matrix('W3'); eye=torch.eye(r1.shape[0],dtype=torch.float64); invariant=bool(torch.allclose(r1.T@r1,eye,atol=1e-10,rtol=1e-10) and torch.allclose(r3.T@r3,eye,atol=1e-10,rtol=1e-10)); x1,y1=w.audit_forced_sequence('W1',samples,wc.audit_seed+10); x3,y3=w.audit_forced_sequence('W3',samples,wc.audit_seed+20)
    mean_shift=max(float((x1[:,j].mean(0)-x3[:,j].mean(0)).abs().max()) for j in range(4)); cov_shift=max(float((_cov(x1[:,j])-_cov(x3[:,j])).abs().max()) for j in range(4)); cross_sq=sum(float(((_cov(x1[:,i],y1[:,j])-_cov(x3[:,i],y3[:,j]))**2).sum()) for i in range(4) for j in range(4) if i!=j); p1,p3=_partners(w,'W1'),_partners(w,'W3'); selfchg=[float((_cov(x1[:,j],y1[:,j])-_cov(x3[:,j],y3[:,j])).norm()) for j in range(4)]; partnerchg=[float((_cov(x1[:,p1[j]],y1[:,j])-_cov(x3[:,p3[j]],y3[:,j])).norm()) for j in range(4)]
    ntr=3072 if samples==4096 else int(samples*.75); gains=[]
    for regime,xx,yy in [('W1',x1,y1),('W3',x3,y3)]:
        ps=_partners(w,regime)
        for t in range(4):
            p=ps[t]; yte=yy[ntr:,t].double(); pl=_ridge(xx[:ntr,t],yy[:ntr,t],xx[ntr:,t]); pp=_ridge(torch.cat([xx[:ntr,t],xx[:ntr,p]],1),yy[:ntr,t],torch.cat([xx[ntr:,t],xx[ntr:,p]],1)); gains.append(float(torch.mean((pl-yte)**2)-torch.mean((pp-yte)**2)))
    if min(gains)<=bc.task_opportunity_min_gain: raise RuntimeError(f'ENVIRONMENT_OPPORTUNITY_FAILURE:{min(gains):.6g}')
    block_size=128 if samples==4096 else max(1,samples//32); block_count=samples//block_size; block_g=[]
    # serial-block dispersion using true partner ridge fit within each 128 block vs local mean baseline
    for regime,xx,yy in [('W1',x1,y1),('W3',x3,y3)]:
        ps=_partners(w,regime)
        for b in range(block_count):
            s=slice(b*block_size,(b+1)*block_size); half=block_size//2
            for t in range(4):
                p=ps[t]; xb=xx[s]; yb=yy[s,t].double(); pl=_ridge(xb[:half,t],yb[:half],xb[half:,t]); pp=_ridge(torch.cat([xb[:half,t],xb[:half,p]],1),yb[:half],torch.cat([xb[half:,t],xb[half:,p]],1)); block_g.append(float(torch.mean((pl-yb[half:])**2)-torch.mean((pp-yb[half:])**2)))
    return WorldAuditReport(invariant,mean_shift,cov_shift,float((wc.rho*(r1-r3)).norm()),cross_sq**.5,float((wc.rho*wc.alpha*w._q).norm()),float((wc.rho*wc.beta*w._q).norm()),sum(selfchg)/4,sum(partnerchg)/4,min(gains),block_count,block_size,samples,float(torch.tensor(block_g).std(unbiased=False)))
