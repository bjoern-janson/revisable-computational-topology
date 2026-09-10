from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib
import torch

class DecisionMode(str,Enum):
    GREEDY='GREEDY'; EXPLORE='EXPLORE'

@dataclass(frozen=True)
class PendingDecision:
    decision_id:str; decision_step:int; candidate_id:str; phi:torch.Tensor; chosen_by:str; accumulated_return:float=0.0
@dataclass(frozen=True)
class MaturedDecision:
    decision_id:str; decision_step:int; candidate_id:str; phi:torch.Tensor; chosen_by:str; realized_return:float

class ValuationLedger:
    def __init__(self,horizon:int=256): self.horizon=int(horizon); self._pending=[]; self._matured=[]
    @property
    def pending(self): return tuple(self._pending)
    @property
    def matured(self): return tuple(self._matured)
    @property
    def matured_count(self): return len(self._matured)
    def record_decision(self,decision_id,step,phi,*,chosen_by,candidate_id):
        self._pending.append(PendingDecision(str(decision_id),int(step),str(candidate_id),phi.detach().cpu().to(torch.float64).clone(),str(chosen_by),0.0))
    def observe_cost_step(self,step,*,task,probe,edit,maintenance,reopen):
        step=int(step)
        still=[]
        for row in self._pending:
            if step>=row.decision_step+self.horizon:
                self._matured.append(MaturedDecision(row.decision_id,row.decision_step,row.candidate_id,row.phi.clone(),row.chosen_by,row.accumulated_return))
            else: still.append(row)
        self._pending=still
        cost=float(task)+float(probe)+float(edit)+float(maintenance)+float(reopen)
        updated=[]
        for row in self._pending:
            if row.decision_step<=step<=row.decision_step+self.horizon-1:
                updated.append(PendingDecision(row.decision_id,row.decision_step,row.candidate_id,row.phi,row.chosen_by,row.accumulated_return-cost))
            else: updated.append(row)
        self._pending=updated

class LinearActionValuation:
    def __init__(self,dim:int=38,ridge_lambda:float=1.0): self.eta=torch.zeros(dim,dtype=torch.float64); self.ridge_lambda=float(ridge_lambda)
    def fit(self,rows):
        if not rows: self.eta.zero_(); return self.eta
        X=torch.stack([r.phi.to(torch.float64) for r in rows]); y=torch.tensor([r.realized_return for r in rows],dtype=torch.float64); I=torch.eye(X.shape[1],dtype=torch.float64); self.eta=torch.linalg.solve(X.T@X+self.ridge_lambda*I,X.T@y); return self.eta
    def score(self,phi): return float(torch.dot(self.eta,phi.to(torch.float64)).item())
    def fingerprint(self,matured_count:int=0,feature_schema_hash:str=''):
        h=hashlib.sha256(); h.update(self.eta.detach().cpu().contiguous().numpy().tobytes()); h.update(str(self.ridge_lambda).encode()); h.update(str(matured_count).encode()); h.update(feature_schema_hash.encode()); return h.hexdigest()
