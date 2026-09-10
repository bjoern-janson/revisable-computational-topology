from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import hashlib, torch
from .types import ActionCandidate,ActionKind,NodeId

class ExplorationSchedule:
    def __init__(self,seed:int=43): self._gen=torch.Generator().manual_seed(seed); self._slots={}
    def _slot(self,block:int):
        if block not in self._slots: self._slots[block]=int(torch.randint(0,4,(1,),generator=self._gen).item())
        return self._slots[block]
    def is_explore(self,k:int)->bool: return (k%4)==self._slot(k//4)

class ExposureLedger:
    def __init__(self): self.class_counts=Counter(); self.identity_counts=Counter(); self.eligibility_counts=Counter()
    def record_eligibility(self,cands):
        for c in cands: self.eligibility_counts[c.kind]+=1
    def record_choice(self,c): self.class_counts[c.kind]+=1; self.identity_counts[c.candidate_id]+=1
    def class_count(self,k): return self.class_counts[k]
    def identity_count(self,i): return self.identity_counts[i]

def stable_hash(text:str): return hashlib.sha256(text.encode()).hexdigest()
def select_exploration(candidates,exposure:ExposureLedger):
    legal=[c for c in candidates if c.kind is not ActionKind.RETIRE]
    if not legal: raise ValueError('no non-RETIRE exploration candidate')
    chosen=min(legal,key=lambda c:(exposure.class_count(c.kind),exposure.identity_count(c.candidate_id),stable_hash('47|'+c.candidate_id)))
    exposure.record_choice(chosen); return chosen

@dataclass(frozen=True)
class RetirementContextKey:
    target_node:NodeId; pressure_bin:int; contribution_bin:int; usage_bin:int; rank_bin:int
    @staticmethod
    def from_values(target_node,pressure,contribution,usage,rank):
        p=0 if pressure<.25 else 1 if pressure<.5 else 2 if pressure<1 else 3
        c=0 if contribution<0 else 1 if contribution<.25 else 2
        u=0 if usage<.25 else 1 if usage<.5 else 2
        r=0 if rank<=2 else 1 if rank<=5 else 2
        return RetirementContextKey(target_node,p,c,u,r)
@dataclass(frozen=True)
class SupportRow:
    context:RetirementContextKey; kind:ActionKind
class RetirementSupportGate:
    ALLOWED=frozenset({ActionKind.MODIFY_UP,ActionKind.MODIFY_DOWN,ActionKind.DORMANT,ActionKind.REOPEN})
    def __init__(self,min_rows:int=6,min_action_kinds:int=2): self.min_rows=min_rows; self.min_action_kinds=min_action_kinds
    def supported(self,key,rows):
        xs=[r for r in rows if r.context==key and r.kind in self.ALLOWED]
        return len(xs)>=self.min_rows and len({r.kind for r in xs})>=self.min_action_kinds
