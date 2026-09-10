from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CorrectionRoute:
    commitment_id:str
    route_id:str
    operation:str
    edge_generation:int
    target_rank_or_handle:str

@dataclass(frozen=True)
class CorrectiveAccessProfile:
    retained:float|None
    reachable:float|None
    retained_count:int
    reachable_count:int
    route_count:int

@dataclass(frozen=True)
class CorrectionOutcome:
    route_id:str
    exercised_step:int
    evaluation_state:str = 'UNMEASURED_CONTRACT_NOT_FROZEN'

class CorrectiveAccessLedger:
    def __init__(self):
        self._routes:dict[str,CorrectionRoute]={}
        self._state:dict[str,tuple[bool,bool]]={}
        self._outcomes:list[CorrectionOutcome]=[]
    def admit(self,route:CorrectionRoute,*,retained:bool,reachable:bool):
        if route.route_id in self._routes and self._routes[route.route_id]!=route: raise ValueError('route identity collision')
        self._routes[route.route_id]=route; self._state[route.route_id]=(bool(retained),bool(reachable))
    def set_state(self,route_id:str,*,retained:bool,reachable:bool):
        if route_id not in self._routes: raise KeyError(route_id)
        if reachable and not retained: raise ValueError('reachable implies retained')
        self._state[route_id]=(bool(retained),bool(reachable))
    def profile(self)->CorrectiveAccessProfile:
        n=len(self._routes); r=sum(1 for x in self._state.values() if x[0]); q=sum(1 for x in self._state.values() if x[1])
        return CorrectiveAccessProfile(None if n==0 else r/n,None if n==0 else q/n,r,q,n)
    @property
    def outcomes(self): return tuple(self._outcomes)
    def record_unmeasured_exercise(self,route_id:str,step:int):
        if route_id not in self._routes: raise KeyError(route_id)
        self._outcomes.append(CorrectionOutcome(route_id,int(step)))
