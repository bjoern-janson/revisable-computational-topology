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

    @property
    def routes(self)->tuple[CorrectionRoute,...]:
        return tuple(self._routes[k] for k in sorted(self._routes))

    @property
    def outcomes(self):
        return tuple(self._outcomes)

    @property
    def commitment_ids(self)->frozenset[str]:
        return frozenset(r.commitment_id for r in self._routes.values())

    def routes_for_commitment(self,commitment_id:str)->tuple[CorrectionRoute,...]:
        return tuple(r for r in self.routes if r.commitment_id==commitment_id)

    def state(self,route_id:str)->tuple[bool,bool]:
        return self._state[route_id]

    def admit(self,route:CorrectionRoute,*,retained:bool,reachable:bool):
        if reachable and not retained:
            raise ValueError('reachable implies retained')
        if route.route_id in self._routes and self._routes[route.route_id]!=route:
            raise ValueError('route identity collision')
        if route.route_id not in self._routes:
            self._routes[route.route_id]=route
            self._state[route.route_id]=(bool(retained),bool(reachable))

    def set_state(self,route_id:str,*,retained:bool,reachable:bool):
        if route_id not in self._routes: raise KeyError(route_id)
        if reachable and not retained: raise ValueError('reachable implies retained')
        self._state[route_id]=(bool(retained),bool(reachable))

    def profile(self)->CorrectiveAccessProfile:
        n=len(self._routes)
        r=sum(1 for x in self._state.values() if x[0])
        q=sum(1 for x in self._state.values() if x[1])
        return CorrectiveAccessProfile(None if n==0 else r/n,None if n==0 else q/n,r,q,n)

    def preview_state(self,candidate,*,structural_cooldown:bool)->dict[str,tuple[bool,bool]]:
        """Metadata-only route state after one candidate, without mutation."""
        state=dict(self._state)
        kind=getattr(candidate.kind,'value',str(candidate.kind))
        if kind=='RETIRE' and candidate.generation is not None:
            cid=f'C:{candidate.source}->{candidate.target}:g{candidate.generation}'
            for route in self.routes_for_commitment(cid):
                state[route.route_id]=(False,False)
        elif kind=='REOPEN' and candidate.handle_id is not None:
            for route in self.routes:
                if route.operation=='REOPEN' and route.target_rank_or_handle==candidate.handle_id:
                    state[route.route_id]=(False,False)
        if structural_cooldown:
            state={rid:(ret,False) for rid,(ret,_) in state.items()}
        return state

    def preview_profile(self,candidate,*,structural_cooldown:bool)->CorrectiveAccessProfile:
        state=self.preview_state(candidate,structural_cooldown=structural_cooldown)
        n=len(state); r=sum(1 for x in state.values() if x[0]); q=sum(1 for x in state.values() if x[1])
        return CorrectiveAccessProfile(None if n==0 else r/n,None if n==0 else q/n,r,q,n)

    def record_unmeasured_exercise(self,route_id:str,step:int):
        if route_id not in self._routes: raise KeyError(route_id)
        self._outcomes.append(CorrectionOutcome(route_id,int(step)))
