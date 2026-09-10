from __future__ import annotations
from pathlib import Path
import hashlib,json
import torch
from .features import FEATURE_NAMES

def structural_fingerprint(graph): return graph.structural_fingerprint()
def parameter_fingerprint(backbone,graph=None):
    h=hashlib.sha256()
    for p in backbone.parameters(): h.update(p.detach().cpu().contiguous().numpy().tobytes())
    if graph is not None: h.update(graph.parameter_fingerprint().encode())
    return h.hexdigest()
def valuation_fingerprint(valuation,matured_count):
    schema=hashlib.sha256('|'.join(FEATURE_NAMES).encode()).hexdigest(); return valuation.fingerprint(matured_count,schema)

class TrajectoryRecorder:
    def __init__(self,output_dir=None):
        self.output_dir=Path(output_dir) if output_dir is not None else None; self.count=0
        if self.output_dir is not None: self.path=self.output_dir/'trajectory.jsonl'
    def append(self,record):
        self.count+=1
        if self.output_dir is None:return
        self.output_dir.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
