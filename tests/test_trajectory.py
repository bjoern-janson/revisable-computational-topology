import json, torch
from rct.bob.trajectory import TrajectoryRecorder,structural_fingerprint,parameter_fingerprint,valuation_fingerprint
from rct.bob.interfaces import AdapterGraph
from rct.bob.backbone import BobBackbone
from rct.bob.config import BobConfig,WorldConfig
from rct.bob.valuation import LinearActionValuation

def test_eta_change_does_not_change_structural_fingerprint():
    g=AdapterGraph(32,8,2); m=BobBackbone(WorldConfig(),BobConfig()); v=LinearActionValuation()
    s=structural_fingerprint(g); v.eta[0]=1
    assert structural_fingerprint(g)==s
    assert valuation_fingerprint(v,0)!=valuation_fingerprint(LinearActionValuation(),0)

def test_recorder_writes_canonical_append_only_jsonl(tmp_path):
    r=TrajectoryRecorder(tmp_path); r.append({'step':1,'z':2,'a':1}); r.append({'step':2,'a':3})
    lines=(tmp_path/'trajectory.jsonl').read_text().splitlines(); assert len(lines)==2
    assert lines[0]=='{"a":1,"step":1,"z":2}'

def test_no_output_directory_means_no_io():
    r=TrajectoryRecorder(None); r.append({'step':1}); assert r.count==1

def test_render_trajectory_includes_eta_and_action_exposure_plots(tmp_path):
    from rct.bob.visualize import render_trajectory
    records=[{'step':0,'loss':1.0,'eta':[0.0]*38,'action_exposure':{'ABSTAIN':1},'abstain_rate':1.0}]
    paths=render_trajectory(records,tmp_path)
    names={p.name for p in paths}
    assert 'eta_coefficients.png' in names and 'action_class_exposure.png' in names
