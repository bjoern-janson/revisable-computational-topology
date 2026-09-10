import torch
from rct.bob.valuation import ValuationLedger,LinearActionValuation

def test_row_matures_at_t_plus_256_and_excludes_t_plus_256():
    v=ValuationLedger(horizon=256); v.record_decision('d',64,torch.ones(38),chosen_by='GREEDY',candidate_id='c')
    for t in range(64,320): v.observe_cost_step(t,task=1,probe=0,edit=0,maintenance=0,reopen=0)
    assert v.matured_count==0
    v.observe_cost_step(320,task=99,probe=0,edit=0,maintenance=0,reopen=0)
    assert v.matured_count==1 and v.matured[0].realized_return==-256

def test_ridge_refit_uses_completed_rows_and_eta_starts_zero():
    v=ValuationLedger(horizon=1); model=LinearActionValuation()
    assert torch.equal(model.eta,torch.zeros(38,dtype=torch.float64))
    x=torch.zeros(38,dtype=torch.float64); x[0]=1
    v.record_decision('d',0,x,chosen_by='EXPLORE',candidate_id='a'); v.observe_cost_step(0,task=2,probe=0,edit=0,maintenance=0,reopen=0); v.observe_cost_step(1,task=0,probe=0,edit=0,maintenance=0,reopen=0)
    model.fit(v.matured); assert model.score(x)<0

def test_feature_bytes_are_frozen_at_decision():
    v=ValuationLedger(); x=torch.ones(38,dtype=torch.float64); v.record_decision('d',0,x,chosen_by='GREEDY',candidate_id='a'); x.zero_(); assert v.pending[0].phi[0]==1

def test_valuation_api_has_no_causal_effect_language():
    import rct.bob.valuation as m
    names=' '.join(dir(m)).lower(); assert 'causal_value' not in names and 'counterfactual_effect' not in names

def test_matured_row_preserves_prediction_made_at_decision_time():
    v=ValuationLedger(horizon=1)
    phi=torch.zeros(38,dtype=torch.float64); phi[0]=1
    v.record_decision('d',0,phi,chosen_by='GREEDY',candidate_id='a',predicted_return=-1.5)
    v.observe_cost_step(0,task=2,probe=0,edit=0,maintenance=0,reopen=0)
    v.observe_cost_step(1,task=99,probe=0,edit=0,maintenance=0,reopen=0)
    row=v.matured[0]
    assert row.predicted_return==-1.5
    assert row.realized_return==-2.0
    assert row.prediction_residual==-0.5
