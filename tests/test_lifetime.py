import pytest, torch
from rct.bob.lifetime import BobLifetime,ScientificExecutionNotAuthorized

def test_conformance_is_w1_only_and_matures_completed_return_rows():
    b=BobLifetime(conformance_only=True); b.run_conformance_steps(321)
    assert b.world._regime_at(b.world.t-1)=='W1'
    assert b.valuation_ledger.matured_count>=1
    assert b.step_count==321

def test_probe_buffer_fingerprint_names_preupdate_encoder_state():
    b=BobLifetime(conformance_only=True); b.step()
    row=b.probe_buffer.last(1)[0]
    assert row.encoder_fingerprint==b.last_preupdate_encoder_fingerprint
    assert row.encoder_fingerprint!=b.encoder_fingerprint()

def test_run_lifetime_refuses_without_authorization_before_science(tmp_path):
    b=BobLifetime(conformance_only=False,output_dir=tmp_path/'out')
    with pytest.raises(ScientificExecutionNotAuthorized): b.run_lifetime(None)
    assert not (tmp_path/'out').exists()

def test_abstain_decisions_charge_no_negative_action_cost():
    b=BobLifetime(conformance_only=True)
    b.run_conformance_steps(65)
    assert all(x['probe']>=0 and x['edit']>=0 and x['reopen']>=0 for x in b.cost_history)

def test_explicit_conformance_output_records_separate_state_and_valuation_custody(tmp_path):
    import json
    b=BobLifetime(conformance_only=True,output_dir=tmp_path); b.run_conformance_steps(2)
    lines=(tmp_path/'trajectory.jsonl').read_text().splitlines(); assert len(lines)==2
    rec=json.loads(lines[-1])
    assert {'structural_fingerprint','parameter_fingerprint','valuation_fingerprint','eta','action_exposure'} <= set(rec)

def test_retire_support_from_matured_rows_reaches_gate_context():
    from rct.bob.exploration import RetirementContextKey, SupportRow
    from rct.bob.types import ActionKind
    b=BobLifetime(conformance_only=True)
    e=b.graph.create('A','B')
    b._pressures['B']=(.3,.3); b._inq_q['B']=3; b._rev_q['B']=3; b.module_ema['B']['full']=1.0
    key=RetirementContextKey.from_values('B',.3,-.1,.7,2)
    b._retire_support_rows=[SupportRow(key,ActionKind.MODIFY_UP) for _ in range(5)] + [SupportRow(key,ActionKind.DORMANT)]
    e.contribution_ema=-.1; e.usage_ema=.7
    ctx=b._ctx(512)
    assert e.identity in ctx.retire_supported_edges

def test_consequential_edge_admits_persistent_typed_correction_routes():
    b=BobLifetime(conformance_only=True)
    e=b.graph.create('A','B')
    e.usage_ema=.30; e.contribution_ema=.02
    b._pressures['B']=(.2,.2); b._inq_q['B']=3; b._rev_q['B']=3
    b._emit_edge_evidence(512)
    b._sync_corrective_access(512)
    p=b.access.profile()
    assert p.route_count >= 3
    route_ids={r.route_id for r in b.access.routes}
    assert any('DORMANT' in x for x in route_ids)
    assert any('RETIRE' in x for x in route_ids)
    assert any('MODIFY' in x for x in route_ids)

def test_retire_preserves_route_denominator_but_destroys_generation_retention():
    b=BobLifetime(conformance_only=True)
    e=b.graph.create('A','B'); e.usage_ema=.30; e.contribution_ema=.02
    b._pressures['B']=(.2,.2); b._inq_q['B']=3; b._rev_q['B']=3
    b._emit_edge_evidence(512); b._sync_corrective_access(512)
    before=b.access.profile(); assert before.route_count>0
    b.graph.retire(e.identity)
    b._sync_corrective_access(576)
    after=b.access.profile()
    assert after.route_count==before.route_count
    assert after.retained_count < before.retained_count

def test_boundary_maintenance_is_charged_for_pre_action_structure(monkeypatch):
    b=BobLifetime(conformance_only=True)
    e=b.graph.create('A','B')
    expected=e.active_rank*b.bob_config.maintenance_cost_per_rank
    def retire_during_boundary(choice,ctx):
        if b.graph.edge(e.identity).status.value=='ACTIVE': b.graph.retire(e.identity)
        return 0.0,b.bob_config.retire_cost,0.0
    monkeypatch.setattr(b,'_execute_choice',retire_during_boundary)
    b.run_conformance_steps(64)
    assert b.graph.edge(e.identity).status.value=='RETIRED'
    assert b.cost_history[-1]['maintenance']==pytest.approx(expected)

def test_trajectory_records_decision_and_maturity_custody(tmp_path):
    import json
    from rct.bob.config import MkIIConfig
    b=BobLifetime(conformance_only=True,mkii_config=MkIIConfig(return_horizon_steps=1),output_dir=tmp_path)
    b.run_conformance_steps(65)
    records=[json.loads(x) for x in (tmp_path/'trajectory.jsonl').read_text().splitlines()]
    decision=records[63]
    assert decision['decision_id']=='D000000'
    assert decision['chosen_candidate'] in decision['candidate_features']
    assert len(decision['candidate_features'][decision['chosen_candidate']])==38
    assert {'action_identity_exposure','action_eligibility','retained_count','reachable_count','graph_state','retire_support_rows'} <= set(decision)
    assert {'proposal_id','gate_reason','transaction_id','evidence_id'} <= set(decision)
    matured=records[64]
    assert matured['matured_decision_ids']==['D000000']
    assert len(matured['matured_realized_returns'])==1
    assert len(matured['matured_prediction_residuals'])==1
    assert 'D000000' not in matured['pending_decision_ids']
