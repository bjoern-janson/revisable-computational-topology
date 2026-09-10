from rct.bob.config import BobConfig, WorldConfig
from rct.bob.audit import audit_world


def test_audit_reports_constructed_invariance_and_relational_opportunity():
    report = audit_world(WorldConfig(), BobConfig(), samples=4096)
    assert report.analytic_latent_marginal_invariance
    assert report.analytic_cross_lag_change > 0.0
    assert report.empirical_cross_lag_change > 0.0
    assert report.min_partner_prediction_gain > 0.0
    assert report.block_count == 32
    assert report.block_size == 128
    assert report.samples_per_regime == 4096


def test_audit_is_environment_evidence_not_bob_execution():
    report = audit_world(WorldConfig(), BobConfig(), samples=4096)
    assert report.scope == "ENVIRONMENT_OPPORTUNITY_ONLY"
    assert not report.bob_executed


def test_audit_reports_block_dispersion_and_fails_closed_below_threshold():
    import pytest
    report=audit_world(WorldConfig(),BobConfig(),samples=4096)
    assert report.block_gain_std >= 0.0
    with pytest.raises(RuntimeError,match='ENVIRONMENT_OPPORTUNITY_FAILURE'):
        audit_world(WorldConfig(),BobConfig(task_opportunity_min_gain=999.0),samples=4096)
