import pytest
from rct.bob.interfaces import AdapterGraph
from rct.bob.lineage import StructuralLedger, StructuralTransaction
from rct.bob.types import Operation


def test_structural_ledger_is_append_only_and_rejects_duplicate_transaction():
    ledger = StructuralLedger()
    tx = StructuralTransaction(
        transaction_id="T000001", proposal_id="P000001", operation=Operation.CREATE,
        graph_before="a", graph_after="b", evidence_ref="E000001",
        affected_routes=(), reopen_handle_id=None, charged_cost=0.25,
    )
    ledger.append_transaction(tx)
    with pytest.raises(ValueError):
        ledger.append_transaction(tx)
    assert ledger.transactions == (tx,)


def test_dormant_handle_snapshot_is_copy_safe():
    g = AdapterGraph(hidden_dim=4, max_rank=3, initial_rank=2)
    e = g.create("A", "B")
    h = g.dormant(e.identity, historical_support=0.1)
    first = g.handle(h.handle_id)
    first_u = first.snapshot.u.clone()
    first.snapshot.u.add_(100)
    second = g.handle(h.handle_id)
    assert (second.snapshot.u == first_u).all()
