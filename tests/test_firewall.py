import ast,inspect
import rct.bob.manager as manager, rct.bob.features as features, rct.bob.exploration as exploration, rct.bob.probe as probe, rct.bob.gate as gate
from rct.bob.gate import GateContext
from dataclasses import fields

def test_manager_side_modules_do_not_import_world_or_audit():
    for mod in (manager,features,exploration,probe,gate):
        tree=ast.parse(inspect.getsource(mod))
        imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import): imports += [x.name for x in n.names]
            if isinstance(n,ast.ImportFrom): imports += [n.module or '']
        assert not any(x.endswith('.world') or x.endswith('.audit') or x in {'world','audit'} for x in imports), mod.__name__

def test_gate_context_has_no_hidden_world_fields():
    names={f.name for f in fields(GateContext)}
    assert names.isdisjoint({'regime','pairing','transition_operator','correct_edges','future_loss','future_probe_result','audit_pair_table'})
