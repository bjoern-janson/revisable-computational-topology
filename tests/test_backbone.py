import torch
from rct.bob.backbone import BobBackbone
from rct.bob.config import BobConfig, WorldConfig
from rct.bob.types import NODE_IDS


def test_local_and_full_heads_are_distinct_and_losses_sum_domains():
    model = BobBackbone(WorldConfig(), BobConfig())
    assert model.local_decoders is not model.full_decoders
    losses = model.aggregate_losses({k: torch.tensor(float(i + 1)) for i, k in enumerate(NODE_IDS)})
    assert losses == 10.0


def test_backbone_shapes_and_message_changes_full_not_local_path():
    model = BobBackbone(WorldConfig(), BobConfig())
    x = {k: torch.randn(16) for k in NODE_IDS}
    h = model.encode(x)
    assert all(v.shape == (32,) for v in h.values())
    local = model.decode_local(h)
    zero = {k: [] for k in NODE_IDS}
    full0 = model.decode_full(h, zero)
    msgs = {k: [] for k in NODE_IDS}
    msgs["B"] = [torch.ones(32)]
    full1 = model.decode_full(h, msgs)
    assert local["B"].shape == (16,)
    assert full0["B"].shape == (16,)
    assert not torch.allclose(full0["B"], full1["B"])


def test_training_losses_use_coordinate_mean_then_domain_sum():
    model = BobBackbone(WorldConfig(), BobConfig())
    target = {k: torch.zeros(16) for k in NODE_IDS}
    local_pred = {k: torch.ones(16) * (i + 1) for i, k in enumerate(NODE_IDS)}
    full_pred = {k: torch.ones(16) * 2 * (i + 1) for i, k in enumerate(NODE_IDS)}
    losses = model.losses(local_pred, full_pred, target)
    assert losses.local.item() == sum((i + 1) ** 2 for i in range(4))
    assert losses.full.item() == sum((2 * (i + 1)) ** 2 for i in range(4))
    assert losses.train.item() == losses.full.item() + losses.local.item()
