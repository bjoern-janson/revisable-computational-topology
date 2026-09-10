from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .config import BobConfig, WorldConfig
from .types import NODE_IDS, NodeId


@dataclass(frozen=True)
class PredictionLosses:
    local: torch.Tensor
    full: torch.Tensor
    train: torch.Tensor


class BobBackbone(nn.Module):
    def __init__(self, world_config: WorldConfig, bob_config: BobConfig):
        super().__init__()
        self.world_config = world_config
        self.bob_config = bob_config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(world_config.network_seed)
            self.encoders = nn.ModuleDict({
                n: nn.Sequential(
                    nn.Linear(world_config.obs_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, bob_config.hidden_dim),
                ) for n in NODE_IDS
            })
            self.local_decoders = nn.ModuleDict({
                n: nn.Sequential(
                    nn.Linear(bob_config.hidden_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, world_config.obs_dim),
                ) for n in NODE_IDS
            })
            self.full_decoders = nn.ModuleDict({
                n: nn.Sequential(
                    nn.Linear(2 * bob_config.hidden_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, world_config.obs_dim),
                ) for n in NODE_IDS
            })

    def encode(self, observations: dict[NodeId, torch.Tensor]) -> dict[NodeId, torch.Tensor]:
        return {n: self.encoders[n](observations[n]) for n in NODE_IDS}

    def decode_local(self, latents: dict[NodeId, torch.Tensor]) -> dict[NodeId, torch.Tensor]:
        return {n: self.local_decoders[n](latents[n]) for n in NODE_IDS}

    def decode_full(
        self,
        latents: dict[NodeId, torch.Tensor],
        incoming_messages: dict[NodeId, list[torch.Tensor]],
    ) -> dict[NodeId, torch.Tensor]:
        out: dict[NodeId, torch.Tensor] = {}
        for n in NODE_IDS:
            messages = incoming_messages.get(n, [])
            if messages:
                message = torch.stack(messages).sum(dim=0)
            else:
                message = torch.zeros_like(latents[n])
            out[n] = self.full_decoders[n](torch.cat([latents[n], message], dim=-1))
        return out

    @staticmethod
    def aggregate_losses(per_target: dict[NodeId, torch.Tensor]) -> float:
        return float(sum(v.item() for v in per_target.values()))

    def losses(
        self,
        local_predictions: dict[NodeId, torch.Tensor],
        full_predictions: dict[NodeId, torch.Tensor],
        targets: dict[NodeId, torch.Tensor],
    ) -> PredictionLosses:
        local_terms = [F.mse_loss(local_predictions[n], targets[n], reduction="mean") for n in NODE_IDS]
        full_terms = [F.mse_loss(full_predictions[n], targets[n], reduction="mean") for n in NODE_IDS]
        l_local = torch.stack(local_terms).sum()
        l_full = torch.stack(full_terms).sum()
        return PredictionLosses(
            local=l_local,
            full=l_full,
            train=l_full + self.bob_config.local_loss_weight * l_local,
        )
