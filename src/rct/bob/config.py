from dataclasses import dataclass


@dataclass(frozen=True)
class BobConfig:
    hidden_dim: int = 32
    interface_max_rank: int = 8
    interface_initial_rank: int = 2
    structural_interval: int = 64
    probe_window: int = 256
    probes_per_window: int = 2
    probe_cost: float = 0.02
    create_cost: float = 0.25
    modify_cost: float = 0.10
    dormant_cost: float = 0.05
    retire_cost: float = 0.05
    edit_budget_per_window: float = 0.50
    maintenance_cost_per_rank: float = 0.001
    manager_warmup_steps: int = 256
    local_loss_weight: float = 1.0
    task_opportunity_min_gain: float = 0.02


@dataclass(frozen=True)
class WorldConfig:
    latent_dim: int = 8
    obs_dim: int = 16
    emission_hidden_dim: int = 32
    rho: float = 0.85
    alpha: float = 0.6
    beta: float = 0.8
    emission_noise_std: float = 0.05
    w1_steps_min: int = 1536
    w1_steps_max: int = 2560
    w2_steps_min: int = 1536
    w2_steps_max: int = 2560
    w3_steps_min: int = 1536
    w3_steps_max: int = 2560
    world_seed: int = 7
    emission_seed: int = 17
    network_seed: int = 29
    manager_seed: int = 31
    probe_seed: int = 37
    audit_seed: int = 41


@dataclass(frozen=True)
class MkIIConfig:
    return_horizon_steps: int = 256
    eta_ridge_lambda: float = 1.0
    feature_dim: int = 38
    feature_eps: float = 1e-6
    feature_clip: float = 2.0
    relation_candidates_per_boundary: int = 2
    exploration_block_size: int = 4
    exploration_slots_per_block: int = 1
    exploration_seed: int = 43
    tie_break_seed: int = 47
    retire_support_min_rows: int = 6
    retire_support_min_action_kinds: int = 2
