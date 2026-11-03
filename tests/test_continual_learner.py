import torch

from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.continual import ContinualLearner


def _tiny_model():
    return SelfStabilizingDynamicMorphicMoE(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=1,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
    )


def test_step_runs_and_returns_expected_metric_keys():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, optimizer, mutation_check_interval=100)

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))
    metrics = learner.step(x, targets, output_head="token")

    for key in ["task_loss", "rehearsal_loss", "distillation_loss", "total_loss", "novelty_score", "mutation_report"]:
        assert key in metrics
    assert metrics["mutation_report"] is None


def test_step_populates_replay_buffer_up_to_capacity():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, optimizer, replay_buffer_capacity=3, mutation_check_interval=100)

    for step in range(5):
        x = torch.randint(0, 50, (2, 4), dtype=torch.long)
        targets = torch.randint(0, 50, (2, 4))
        learner.step(x, targets, output_head="token")
        assert len(learner.replay_buffer) == min(step + 1, 3)


def test_replay_loss_is_nonzero_once_buffer_has_items():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    learner = ContinualLearner(
        model, optimizer, replay_buffer_capacity=10, replay_batch_size=2, mutation_check_interval=100
    )

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))
    learner.step(x, targets, output_head="token")
    metrics = learner.step(x, targets, output_head="token")

    assert metrics["rehearsal_loss"] > 0.0
    assert metrics["distillation_loss"] > 0.0


def test_mutation_check_only_fires_on_interval_boundary():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, optimizer, mutation_check_interval=3)

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))

    reports = []
    for _ in range(6):
        metrics = learner.step(x, targets, output_head="token")
        reports.append(metrics["mutation_report"])

    assert reports[0] is None
    assert reports[1] is None
    assert reports[2] is not None
    assert reports[3] is None
    assert reports[4] is None
    assert reports[5] is not None


def test_continual_learner_applies_thresholds_freezes_experts_and_rebuilds_optimizer():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(
        model,
        optimizer,
        maturity_age=2,
        tau_novelty=0.0,
        entropy_floor=0.0,
        theta_novelty_vertical=100.0,
        mutation_check_interval=2,
    )

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))

    learner.step(x, targets, output_head="token")
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
    metrics = learner.step(x, targets, output_head="token")

    assert metrics["mutation_report"] is not None
    # Initial 2 experts reached age >= 2 and were frozen; newly spawned expert (index 2) is unfrozen
    assert model.layers[0].moe.num_experts == 3
    assert bool(model.layers[0].moe.expert_frozen[0]) is True
    assert bool(model.layers[0].moe.expert_frozen[1]) is True
    assert bool(model.layers[0].moe.expert_frozen[2]) is False
    # Optimizer includes the newly spawned expert's parameters
    opt_param_ids = {id(p) for g in learner.optimizer.param_groups for p in g["params"]}
    assert id(model.layers[0].moe.experts[2].linear1.weight_real) in opt_param_ids

