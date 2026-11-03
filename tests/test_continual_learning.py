import random
import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.continual import ContinualLearner


def _retention_model():
    return SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
    )


def _task_a_batch(batch_size=8, seq_len=6):
    tokens = torch.randint(0, 50, (batch_size, seq_len), dtype=torch.long)
    targets = (tokens + 1) % 50
    return tokens, targets


def _task_b_batch(batch_size=8, seq_len=6):
    tokens = torch.randint(100, 150, (batch_size, seq_len), dtype=torch.long)
    targets = 100 + (tokens * 2) % 50
    return tokens, targets


def _eval_task_a_loss(model, eval_x, eval_targets):
    model.eval()
    with torch.no_grad():
        out = model(eval_x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), eval_targets.view(-1))
    model.train()
    return loss.item()


def test_continual_learner_retains_task_a_better_than_naive_baseline():
    random.seed(0)
    torch.manual_seed(0)
    mitigated = _retention_model()
    torch.manual_seed(0)
    baseline = _retention_model()

    eval_x, eval_targets = _task_a_batch(batch_size=16)

    mitigated_optimizer = torch.optim.AdamW(mitigated.parameters(), lr=5e-3)
    learner = ContinualLearner(
        mitigated,
        mitigated_optimizer,
        replay_buffer_capacity=200,
        replay_batch_size=8,
        distillation_weight=0.5,
        mutation_check_interval=1000,  # far beyond this test's step count: isolates the retention
    )                                   # effect from architecture mutation, which Task 3 already covers
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=5e-3)

    train_steps_per_task = 40

    for _ in range(train_steps_per_task):
        x, targets = _task_a_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_a_loss_after_a = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_a_loss_after_a = _eval_task_a_loss(baseline, eval_x, eval_targets)

    for _ in range(train_steps_per_task):
        x, targets = _task_b_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_a_loss_after_b = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_a_loss_after_b = _eval_task_a_loss(baseline, eval_x, eval_targets)

    mitigated_forgetting = mitigated_a_loss_after_b - mitigated_a_loss_after_a
    baseline_forgetting = baseline_a_loss_after_b - baseline_a_loss_after_a

    print(
        f"\nRetention check: mitigated_after_a={mitigated_a_loss_after_a:.4f}, "
        f"mitigated_after_b={mitigated_a_loss_after_b:.4f}, "
        f"mitigated_forgetting={mitigated_forgetting:.4f} | "
        f"baseline_after_a={baseline_a_loss_after_a:.4f}, "
        f"baseline_after_b={baseline_a_loss_after_b:.4f}, "
        f"baseline_forgetting={baseline_forgetting:.4f}"
    )

    assert mitigated_forgetting < baseline_forgetting
    assert mitigated_a_loss_after_b < baseline_a_loss_after_b


def test_continual_learner_retains_from_identical_task_a_state():
    random.seed(1)
    torch.manual_seed(1)
    mitigated = _retention_model()
    eval_x, eval_targets = _task_a_batch(batch_size=16)

    mitigated_optimizer = torch.optim.AdamW(mitigated.parameters(), lr=5e-3)
    learner = ContinualLearner(
        mitigated,
        mitigated_optimizer,
        replay_buffer_capacity=200,
        replay_batch_size=4,
        distillation_weight=0.5,
        maturity_age=20,
        mutation_check_interval=20,
    )

    for _ in range(30):
        x, targets = _task_a_batch()
        learner.step(x, targets, output_head="token")

    # Clone exact model weights after Task A so both start Task B at identical Task-A loss
    baseline = _retention_model()
    baseline.load_state_dict({k: v.clone() for k, v in mitigated.state_dict().items()})
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=5e-3)

    loss_after_a = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    assert abs(loss_after_a - _eval_task_a_loss(baseline, eval_x, eval_targets)) < 1e-6

    for _ in range(30):
        x, targets = _task_b_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_after_b = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_after_b = _eval_task_a_loss(baseline, eval_x, eval_targets)

    assert mitigated_after_b < baseline_after_b

