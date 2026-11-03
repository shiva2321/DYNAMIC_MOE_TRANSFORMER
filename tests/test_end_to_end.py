import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def _small_model():
    return SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        initial_experts=2,
        theta_up=0.4,
        theta_down=0.05,
        min_layers=1,
        max_layers=4,
    )


def test_sequential_multimodal_forward_passes_succeed():
    model = _small_model()
    model.eval()

    text = torch.randint(0, 500, (2, 6), dtype=torch.long)
    out_text = model(text, output_head="token")
    assert out_text["output"].shape[:2] == (2, 6)

    image = torch.randn(2, 3, 16, 16)
    out_image = model(image, output_head="continuous")
    assert out_image["output"].shape[0] == 2

    numeric = torch.randn(2, 10, 5)
    out_numeric = model(numeric, output_head="continuous")
    assert out_numeric["output"].shape[0] == 2


def test_short_optimization_loop_runs_without_errors_and_loss_is_finite():
    torch.manual_seed(0)
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    text = torch.randint(0, 500, (2, 6), dtype=torch.long)
    targets = torch.randint(0, 500, (2, 6))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = model(text, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss = loss + 0.1 * out["structural_error"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert torch.isfinite(loss).all()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_full_mutation_cycle_then_optimization_step_does_not_error():
    model = _small_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()

    model.layers[0].moe._last_entropy = torch.tensor(5.0)
    model.execute_graph_mutation()

    with torch.no_grad():
        model.layers[0].moe.expert_usage[0] = 0.0001
    model.layers[0].moe._last_entropy = torch.tensor(0.1)
    model.execute_graph_mutation()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    text = torch.randint(0, 500, (2, 6), dtype=torch.long)
    targets = torch.randint(0, 500, (2, 6))

    out = model(text, output_head="token")
    loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).all()
