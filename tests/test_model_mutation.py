import pytest
import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE, rebuild_optimizer_preserving_state


def _tiny_model(**overrides):
    kwargs = dict(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=2,
        initial_experts=2,
        theta_up=0.4,
        theta_down=0.05,
        min_layers=1,
        max_layers=4,
    )
    kwargs.update(overrides)
    return SelfStabilizingDynamicMorphicMoE(**kwargs)


def test_vertical_growth_triggers_above_theta_up():
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    report = model.execute_graph_mutation()
    assert model.num_layers == 3
    assert any("Vertical Growth" in log for log in report["mutation_logs"])


def test_vertical_growth_respects_max_layers():
    model = _tiny_model(initial_layers=4, max_layers=4)
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()
    assert model.num_layers == 4


def test_vertical_pruning_triggers_below_theta_down():
    model = _tiny_model(initial_layers=3, min_layers=1)
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
        model.layers[1].layer_salience.fill_(0.01)
    report = model.execute_graph_mutation()
    assert model.num_layers == 2
    assert any("Vertical Pruning" in log for log in report["mutation_logs"])


def test_vertical_pruning_respects_min_layers():
    model = _tiny_model(initial_layers=1, min_layers=1)
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
        model.layers[0].layer_salience.fill_(0.01)
    model.execute_graph_mutation()
    assert model.num_layers == 1


def test_checkpoint_round_trip_reproduces_output(tmp_path):
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out_before = model(x)["output"]

    path = tmp_path / "ckpt.pt"
    model.save_checkpoint(str(path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(path))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)["output"]

    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_checkpoint_round_trip_after_mutation_with_nonuniform_experts(tmp_path):
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()

    model.layers[0].moe._last_entropy = torch.tensor(5.0)
    model.execute_graph_mutation()

    experts_before = [l.moe.num_experts for l in model.layers]
    assert len(set(experts_before)) > 1  # proves per-layer expert counts diverged

    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    num_x = torch.randn(1, 5, 6)
    with torch.no_grad():
        out_before = model(x)["output"]
        out_num_before = model(num_x, output_head="continuous")["output"]

    path = tmp_path / "ckpt_mutated.pt"
    model.save_checkpoint(str(path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(path))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)["output"]
        out_num_after = loaded(num_x, output_head="continuous")["output"]

    assert [l.moe.num_experts for l in loaded.layers] == experts_before
    assert torch.allclose(out_before, out_after, atol=1e-6)
    assert torch.allclose(out_num_before, out_num_after, atol=1e-6)


def test_rebuild_optimizer_preserves_state_for_surviving_params():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    targets = torch.randint(0, 100, (1, 4))
    out = model(x)["output"]
    loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()

    tracked_param = model.layers[0].moe.experts[0].linear1.weight_real
    state_before = optimizer.state[tracked_param]["exp_avg"].clone()

    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()

    new_param = model.layers[-1].moe.experts[0].linear1.weight_real
    new_optimizer = rebuild_optimizer_preserving_state(optimizer, model, lr=1e-3)
    state_after = new_optimizer.state[tracked_param]["exp_avg"]
    assert torch.equal(state_before, state_after)
    assert new_param not in new_optimizer.state


def test_model_mutate_and_move_to_device_is_consistent():
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()
    model.layers[0].moe._last_entropy = torch.tensor(5.0)
    model.execute_graph_mutation()

    devices = {p.device for p in model.parameters()}
    assert len(devices) == 1

    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)["output"]
    assert out.device == next(model.parameters()).device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_runs_on_cuda_after_mutation():
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()
    model = model.to("cuda")
    x = torch.randint(0, 100, (1, 4), dtype=torch.long, device="cuda")
    out = model(x)["output"]
    assert out.device.type == "cuda"


def test_vertical_growth_triggers_from_novelty_score_alone():
    model = _tiny_model(theta_up=10.0)  # make the error-magnitude trigger unreachable
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
        model.manifold_corr.last_novelty_score = torch.tensor(5.0)
    report = model.execute_graph_mutation()
    assert model.num_layers == 3
    assert any("Novelty Z-Score" in log for log in report["mutation_logs"])


def test_execute_graph_mutation_accepts_explicit_novelty_override():
    model = _tiny_model(theta_up=10.0)
    report = model.execute_graph_mutation(novelty_score=5.0)
    assert model.num_layers == 3
    assert any("Novelty Z-Score" in log for log in report["mutation_logs"])


def test_forward_returns_novelty_score_in_output_dict():
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert "novelty_score" in out
    assert out["novelty_score"].dim() == 0


def test_checkpoint_round_trip_with_simulation_loop_enabled(tmp_path):
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=2,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
        enable_simulation_loop=True,
        max_simulation_steps=3,
        theta_converge=0.1,
        min_improvement=0.01,
        skip_threshold=-100.0,
    )
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out_before = model(x)["output"]

    path = tmp_path / "ckpt_sim_loop.pt"
    model.save_checkpoint(str(path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(path))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)["output"]

    assert loaded.enable_simulation_loop is True
    assert hasattr(loaded, "loop_damping_logit")
    assert torch.equal(model.loop_damping_logit, loaded.loop_damping_logit)
    assert torch.allclose(out_before, out_after, atol=1e-6)
