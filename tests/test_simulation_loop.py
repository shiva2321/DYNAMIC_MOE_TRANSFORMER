import torch

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def _sim_model(**overrides):
    kwargs = dict(
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
        max_simulation_steps=4,
        theta_converge=0.1,
        min_improvement=0.01,
        skip_threshold=0.0,
    )
    kwargs.update(overrides)
    return SelfStabilizingDynamicMorphicMoE(**kwargs)


def test_disabled_by_default_matches_phase1_behavior():
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=8, num_heads=2, d_ff=16, initial_layers=1, initial_experts=2
    )
    assert model.enable_simulation_loop is False
    assert not hasattr(model, "loop_damping_logit")

    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 0


def test_training_mode_uses_randomized_step_count_within_bounds():
    # skip_threshold set absurdly low so the skip-hint never fires here --
    # this test is isolating randomized-depth behavior, not the skip-hint.
    model = _sim_model(max_simulation_steps=4, skip_threshold=-100.0)
    model.train()

    seen_counts = set()
    x = torch.randint(0, 100, (2, 4), dtype=torch.long)
    for _ in range(30):
        out = model(x)
        count = out["simulation_steps_used"]
        assert 1 <= count <= 4
        seen_counts.add(count)

    assert len(seen_counts) > 1


def test_eval_mode_halts_immediately_when_error_already_below_convergence_threshold():
    # theta_converge set absurdly high so the very first iteration's error
    # is guaranteed to be below it, regardless of the actual numeric value.
    model = _sim_model(max_simulation_steps=4, theta_converge=1000.0, skip_threshold=-100.0)
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 1


def test_eval_mode_runs_to_max_steps_when_error_stays_high_and_does_not_improve_enough():
    # theta_converge and min_improvement set unreachable so neither halting
    # condition can fire -- the loop must run the full max_simulation_steps.
    model = _sim_model(
        max_simulation_steps=4, theta_converge=-1000.0, min_improvement=-1000.0, skip_threshold=-100.0
    )
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 4


def test_skip_hint_prevents_loop_even_when_enabled():
    # skip_threshold set absurdly high so any real computed novelty score
    # falls at or below it, regardless of the actual numeric value.
    model = _sim_model(skip_threshold=1000.0)
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 0


def test_damped_loop_stays_bounded_under_forced_excessive_iterations():
    # Forces far more iterations than any sane trained cap (20, vs. the
    # 4-ish range this phase trains with) and asserts the damping
    # mechanism keeps the latent state finite and bounded -- direct
    # evidence against the divergence this whole design exists to avoid.
    model = _sim_model(
        max_simulation_steps=20, theta_converge=-1000.0, min_improvement=-1000.0, skip_threshold=-100.0
    )
    model.eval()
    x = torch.randint(0, 100, (2, 6), dtype=torch.long)
    out = model(x)

    max_r = out["z_real"].abs().max().item()
    max_i = out["z_imag"].abs().max().item()
    print(f"\nForced 20-step loop max |z_real|={max_r:.4f}, max |z_imag|={max_i:.4f}")

    assert out["simulation_steps_used"] == 20
    assert torch.isfinite(out["z_real"]).all()
    assert torch.isfinite(out["z_imag"]).all()
    assert max_r < 1e4
    assert max_i < 1e4


def test_gradients_flow_through_damped_loop_including_damping_parameter():
    model = _sim_model(max_simulation_steps=3, skip_threshold=-100.0)
    model.train()
    x = torch.randint(0, 100, (2, 4), dtype=torch.long)

    out = model(x, output_head="token")
    loss = out["output"].sum()
    loss.backward()

    assert model.loop_damping_logit.grad is not None
    assert not torch.isnan(model.loop_damping_logit.grad).any()
    assert model.layers[-1].attn.q_proj.weight_real.grad is not None
    assert not torch.isnan(model.layers[-1].attn.q_proj.weight_real.grad).any()
