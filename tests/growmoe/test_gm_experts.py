import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.experts import MLPExpert, agem_project

D = 32


def _toy(n, seed, n_classes=3, noise=0.5):
    g = np.random.default_rng(seed)
    centres = np.random.default_rng(1234).standard_normal((n_classes, D))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    y = g.integers(0, n_classes, size=n)
    x = centres[y] + noise * g.standard_normal((n, D)) / np.sqrt(D)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()


def _units(n, seed, d=D):
    g = np.random.default_rng(seed)
    x = g.standard_normal((n, d))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()


def _expert(**kw):
    kw.setdefault("d_in", D)
    kw.setdefault("hidden", 16)
    kw.setdefault("n_classes", 8)
    e = MLPExpert(**kw)
    e.mark_seen(torch.arange(kw["n_classes"]))
    return e


def _grads(expert, x, y):
    params = [p for p in expert.parameters() if p.requires_grad]
    gs = torch.autograd.grad(F.cross_entropy(expert(x), y), params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])


class _Grab:
    """Optimiser stand-in that records the gradient it is asked to apply, and applies nothing."""
    def __init__(self, params):
        self.params = list(params)
        self.applied = None

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        self.applied = torch.cat([p.grad.flatten() for p in self.params])


# ----------------------------------------------------------------- forward / masking
def test_logits_shape_and_unseen_classes_are_masked():
    e = MLPExpert(d_in=D, hidden=16, n_classes=10, seed=0)
    u = _units(5, 0)
    assert e(u).shape == (5, 10)
    assert (e(u) == -1e9).all()                       # nothing seen yet
    e.mark_seen(torch.tensor([2, 5, 2]))
    logits = e(u)
    seen = torch.zeros(10, dtype=torch.bool)
    seen[[2, 5]] = True
    assert (logits[:, ~seen] == -1e9).all()
    assert (logits[:, seen] > -1e8).all()
    e.mark_seen(torch.tensor([2]))                    # idempotent
    assert torch.equal(e.seen, seen)


def test_proba_is_a_distribution_over_seen_classes_only():
    e = MLPExpert(d_in=D, hidden=16, n_classes=10, seed=0)
    e.mark_seen(torch.tensor([1, 4, 7]))
    p = e.proba(_units(6, 1))
    assert torch.allclose(p.sum(dim=1), torch.ones(6), atol=1e-5)
    assert (p[:, [0, 2, 3, 5, 6, 8, 9]] == 0).all()


def test_loss_has_no_side_effects_and_unseen_targets_are_hugely_penalised():
    e = MLPExpert(d_in=D, hidden=16, n_classes=6, seed=0)
    e.mark_seen(torch.tensor([0, 1]))
    u = _units(4, 2)
    before = e.seen.clone()
    ok = e.loss(u, torch.tensor([0, 1, 0, 1]))
    bad = e.loss(u, torch.tensor([0, 1, 0, 5]))
    assert torch.equal(e.seen, before)
    assert ok.item() < 5.0 and bad.item() > 1e8


def test_same_seed_same_function_different_seed_different():
    u = _units(8, 3)
    a, b, c = _expert(seed=0), _expert(seed=0), _expert(seed=1)
    assert torch.equal(a(u), b(u))
    assert not torch.allclose(a(u), c(u), atol=1e-3)


def test_learns_a_three_class_problem():
    torch.manual_seed(0)
    e = MLPExpert(d_in=D, hidden=32, n_classes=8, seed=0)
    opt = torch.optim.Adam(e.parameters(), lr=3e-3)
    x, y = _toy(600, 0)
    for step in range(150):
        i = torch.randint(0, 600, (64,))
        e.train_step(x[i], y[i], opt)
    xt, yt = _toy(300, 99)
    assert (e(xt).argmax(dim=1) == yt).float().mean().item() > 0.95


# ------------------------------------------------------------------------- clone
def test_clone_is_identical_and_independent():
    e = _expert(seed=3)
    c = e.clone()
    u = _units(6, 4)
    assert torch.equal(e(u), c(u)) and torch.equal(e.seen, c.seen)
    with torch.no_grad():
        c.head.weight.add_(1.0)
    c.mark_seen(torch.tensor([7]))
    assert not torch.equal(e(u), c(u))


# ------------------------------------------------------------------------ growth
@pytest.mark.parametrize("op", ["widen", "deepen"])
def test_growth_is_function_preserving_exactly_in_float64_and_closely_in_float32(op):
    def grow(e):
        e.widen(16) if op == "widen" else e.deepen()
        return e

    u = _units(10, 5)
    e64 = _expert(depth=2, seed=2).double()
    before = e64(u.double())
    n_before = e64.num_params()
    grow(e64)
    assert torch.allclose(e64(u.double()), before, atol=1e-12, rtol=0)
    assert e64.num_params() > n_before

    e32 = _expert(depth=2, seed=2)
    before32 = e32(u)
    grow(e32)
    assert torch.allclose(e32(u), before32, atol=1e-5, rtol=0)


def test_widen_adds_hidden_units_to_every_block_and_deepen_adds_one_block():
    e = _expert(depth=3, hidden=16)
    e.widen(8)
    assert e.hidden == 24 and all(b.w1.shape[0] == 24 and b.w2.shape[1] == 24 for b in e.blocks)
    e.deepen()
    assert len(e.blocks) == 4 and e.blocks[-1].w1.shape[0] == 24


def test_growth_is_deterministic_given_the_seed():
    a, b = _expert(seed=4), _expert(seed=4)
    a.widen(8)
    b.widen(8)
    a.deepen()
    b.deepen()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_new_capacity_is_trainable_after_one_step():
    e = _expert(hidden=16)
    e.widen(8)
    opt = torch.optim.Adam(e.parameters(), lr=1e-2)
    x, y = _toy(64, 7)
    assert (e.blocks[0].w2[:, 16:] == 0).all()
    e.train_step(x, y, opt)
    assert (e.blocks[0].w2[:, 16:] != 0).any()


def test_deepened_block_is_trainable_after_one_step():
    e = _expert(hidden=16)
    e.deepen()
    opt = torch.optim.Adam(e.parameters(), lr=1e-2)
    x, y = _toy(64, 8)
    assert (e.blocks[-1].w2 == 0).all()
    e.train_step(x, y, opt)
    assert (e.blocks[-1].w2 != 0).any()


# ---------------------------------------------------------------------- inherit
def test_inherit_with_no_redraw_and_no_noise_is_a_clone():
    e = _expert(depth=2, seed=5)
    c = e.inherit(0.0, 0.0, np.random.default_rng(0))
    u = _units(6, 9)
    assert torch.equal(e(u), c(u))


def test_inherit_with_full_redraw_changes_the_function_on_nearly_every_input():
    e = _expert(depth=2, seed=5)
    c = e.inherit(1.0, 0.0, np.random.default_rng(0))
    u = _units(500, 10)
    changed = ((e(u) - c(u)).abs().amax(dim=1) > 1e-3).float().mean().item()
    assert changed > 0.99


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_inherit_redraws_round_rho_h_hidden_units_per_block(rho):
    e = _expert(depth=3, hidden=64, seed=6)
    c = e.inherit(rho, 0.0, np.random.default_rng(1))
    expect = round(rho * 64)
    for pb, cb in zip(e.blocks, c.blocks):
        redrawn = (pb.w1 != cb.w1).any(dim=1)
        assert int(redrawn.sum()) == expect
        assert torch.equal(redrawn, (pb.w2 != cb.w2).any(dim=0))     # a unit's in- and out-weights move together
    assert torch.equal(e.head.weight, c.head.weight)                 # the head and norms are inherited verbatim


def test_inherit_noise_scale_is_sigma_times_the_weight_std():
    e = _expert(depth=1, hidden=128, d_in=64, seed=6)
    c = e.inherit(0.0, 0.1, np.random.default_rng(2))
    for w_p, w_c in ((e.blocks[0].w1, c.blocks[0].w1), (e.blocks[0].w2, c.blocks[0].w2)):
        ratio = ((w_c - w_p).std() / w_p.std()).item()
        assert 0.09 < ratio < 0.11


def test_inherit_is_deterministic_given_the_rng():
    e = _expert(depth=2, seed=7)
    a = e.inherit(0.5, 0.05, np.random.default_rng(3))
    b = e.inherit(0.5, 0.05, np.random.default_rng(3))
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


# ----------------------------------------------------------- replay / A-GEM
def test_train_step_returns_the_new_batch_loss_before_the_update():
    e = _expert(seed=8)
    x, y = _toy(32, 11)
    xr, yr = _toy(32, 12)
    expected = e.loss(x, y).item()
    opt = torch.optim.SGD(e.parameters(), lr=0.1)
    assert e.train_step(x, y, opt, replay=(xr, yr)) == pytest.approx(expected, abs=1e-5)


def test_agem_project_on_hand_computable_vectors():
    g, g_ref = torch.tensor([1.0, -1.0]), torch.tensor([0.0, 1.0])          # dot = -1: conflict
    assert torch.allclose(agem_project(g, g_ref), torch.tensor([1.0, 0.0]))
    g = torch.tensor([1.0, 1.0])                                            # dot = +1: left alone
    assert torch.equal(agem_project(g, g_ref), g)
    g = torch.tensor([1.0, 0.0])                                            # dot = 0: left alone
    assert torch.equal(agem_project(g, g_ref), g)


def test_agem_project_never_increases_the_reference_loss_to_first_order():
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        g, g_ref = torch.randn(500, generator=gen), torch.randn(500, generator=gen)
        out = agem_project(g, g_ref)
        assert (out @ g_ref).item() >= -1e-4 * out.norm().item() * g_ref.norm().item()
        assert out.norm().item() <= g.norm().item() + 1e-6
        if (g @ g_ref).item() >= 0:
            assert torch.equal(out, g)
        else:
            assert abs((out @ g_ref).item()) < 1e-4 * g.norm().item() * g_ref.norm().item()


def test_agem_projection_in_train_step_removes_the_conflicting_component():
    # With two seen classes the CE gradient of one input is p1*delta for label 0 and -p0*delta for label 1
    # (delta = d(z1 - z0)/d theta), so the same input under opposite labels is an exact anti-parallel conflict.
    # The new batch adds a perturbed copy so that a non-zero remainder survives the projection.
    # (With many classes or many unrelated inputs the sign is decided by cross terms, hence this construction.)
    e = _expert(hidden=16, n_classes=2, seed=9)
    x1 = _units(1, 13)
    x2 = F.normalize(x1 + 0.3 * _units(1, 14), dim=1)
    x, y_new, y_ref = torch.cat([x1, x2]), torch.tensor([1, 1]), torch.tensor([0])
    g_new, g_ref = _grads(e, x, y_new), _grads(e, x1, y_ref)
    assert (g_new @ g_ref).item() < 0, "scenario must be a real gradient conflict"
    grab = _Grab(e.parameters())
    e.train_step(x, y_new, grab, replay=(x1, y_ref), project=True)
    applied = grab.applied
    assert torch.allclose(applied, agem_project(g_new, g_ref), atol=1e-6)
    assert abs((applied @ g_ref).item()) < 1e-4 * applied.norm().item() * g_ref.norm().item()
    assert 0.0 < applied.norm().item() < g_new.norm().item()                # something remains, something was removed


def test_agem_projection_is_a_no_op_without_conflict_and_without_replay():
    e = _expert(hidden=16, seed=9)
    x, y = _toy(48, 14)
    g = _grads(e, x, y)
    grab = _Grab(e.parameters())
    e.train_step(x, y, grab, replay=(x, y), project=True)             # identical gradients: dot > 0
    assert torch.allclose(grab.applied, g, atol=1e-6)
    grab = _Grab(e.parameters())
    e.train_step(x, y, grab, replay=None, project=True)               # nothing to project against
    assert torch.allclose(grab.applied, g, atol=1e-6)


def test_replay_without_projection_mixes_the_replay_gradient_in():
    e = _expert(hidden=16, seed=9)
    x, y = _toy(48, 15)
    xr, yr = _toy(48, 16)
    grab = _Grab(e.parameters())
    e.train_step(x, y, grab, replay=(xr, yr), project=False)
    both = _grads(e, torch.cat([x, xr]), torch.cat([y, yr]))
    assert torch.allclose(grab.applied, both, atol=1e-5)


# ------------------------------------------------------------- shrink-and-perturb (child initialisation)
def test_shrink_perturb_with_lambda_zero_is_exactly_a_fresh_expert_and_with_one_a_clone():
    parent = _expert(depth=2, seed=3)
    with torch.no_grad():
        for p in parent.parameters():
            p.add_(0.5)                                                  # make it clearly different from any fresh init
    fresh = MLPExpert(d_in=D, hidden=16, depth=2, n_classes=8, seed=11)
    zero = parent.shrink_perturb(0.0, seed=11)
    assert all(torch.equal(a, b) for a, b in zip(zero.parameters(), fresh.parameters()))
    one = parent.shrink_perturb(1.0, seed=11)
    assert all(torch.equal(a, b) for a, b in zip(one.parameters(), parent.parameters()))


def test_shrink_perturb_is_the_convex_combination_of_parent_and_fresh_weights():
    parent = _expert(seed=4)
    fresh = MLPExpert(d_in=D, hidden=16, depth=1, n_classes=8, seed=5)
    child = parent.shrink_perturb(0.3, seed=5)
    for c, p, f in zip(child.parameters(), parent.parameters(), fresh.parameters()):
        assert torch.allclose(c, 0.3 * p + 0.7 * f, atol=1e-6)


def test_shrink_perturb_keeps_the_label_mask_leaves_the_parent_alone_and_validates_lambda():
    parent = _expert(seed=6)
    before = [p.detach().clone() for p in parent.parameters()]
    child = parent.shrink_perturb(0.5, seed=1)
    assert torch.equal(child.seen, parent.seen)
    assert all(torch.equal(a, b) for a, b in zip(before, parent.parameters()))
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            parent.shrink_perturb(bad)


def test_shrink_perturb_works_on_a_grown_parent():
    parent = _expert(depth=1, hidden=8, seed=7)
    parent.widen(8)
    parent.deepen()
    child = parent.shrink_perturb(0.5, seed=2)
    assert child.hidden == 16 and len(child.blocks) == 2
    assert child(_units(4, 3)).shape == (4, 8)


def test_shrink_perturb_output_moves_from_the_parents_towards_a_fresh_expert_as_lambda_falls():
    parent = _expert(seed=8)
    x = _units(64, 9)
    gaps = [(parent.shrink_perturb(lam, seed=3)(x) - parent(x)).abs().mean().item() for lam in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert gaps[0] == 0.0 and all(a < b for a, b in zip(gaps, gaps[1:]))
