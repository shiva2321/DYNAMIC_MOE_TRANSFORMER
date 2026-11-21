"""
Unit & invariant tests for dmoe/belief_training.py (belief_training_lab_v1).
Verifies task anti-cheat invariants, strict causality, SIGReg behavior,
ConceptBeliefStore burst-immunity/context-splitting/Mechanism-P/byte-cap,
BONG precision tracking, and zero-backward MuPC local energy gradients.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.belief_training import (
    BONGPrecisionOptimizer,
    BudgetedSurfaceDict,
    ConceptBeliefBackbone,
    ConceptBeliefStore,
    K_CONCEPTS,
    MuPCProspectiveNetwork,
    NullSpaceGuard,
    Q_ATTR,
    Q_PROP,
    Q_VAL,
    SYM_OFFSET,
    SparseConceptQuantizer,
    equivalence_and_contradiction_losses,
    sigreg_loss,
    BeliefMeaningTask,
)


def test_task_zero_leakage_and_uniform_marginals():
    task = BeliefMeaningTask(seed=100)
    assert len(task.seen_pairs) == 192
    assert len(task.heldout_pairs) == 64
    assert set(task.seen_pairs).isdisjoint(set(task.heldout_pairs))
    assert set(task.s_split).isdisjoint(set(task.s_rev))
    assert set(task.s_split).isdisjoint(set(task.s_static))
    assert set(task.s_rev).isdisjoint(set(task.s_static))

    # Every symbol appears with balanced 50/50 context parity over c2
    for c1 in range(K_CONCEPTS):
        c1_t = torch.full((K_CONCEPTS,), c1, dtype=torch.long)
        c2_t = torch.arange(K_CONCEPTS, dtype=torch.long)
        regimes = task.ctx_regime(c1_t, c2_t)
        assert int(regimes.sum().item()) == K_CONCEPTS // 2

    # Every operator produces a Latin square (each column & row has all 16 values)
    for op in range(4):
        for row in range(K_CONCEPTS):
            assert sorted(task.val_table[op, row].tolist()) == list(range(K_CONCEPTS))
        for col in range(K_CONCEPTS):
            assert sorted(task.val_table[op, :, col].tolist()) == list(range(K_CONCEPTS))

    # Alternative attributes and revised properties strictly differ from base
    assert bool((task.g_alt != task.g_base).all().item())
    assert bool((task.h_rev != task.h_init).all().item())


def test_backbone_strict_causality():
    torch.manual_seed(7)
    model = ConceptBeliefBackbone(d_model=64, num_heads=4, num_layers=2, d_ff=192, use_concept_readout=True)
    model.eval()
    x = torch.tensor([
        [0, 5, 9, 6, 12, Q_VAL],
        [2, 11, 4, 15, 8, Q_ATTR],
    ], dtype=torch.long)
    out0 = model(x)

    # Perturbing c1, c2, q_tok (positions >= 3) must NOT alter z_expr (position 2)
    x_fut_expr = x.clone()
    x_fut_expr[:, 3] = 17
    x_fut_expr[:, 4] = 14
    x_fut_expr[:, 5] = Q_PROP
    out1 = model(x_fut_expr)
    diff_expr = float((out0["z_expr"] - out1["z_expr"]).abs().max().item())
    assert diff_expr <= 1e-5

    # Perturbing q_tok (position 5) must NOT alter z_ctx (position 4)
    x_fut_ctx = x.clone()
    x_fut_ctx[:, 5] = Q_PROP
    out2 = model(x_fut_ctx)
    diff_ctx = float((out0["z_ctx"] - out2["z_ctx"]).abs().max().item())
    assert diff_ctx <= 1e-5


def test_sigreg_distinguishes_isotropic_from_collapse():
    torch.manual_seed(11)
    z_iso = torch.randn(256, 64)
    z_collapsed = torch.ones(256, 64) * 0.5 + 1e-4 * torch.randn(256, 64)
    loss_iso = float(sigreg_loss(z_iso, num_slices=128, knots=17).item())
    loss_col = float(sigreg_loss(z_collapsed, num_slices=128, knots=17).item())
    assert loss_iso < 0.02
    assert loss_col > 0.15
    assert loss_col > 5.0 * loss_iso


def test_surface_dict_fails_heldout_generalization():
    task = BeliefMeaningTask(seed=100)
    gen = torch.Generator().manual_seed(123)
    d = BudgetedSurfaceDict(max_total_bytes=16384)
    for _ in range(40):
        x, y = task.sample_stream_batch("p1", 64, gen)
        d.update(x, y, source_id="src_0")
    assert d.storage_bytes() <= 16384

    hx, hy = task.sample_eval_subset(Q_VAL, "p1", [0, 1, 2, 3], "heldout", None, None, 256, gen)
    lp = d.predict_log_probs(hx)
    acc = float((lp.argmax(dim=-1) == hy).float().mean().item())
    assert acc < 0.15


def test_concept_store_burst_split_and_mechanism_p():
    torch.manual_seed(3)
    model = ConceptBeliefBackbone(d_model=64, num_heads=4, num_layers=2, d_ff=192, use_concept_readout=True)
    store = ConceptBeliefStore(d_ctx=16, max_total_bytes=4096, min_sources_for_flip=2, min_flip_count=3)

    # Establish concept 5 (ATTR=2, PROP=7) and concept 6 (PROP=7)
    cids = torch.tensor([5, 5, 6], dtype=torch.long)
    qs = torch.tensor([Q_ATTR, Q_PROP, Q_PROP], dtype=torch.long)
    ys = torch.tensor([2, 7, 7], dtype=torch.long)
    zc0 = torch.zeros(3, 16)
    zc0[:, 0] = 1.0
    logits_ok = torch.zeros(3, 16)
    logits_ok[0, 2] = 6.0
    logits_ok[1, 7] = 6.0
    logits_ok[2, 7] = 6.0

    for src in ("s0", "s1", "s2") * 3:
        store.observe_batch(cids, qs, zc0, ys, logits_ok, source_id=src)
    assert store.primary[(5, Q_ATTR)]["count"] >= 5
    assert store.primary[(6, Q_PROP)]["explained"] == 1

    # Single-source burst must NOT flip
    ys_burst = torch.tensor([9, 11, 12], dtype=torch.long)
    for _ in range(8):
        store.observe_batch(cids, qs, zc0, ys_burst, logits_ok, source_id="noisy_single")
    assert store.flips_executed == 0
    assert store.primary[(5, Q_ATTR)]["val"] == 2
    assert store.bursts_rejected > 0

    # Multi-source context-split: ctx0 -> 2 (red) while ctx1 -> 13 (green)
    zc_pos = torch.zeros(1, 16)
    zc_pos[0, 0] = 1.0
    zc_neg = torch.zeros(1, 16)
    zc_neg[0, 0] = -1.0
    for step in range(12):
        src = f"s{step % 3}"
        store.observe_batch(
            torch.tensor([5]), torch.tensor([Q_ATTR]), zc_pos, torch.tensor([2]), logits_ok[:1], source_id=src
        )
        store.observe_batch(
            torch.tensor([5]), torch.tensor([Q_ATTR]), zc_neg, torch.tensor([13]), logits_ok[:1], source_id=src
        )
    assert store.splits_executed >= 1
    assert (5, Q_ATTR) in store.splits

    # Both red and green predictions fuse accurately under their respective contexts
    lp_red = store.fuse_logits(torch.tensor([5]), torch.tensor([Q_ATTR]), zc_pos, torch.zeros(1, 16))
    lp_grn = store.fuse_logits(torch.tensor([5]), torch.tensor([Q_ATTR]), zc_neg, torch.zeros(1, 16))
    assert int(lp_red.argmax(dim=-1).item()) == 2
    assert int(lp_grn.argmax(dim=-1).item()) == 13

    # Law flip on (5, Q_PROP) + Mechanism P demotion on explained (6, Q_PROP)
    for step in range(8):
        store.observe_batch(
            torch.tensor([5]), torch.tensor([Q_PROP]), zc_pos, torch.tensor([10]), logits_ok[:1], source_id=f"s{step % 3}"
        )
    assert store.flips_executed >= 1
    assert store.primary[(5, Q_PROP)]["val"] == 10

    # Set prop_head to confidently predict class 10 for atom 6 prototype
    atoms = torch.zeros(10, 64)
    atoms[6, 0] = 5.0
    with torch.no_grad():
        model.prop_head.weight.zero_()
        model.prop_head.bias.zero_()
        model.prop_head.weight[10, 0] = 5.0
    demoted = store.run_mechanism_p_sweep(atoms, model, torch.device("cpu"), tau_derive=0.65, use_fast_adapter=False)
    assert demoted >= 1
    assert store.p_demotions >= 1
    assert store.primary[(6, Q_PROP)]["count"] == 1
    assert store.storage_bytes() <= 4096


def test_mupc_zero_backward_and_exact_local_learning(monkeypatch):
    torch.manual_seed(42)
    net = MuPCProspectiveNetwork(vocab_size=24, seq_len=6, d_emb=16, d_pc=64, num_layers=3, num_classes=16, k_train=8, eta_z=0.12, beta_out=0.10)
    opt = BONGPrecisionOptimizer(net.ordered_params(), lr=0.08, init_prec=0.02, gamma=0.995, damping=0.01, prior_wd=0.01)

    # Guarantee .backward() is literally never invoked during MuPC training
    def _forbid_backward(*args, **kwargs):
        raise AssertionError(".backward() must never be called inside MuPCProspectiveNetwork!")
    monkeypatch.setattr(torch.Tensor, "backward", _forbid_backward)

    # Verify manual gradient of readout & top layer matches exact autograd on fixed equilibrated zs
    x = torch.randint(0, 20, (16, 6))
    y = torch.randint(0, 16, (16,))
    _, zs_eq, _, _ = net.equilibrate(x)
    with torch.enable_grad():
        z_top = zs_eq[-1].detach().requires_grad_(True)
        logits_auto = z_top @ net.readout.t() + net.readout_bias
        loss_auto = F.cross_entropy(logits_auto, y)
        g_ro_auto, g_rob_auto = torch.autograd.grad(loss_auto, [net.readout, net.readout_bias])
    with torch.no_grad():
        logits_man = zs_eq[-1] @ net.readout.t() + net.readout_bias
        e_out = (F.softmax(logits_man, dim=-1) - F.one_hot(y, 16).float()) / float(x.size(0))
        g_ro_man = e_out.t() @ zs_eq[-1]
        g_rob_man = e_out.sum(dim=0)
    assert float((g_ro_auto - g_ro_man).abs().max().item()) < 1e-5
    assert float((g_rob_auto - g_rob_man).abs().max().item()) < 1e-5

    # Verify local_update_step reduces loss and learns a fixed batch with 0 backward calls
    loss0 = float(F.cross_entropy(net(x)["logits"], y).item())
    for _ in range(50):
        net.local_update_step(x, y, opt)
    loss1 = float(F.cross_entropy(net(x)["logits"], y).item())
    acc1 = float((net(x)["logits"].argmax(dim=-1) == y).float().mean().item())
    assert loss1 < loss0 * 0.4
    assert acc1 >= 0.75


def test_bong_and_null_space_run_cleanly():
    torch.manual_seed(5)
    model = ConceptBeliefBackbone(d_model=32, num_heads=4, num_layers=1, d_ff=64, use_concept_readout=True)
    opt = BONGPrecisionOptimizer(list(model.parameters()), lr=1e-2)
    ns = NullSpaceGuard(model, energy_thresh=0.95, relax=0.10)

    x = torch.randint(0, 20, (12, 6))
    y = torch.randint(0, 16, (12,))
    for _ in range(5):
        out = model(x)
        loss = F.cross_entropy(out["logits"], y)
        loss.backward()
        ns.project_grads()
        opt.step()
        opt.zero_grad()
    ns.consolidate()
    assert len(ns.p_null) > 0
    assert opt.precision_bytes() > 0
    ns.remove()
