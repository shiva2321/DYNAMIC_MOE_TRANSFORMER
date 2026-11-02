import numpy as np
import torch
import torch.nn.functional as F

from fsmoe import system_lab as SL
from fsmoe import traces as TR
from growmoe.experts import MLPExpert
from unimoe.heads import LdaHead


# ------------------------------------------------------------------------------------------------------------------------------ traces
def test_world_samples_have_the_right_shapes_and_exact_noiseless_transitions():
    w = TR.World(0)
    for r in range(TR.TAUGHT + TR.WITHHELD):
        x, x2 = w.sample(r, 0, 50, 0.0)
        assert x.shape == (50, 2 * TR.M) and x2.shape == x.shape
        assert torch.allclose(x2[:, :TR.M], w.fns[r](x[:, :TR.M]), atol=1e-5)


def test_regimes_share_one_state_distribution_but_act_differently():
    w = TR.World(1)
    s = torch.randn(200, TR.M)
    outs = [w.fns[r](s) for r in range(TR.TAUGHT + TR.WITHHELD)]
    for a in range(len(outs)):
        for b in range(a + 1, len(outs)):
            assert float(((outs[a] - outs[b]) ** 2).mean()) > 0.05                 # every pair of regimes acts differently on the same states
    xs = [w.sample(r, 0, 2000, 0.1)[0][:, :TR.M] for r in (0, 4, 7)]
    assert max(float((xs[i].mean(0) - xs[0].mean(0)).abs().max()) for i in (1, 2)) < 0.2     # the input state distribution does not depend on the regime


def test_surfaces_move_the_nuisance_features_not_the_function():
    w = TR.World(2)
    x0, _ = w.sample(0, 0, 500, 0.0)
    x2, _ = w.sample(0, 2, 500, 0.0)
    assert float((x0[:, TR.M:].mean(0) - x2[:, TR.M:].mean(0)).norm()) > 3.0
    assert float((x0[:, :TR.M].mean(0) - x2[:, :TR.M].mean(0)).abs().max()) < 0.3


def test_grow_when_stuck_does_not_deepen_a_simple_map():
    w = TR.World(3)
    g = torch.Generator().manual_seed(1)
    s = torch.randn(1500, TR.M, generator=g)
    x = torch.cat([s, 0.3 * torch.randn(1500, TR.M, generator=g)], 1)
    x2 = torch.cat([w.fns[0](s), x[:, TR.M:]], 1)
    out = TR.grow_when_stuck(x, x2, 0, rounds=8, steps=100)
    assert out["depth"] == 1 and out["final_mse"] < 0.2 * out["target_var"] and out["params"] > 0
    assert len(out["hist"]) == 8


def test_fixed_net_reports_its_size():
    w = TR.World(4)
    x, x2 = w.sample(0, 0, 400, 0.05)
    r = TR.fixed_net(x, x2, 0, 16, 2, epochs=3)
    assert r["hidden"] == 16 and r["depth"] == 2 and r["params"] > 0 and r["final_mse"] == r["final_mse"]


# ------------------------------------------------------------------------------------------------------------------------------ diverse selection
def test_diverse_pick_spreads_over_clusters_where_plain_ranking_does_not():
    g = torch.Generator().manual_seed(0)
    centres = F.normalize(torch.randn(3, 16, generator=g), dim=1)
    x = F.normalize(centres.repeat_interleave(8, 0) + 0.05 * torch.randn(24, 16, generator=g), dim=1)
    cluster = np.arange(3).repeat(8)
    score = np.concatenate([np.linspace(10, 9, 8), np.full(8, 8.0), np.full(8, 7.0)])
    plain = np.argsort(-score)[:3]
    div = SL.diverse_pick(score, x, 3)
    assert set(cluster[plain]) == {0} and len(set(cluster[div])) >= 2
    assert len(set(div.tolist())) == 3 and int(div[0]) == 0                    # the most novel input is always first


def test_diverse_pick_with_few_candidates_returns_them_all():
    x = F.normalize(torch.randn(5, 8), dim=1)
    assert sorted(SL.diverse_pick(np.arange(5.0), x, 5).tolist()) == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------------------------------------------------------------------ team answers
class _Rec:
    def __init__(self, module, lda):
        self.module, self.meta = module, {"lda": lda}


class _Cfg:
    lda_blend = 0.5


class _Pool:
    cfg = _Cfg()

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        d, k = 16, 4
        centres = F.normalize(torch.randn(k, d, generator=g), dim=1)
        self.X = F.normalize(centres.repeat_interleave(40, 0) + 0.1 * torch.randn(k * 40, d, generator=g), dim=1)
        self.y = torch.arange(k).repeat_interleave(40)
        self.experts = {}
        for e, cls in enumerate(([0, 1], [2, 3])):
            m = torch.isin(self.y, torch.tensor(cls))
            lda = LdaHead(d, k, 0.1)
            lda.update(self.X[m], self.y[m])
            mod = MLPExpert(d, 16, 1, k, seed=e)
            mod.mark_seen(torch.tensor(cls))
            opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
            for _ in range(60):
                mod.train_step(self.X[m], self.y[m], opt)
            self.experts[e] = _Rec(mod.eval(), lda)


def test_team_predict_answers_with_one_expert_when_one_is_close_and_stays_a_distribution():
    pool = _Pool()
    probs, size = SL.team_predict(pool, pool.X, tau=0.1)
    assert probs.shape == (len(pool.X), 4) and torch.allclose(probs.sum(1), torch.ones(len(pool.X)), atol=1e-4)
    assert (probs.argmax(1) == pool.y).float().mean() > 0.9
    assert float(size.float().mean()) < 1.4 and int(size.min()) >= 1 and int(size.max()) <= 2


def test_team_predict_uses_more_experts_at_a_high_temperature():
    pool = _Pool()
    _, lo = SL.team_predict(pool, pool.X, tau=0.05)
    _, hi = SL.team_predict(pool, pool.X, tau=50.0)
    assert float(hi.float().mean()) >= float(lo.float().mean())
