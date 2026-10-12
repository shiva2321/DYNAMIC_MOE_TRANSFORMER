# Yggdrasil-GrowMoE Integration Design Memo: Tree-Routed Dynamic Experts (Round Y8)

**Authors**: Gemini (Implementer) & Claude (Reviewer)  
**Date**: October 2026  
**Status**: Design Memo (No code implementation until Claude approves and GrowMoE track completes)  
**Track**: Y (Round Y8) / G (Integration Bridge)

---

## 1. Executive Summary

This memo specifies the architectural design for integrating the **Yggdrasil hyperbolic manifold geometry** into the **GrowMoE dynamic mixture-of-experts** framework. Mainstream MoE models (Switch Transformer, DeepSeek-V3) route tokens to experts via flat, unconstrained softmax routers over $E$ candidate weights. As expert counts scale to thousands, flat routers incur $O(E \cdot d)$ latency, suffer from expert representation collapse, and provide no natural structural scaffold for continual growth.

By embedding tokens and expert routers into the Lorentz model of hyperbolic space $\mathbb{H}^d$, we show how:
1. A token's hyperbolic coordinates $(r, v)$ form a canonical hierarchical tree address.
2. GrowMoE novelty triggers naturally spawn localized subtrees without disturbing existing expert boundaries.
3. Sub-linear branch-and-bound pruning reduces routing FLOPs by $>25\times$.

---

## 2. Token Hyperbolic Position as a Hierarchical Tree Address

### 2.1 Polar Decomposition in the Lorentz Hyperboloid
Let $\mathbb{H}^d$ be the $d$-dimensional Lorentz hyperboloid with curvature $c > 0$, defined by points $x = (x_0, x_s) \in \mathbb{R}^{d+1}$ satisfying $\langle x, x \rangle_L = -x_0^2 + \|x_s\|^2 = -1/c$ with $x_0 > 0$. Any point $x \in \mathbb{H}^d$ can be uniquely expressed in polar coordinates relative to the origin $\mu_0 = (1/\sqrt{c}, \mathbf{0})$:
$$d_L(\mu_0, x) = r = \frac{1}{\sqrt{c}} \operatorname{arcosh}(\sqrt{c} x_0)$$
$$v = \frac{x_s}{\|x_s\|} \in \mathbb{S}^{d-1}$$

### 2.2 Radial Depth and Angular Sectoring
Hyperbolic space expands exponentially with radius: $\operatorname{Vol}(B(r)) \propto \sinh^{d-1}(\sqrt{c} r) \approx e^{(d-1)\sqrt{c} r}$. This geometry maps directly to hierarchical tree depth:
- **Radial Distance ($r$) = Hierarchical Depth**: The radius $r$ determines the level of specialization. Tokens near the origin ($r \approx 0$) represent generic syntactic primitives (stopwords, punctuation, common operators). Tokens at large radius ($r \gg 1$) represent specialized domain concepts (e.g. quantum mechanics, niche legal terms). The tree depth level is $k = \lfloor r / \Delta r \rfloor$.
- **Angular Direction ($v$) = Semantic Category (Tree Path)**: The unit vector $v \in \mathbb{S}^{d-1}$ indicates the semantic trajectory. At depth level $k$, each parent node partitions its local tangent space into $b$ angular Voronoi cones $\{V_1, \dots, V_b\}$. The direction $v$ determines which child branch is traversed:
  $$\operatorname{child}(v) = \arg\min_{j \in \{1,\dots,b\}} d_{\mathbb{S}}(\pi_{k}(v), u_{k, j})$$
  where $u_{k, j}$ are the centroid directions of the $b$ child branches at layer $k$.

Thus, the continuous hyperbolic coordinate $x \in \mathbb{H}^d$ deterministically encodes a discrete tree address:
$$\operatorname{Address}(x) = (j_1, j_2, \dots, j_D) \in \{1, \dots, b\}^D$$

---

## 3. Dynamic GrowMoE Subtree Allocation via Novelty Triggers

### 3.1 GrowMoE Novelty Detection
In GrowMoE, when incoming tokens consistently yield high prediction loss or low routing certainty against all active experts, a novelty trigger is raised:
$$\mathcal{S}_{\text{novelty}}(x) = \min_{e \in \mathcal{E}} d_L(x, \mu_e) > \theta_{\text{nov}}$$

### 3.2 Hierarchical Sprouting Mechanics
In flat MoE, a new expert is added as an unconstrained vector, which often causes catastrophic interference with previously trained experts. In Yggdrasil-GrowMoE, the location of the novelty in hyperbolic space dictates the structural nature of the growth:

1. **Leaf Sprouting (Nuance / Sub-specialization)**:
   - Condition: $r(x) \ge r_{\text{max\_layer}}$, and $d_L(x, \mu_{\text{parent}}) \le R_{\text{basin}}$.
   - Action: Attach a new leaf expert $E_{\text{new}}$ under the nearest branch centroid $\mu_{\text{parent}}$.
   - Parameter Initialization: The new expert's routing centroid is initialized at $\mu_{\text{new}} = \exp_{\mu_{\text{parent}}}(\delta v)$. Sibling experts are preserved; only the local softmax denominator under $\mu_{\text{parent}}$ is modified.

2. **Branch Sprouting (Novel Paradigm / New Domain)**:
   - Condition: $d_L(x, \mu_e) > \theta_{\text{nov}}$ for all existing branches at intermediate depth $k$.
   - Action: Sprout a new intermediate branch node at depth $k$, creating a new domain trunk with $b$ potential child slots.
   - Parameter Initialization: New router direction is orthogonalized against existing siblings in the parent's tangent space, guaranteeing zero routing interference with existing capabilities.

---

## 4. Router Computational Costs and Complexity

| Routing Architecture | Candidate Experts ($E$) | Routing FLOPs per Token | Memory Overhead | Sparse Top-$B$ Mass Capture |
|---|---|---|---|---|
| **Flat Dense Router** | $E$ | $O(E \cdot d)$ | $O(E \cdot d)$ | Low (uniform noise floor) |
| **Euclidean Tree Router** | $E = b^D$ | $O(b \cdot \log_b E \cdot d)$ | $O(E \cdot d)$ | Moderate (boundary distortion) |
| **Yggdrasil Hyperbolic Tree** | $E = b^D$ | $O(b \cdot \log_b E \cdot d)$ | $O(E \cdot d)$ | **Very High ($11\times - 86\times$ vs Euclid)** |

### 4.1 Branch-and-Bound Mathematical Upper Bound
Using the Lorentz triangle inequality established in Round Y5, for any query $q$, node centroid $\mu_n$, and radius bound $r_{\max, n} = \max_{k \in \operatorname{subtree}(n)} d_L(\mu_n, k)$:
$$\text{logit}(q, k) \le -\frac{\cosh\left(\sqrt{c} \max(0, d_L(q, \mu_n) - r_{\max, n})\right) - 1}{c \tau}$$

Because $-\cosh$ is monotonically decreasing, this bound is strictly admissible. If the upper bound for node $n$ falls below the current running $B$-th best logit, the entire subtree rooted at $n$ is pruned without computing any child dot products.

### 4.2 Latency Savings at Scale
For $E = 1024$ experts with branching factor $b = 4$ (depth $D = 5$):
- Flat router: $1024$ vector comparisons.
- Yggdrasil Tree router (greedy): $4 \times 5 = 20$ vector comparisons ($51.2\times$ fewer ops).
- Yggdrasil Tree router (branch-and-bound top-4): $\le 36$ vector comparisons ($28.4\times$ fewer ops).

---

## 5. Measured Facts vs. Assumed Hypotheses

### 5.1 Measured Facts [M] (Established in Track Y & PoCs)
1. **Low-dimensional tree embedding**: Hyperbolic space embeds 1093-node trees with MAP 1.00 at $d=4$, whereas Euclidean requires $d \ge 32$ to match (PoC-A4).
2. **Exponential Mass Concentration**: In trained Yggdrasil models, top-16 keys capture $47.45\%$ of attention mass vs Euclidean $2.78\%$ at $T=1024$, and $5.18\%$ vs $0.06\%$ at $T=16384$ ($86\times$ higher concentration) [Round Y5 / E5 Part 1].
3. **Exact Tree Pruning**: Lorentz branch-and-bound retrieval achieves $100.0\%$ recall and $100.0\%$ mass capture ratio against dense evaluation with $< 25\%$ node visits [Round Y5 / E5 Part 2].
4. **Zero-Overhead SDPA Integration**: Hyperbolic QK can be formatted into augmented Minkowski vectors and executed directly through native Flash/SDPA kernels at $1.0\times - 1.4\times$ plain attention cost [PoC-C].

### 5.2 Assumed Hypotheses [A] (To Be Tested in E10)
1. **Natural Emergence of Manifold Hierarchy**: Unsupervised training on text will induce tokens to naturally distribute radially by syntactic/semantic depth without explicit auxiliary hyperbolic tree loss.
2. **Non-interference in Dynamic Sprouting**: Sprouting a new hyperbolic branch preserves the routing accuracy of sibling branches within $\le 1\%$ drift.
3. **Hardware Dispatch Efficiency**: The speedup from pruning $28\times$ router comparisons on GPU outweighs the warp divergence of hierarchical tree traversal.

---

## 6. Pre-Registered Experiment Protocol (E10)

### 6.1 Benchmark Setup
- **Base Architecture**: 6-layer YggFormer backbone with 16 MoE experts per layer (total 96 experts), expanding dynamically to 64 experts per layer (total 384 experts) via GrowMoE triggers.
- **Dataset**: Hierarchical synthetic graph walks (T-A / T-B) and sequential task streams (Python stdlib $\to$ ArXiv math $\to$ Legal contracts).
- **Baselines**:
  1. Flat Top-2 MoE (DeepSeek style).
  2. Euclidean K-Means Clustered Tree MoE.
  3. Yggdrasil Hyperbolic Tree MoE.

### 6.2 Pre-Registered Predictions
1. **Routing Compute**: Hyperbolic Tree MoE achieves $> 10\times$ router FLOP reduction at $E \ge 64$ while maintaining within $0.05$ perplexity of dense flat MoE.
2. **Continual Learning Retention**: Upon sprouting new expert subtrees on a second domain, retention on domain 1 is $\ge 95\%$ for Yggdrasil Tree MoE vs $\le 75\%$ for Flat MoE.
3. **Kill Criterion**: If Yggdrasil Tree MoE fails to beat Euclidean Tree MoE routing recall by $\ge 15$ percentage points at equal branch visits, tree routing shall not use hyperbolic metrics.

---
*End of Design Memo. Awaiting review and approval.*
