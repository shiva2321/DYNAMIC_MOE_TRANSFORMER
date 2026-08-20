# Pre-Registered Empirical Protocol: 5-Seed 10-Domain Continual Learning Study
**Date of Pre-Registration**: August 20, 2026 (Pre-Execution)  
**Repository**: `universal_substrait` / `DYNAMIC_MOE_TRANSFORMER`  
**Purpose**: Pre-define explicit statistical thresholds, degrees-of-freedom calculations, and falsification criteria for expanding the 10-domain continual learning comparison from $N=2$ to $N=5$ seeds.

---

## 1. Study Scope & Constant Compute Parameters

* **Models Under Evaluation**:
  1. `hyperspace_budgeted_spawn`: Dynamic Neurogenesis ($\tau_{\text{spawn}} = 0.35$, max 2 spawns/phase/layer, max 32 experts) + Complex Phasor VSA in $\mathbb{C}^{2048}$ + 2-Compartment Dendritic SwiGLU + `DynamicWarmupAdamW`.
  2. `static_moe`: Matched-Compute 16-Expert Static Softmax MoE ($k=2$, fixed capacity) + AdamW.
* **Seeds Evaluated**:
  * Existing Seeds: `1337`, `42`
  * New Pre-Registered Seeds: `7`, `123`, `999`
  * Total Sample Size: $N_1 = 5$ (`hyperspace_budgeted_spawn`), $N_2 = 5$ (`static_moe`)
* **Current $N=2$ Baseline Anchor (Corrected Sample SD)**:
  * `github_code`: $98.02\% \pm 0.91\%$ vs $93.84\% \pm 0.36\%$ ($+4.18\%$, Welch $t = 6.06$, $df = 1.31$, $p = 0.0645$, strong trend).
  * `gutenberg_literature`: $22.88\% \pm 0.53\%$ vs $19.81\% \pm 0.54\%$ ($+3.08\%$, Welch $t = 5.75$, $df = 2.00$, $p = 0.0290$, statistically significant).
  * `mean_bwt`: $+0.3118 \pm 0.0242$ vs $+0.2995 \pm 0.0186\text{ nats}$ ($\Delta = +0.0123$, Welch $t = 0.57$, $df = 1.87$, $p = 0.6290$, null).
* **Strict Parameter Controls**:
  * Sequence Length: $256$ tokens
  * Micro-Batch Size: $2$, Gradient Accumulation: $6$, Physical Sequences per Step: $12$ ($3,072$ tokens/step)
  * Phase Budget: Exactly $360$ steps per domain ($1,105,920$ tokens/phase)
  * Total Budget: Exactly $3,600$ steps ($11,059,200$ tokens per run)
  * Replay Buffer: Fixed ring buffer ($256$ samples/domain) at $\alpha = 0.20$ replay ratio

---

## 2. Pre-Registered Hypotheses

### Primary Hypotheses (Specialist Domain Retention)
* **$H_{1,\text{code}}$ (Systems Code Retention)**: `hyperspace_budgeted_spawn` maintains higher held-out accuracy on `github_code` at Phase 10 than `static_moe`.
* **$H_{1,\text{lit}}$ (Classic Literature Retention)**: `hyperspace_budgeted_spawn` maintains higher held-out accuracy on `gutenberg_literature` at Phase 10 than `static_moe`.

### Null / Control Hypotheses
* **$H_{0,\text{aggregate}}$ (Aggregate $R_{\text{BWT}}$)**: The mean Backward Transfer across all 10 domains is expected to remain statistically indistinguishable between architectures ($p > 0.05$) due to noise dilution across short/repeated shards.
* **$H_{0,\text{noise}}$ (Shallow / Repetition-Confounded Domains)**: Domains with low data volume subject to looping (`fineweb_edu`, `openweb_math`, `pubmed_biomedical`, `arxiv_physics`) will show no systematic advantage for either architecture ($|t| < 2.0$, $p > 0.05$).

---

## 3. Strict Pre-Registered Criteria for "Replicated Finding"

A domain will be declared **Replicated and Statistically Supported** if and only if **all three** of the following conditions are met:

### Condition 1: Strict Directional Consistency Across All 5 Paired Seeds
The dynamic spawning model must outperform the matched static MoE on that domain for **every single individual seed**:
$$\text{Acc}_{\text{spawn}}(i) > \text{Acc}_{\text{static}}(i) \quad \forall i \in \{1337, 42, 7, 123, 999\}$$
*(Binomial sign test probability under null $p = (0.5)^5 = 0.03125$)*.

### Condition 2: Welch's Two-Sample $t$-test ($p < 0.01$) with Exact Satterthwaite $df$
The two-sample test will compute Welch's $t$ with the exact Welch–Satterthwaite degrees of freedom:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}$$
$$df_{\text{Welch}} = \frac{\left(\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}\right)^2}{\frac{(s_1^2/N_1)^2}{N_1 - 1} + \frac{(s_2^2/N_2)^2}{N_2 - 1}}$$
The resulting two-tailed $p$-value from Student's $t(df_{\text{Welch}})$ distribution must satisfy:
$$p_{\text{Welch}} < 0.01$$

### Condition 3: Effect Size Floor ($\Delta_{\text{mean}} \ge +2.0\%$)
The pooled mean retention delta $\Delta_{\text{mean}} = \bar{X}_{\text{spawn}} - \bar{X}_{\text{static}}$ must remain $\ge +2.00$ percentage points.

---

## 4. Pre-Registered Falsification & Reporting Rules

1. **Sign Inversion Condition**: If even a single seed shows $\text{Acc}_{\text{spawn}}(i) < \text{Acc}_{\text{static}}(i)$ on a target domain, the claim of universal specialist advantage for that domain is **falsified**, and must be reported as "Seed-Sensitive / Inconsistent".
2. **Statistical Attenuation Condition**: If the 5-seed pooled test yields $p \ge 0.05$, the effect must be reported as "Attenuated / Non-Significant at $N=5$", and the paper's claims walked back accordingly.
3. **Zero Cherry-Picking Rule**: All 5 seeds must be reported in full. No seed may be dropped or replaced. No hyperparameter (including $\tau_{\text{spawn}}$, learning rate, or batch size) may be modified during or after the 5-seed run.
