# Deep Research & Architectural Synthesis: Tokenizer-Free Stream Chunking for SS-DM-MoE

Repository: `D:\Dynamic_MOE_Transformer`  
Evaluation Script: `scripts/run_phase2_phase3_eval.py`  
Result Artifact: `results/phase2_phase3_eval.json`

---

## 1. Literature Survey: Tokenizer-Free Sequence Modeling

| Architecture | Core Mechanism | Boundary Determination | Local-to-Global Bridge | Decoding / Reconstruction |
| :--- | :--- | :--- | :--- | :--- |
| **Charformer + GBST** (Tay et al., Google) | Gradient-Based Subword Tokenization over raw bytes | Soft weighted mixture over candidate block sizes | 1D depthwise convolution + block scoring softmax + stride downsampling | Standard sequence head or byte readout |
| **MEGABYTE** (Yu et al., Meta) | Fixed-size multiscale byte patches (`P` bytes per patch) | Rigid fixed stride `P` | Lossless patch embedder (`P * d_byte -> d_global`) | Local Autoregressive Transformer within each patch conditioned on the Global patch vector |
| **Byte Latent Transformer (BLT)** (Pagnoni et al., Meta FAIR) | Entropy-driven dynamic byte patches + rolling hash n-gram embeddings | Next-byte entropy threshold or monotonic entropy increase | Lightweight Local Byte Encoder + cross-attention Patch Pooling (`PatchQ` attending to byte states) | Lightweight Local Byte Decoder using cross-attention back to latent patch representations |
| **H-Net / H-Net++** (Hwang, Wang, Gu et al.) | End-to-end differentiable Dynamic Chunking without an external entropy model | Cosine-Similarity Boundary Router `(1 - cos(q_t, k_{t-1})) / 2` + MoE-style Ratio Loss `L_ratio` | Selects/pools boundary-delimited chunks into coarse latent sequence | EMA Smoothing + Straight-Through Estimator (STE) Dechunker + residual skip back to byte resolution |

---

## 2. Connecting the Literature to `dmoe` (Solving Three Problems at Once)

1. **Eliminating the Token Vocabulary Parameter Sink**:
   - Replacing `Embedding(vocab_size, d_model)` and `Linear(d_model, vocab_size)` with `MultiScaleByteChunker` (`dmoe/chunker.py`) reduces total model parameters from `2123499` (`tokenizer_free=False`) to `105327` (`tokenizer_free=True`) while handling arbitrary UTF-8 text, code, and binary streams without out-of-vocabulary failures.
2. **Fixing Phase 2b-i's `JEPAModule` Context Blindness (I-JEPA / V-JEPA Pattern)**:
   - Pooling over unmasked context chunks (`[:start]` and `[end:]`) and combining with the target span's positional representation before `self.predictor` ensures non-zero gradients flow into unmasked context positions (`max_grad_real_tokens = 0.0308`, `grad_mask_token = 0.1958`).
   - Combined with `MultiScaleByteChunker` (`tokenizer_free=True`), which breaks mean-pooling permutation invariance, JEPA + task loss outperforms task-only loss across `20` / `20` seeds (`jepa_mean = 0.1308`, `jepa_std = 0.0505`, `jepa_median = 0.1165` vs `baseline_mean = 0.5502`, `baseline_std = 0.1266`, `baseline_median = 0.528`; seed 0: `seed0_jepa_probe_loss = 0.0875` vs `seed0_baseline_probe_loss = 0.8783`).
3. **Fixing Phase 2b-ii's Simulation Loop Norm Explosion (Pre-Norm Contraction)**:
   - Applying `_complex_unit_norm` before `self.manifold_corr` inside `_run_simulation_loop` bounds the per-step candidate state, achieving `20` / `20` passing seeds (`max_norm_min = 2.8478`, `max_norm_median = 4.152`, `max_norm_max = 6.1803`).
