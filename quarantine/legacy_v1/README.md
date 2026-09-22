# Quarantined Legacy v1 Precursor Files

The files in this directory are early Generation 1 prototypes developed during the initial proof-of-concept phase of the spherical hyperspace memory project.

They have been retired from the active `ishmn/` runtime and archived here to keep the core architecture clean, cohesive, and maintainable.

## Archived Components

1. **`evolution_engine.py`**:
   - Primitive early world model (`PredictiveWorldModel`) and basic curiosity engine (`IntrinsicCuriosityEngine`).
   - **Superseded by**: `ishmn/universal_world_model.py` (Predictive JEPA + Causal Inversion + Pearl Level 3) and `ishmn/universal_cognitive_engine.py`.

2. **`self_evolving_loop.py`**:
   - Early standalone script for running loop iterations using `evolution_engine.py` and `network.py`.
   - **Superseded by**: `ishmn/continuous_self_loop.py` and `ishmn/autonomous_self_improver.py`.

3. **`network.py`**:
   - Initial feedforward `SphericalHyperspaceControllerNet` used in early synthetic toy tasks.
   - **Superseded by**: `ishmn/spherical_attention.py` (`LoopedRiemannianTransformer`) and `ishmn/universal_cognitive_engine.py`.

4. **`native_language_learner.py`**:
   - Early toy vocabulary embedding trainer for small text strings without a language model.
   - **Superseded by**: `ishmn/developmental_learner.py` (full 4-stage Piagetian ontogeny with Vygotskian ZPD) and `ishmn/grounded_hybrid_engine.py`.
