"""
Infinite Spherical Hyperspace Memory Network (ISHMN)
Universal Cognitive Engine (UCE)

A domain-agnostic General Intelligence Architecture operating natively on
continuous Riemannian manifolds (S^{D-1} x R+).
"""

import sys
import importlib.util
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Subpackages: Neurobiological Subsystems & Data Banks
# -----------------------------------------------------------------------------
from ishmn import neuro
from ishmn.neuro import (
    ModernHopfieldNetwork,
    CognitiveMapGraph,
    CriticalityController,
    ChunkedGatedDeltaRuleLayer,
    TwoCompartmentDendriticExpert,
    DentateGyrusPatternSeparator,
    ToroidalGridEncoder,
    PredictiveGlobalWorkspace,
    HyperspaceGlobalBus,
    HierarchicalRiemannianPredictiveCoding,
    RiemannianPredictiveCodingLayer,
    DendriticCompartment
)

from ishmn import data
from ishmn.data import SCIENTIFIC_INVARIANTS

# -----------------------------------------------------------------------------
# 2. Backwards-Compatibility Bridges (Set up early so internal imports resolve)
# -----------------------------------------------------------------------------
sys.modules["ishmn.dendritic_expert"] = neuro.dendritic_expert
sys.modules["ishmn.attractor_network"] = neuro.attractor_network
sys.modules["ishmn.cognitive_map"] = neuro.cognitive_map
sys.modules["ishmn.criticality"] = neuro.criticality
sys.modules["ishmn.delta_net"] = neuro.delta_net
sys.modules["ishmn.dentate_grid"] = neuro.dentate_grid
sys.modules["ishmn.global_workspace"] = neuro.global_workspace
sys.modules["ishmn.predictive_coding"] = neuro.predictive_coding
sys.modules["ishmn.scientific_knowledge_bank"] = data.scientific_knowledge_bank

# Dynamic loader for quarantined legacy v1 modules
_legacy_dir = Path(__file__).resolve().parent.parent / "quarantine" / "legacy_v1"
for _mod_name in ["network", "evolution_engine", "self_evolving_loop", "native_language_learner"]:
    _fpath = _legacy_dir / f"{_mod_name}.py"
    if _fpath.exists() and f"ishmn.{_mod_name}" not in sys.modules:
        try:
            _spec = importlib.util.spec_from_file_location(f"ishmn.{_mod_name}", _fpath)
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[f"ishmn.{_mod_name}"] = _mod
                _spec.loader.exec_module(_mod)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# 3. Foundational Geometry & Memory
# -----------------------------------------------------------------------------
from ishmn.geometry import (
    project_to_sphere,
    project_to_tangent_space,
    exp_map_sphere,
    log_map_sphere,
    geodesic_distance_sphere,
    spherical_to_cartesian,
    cartesian_to_spherical,
    parallel_transport_sphere,
    analogical_transfer_sphere,
    EPS
)
from ishmn.memory import InfiniteSphericalMemory
from ishmn.perception import UniversalPerceptor

# -----------------------------------------------------------------------------
# 4. Attention & Dynamic MoE
# -----------------------------------------------------------------------------
from ishmn.spherical_attention import (
    RiemannianSphericalAttention,
    RiemannianSphericalTransformerLayer,
    LoopedRiemannianTransformer,
    has_rust_attention,
    fast_riemannian_so2_winding,
    fast_tiled_riemannian_attention
)
sys.modules["ishmn.custom_attention"] = sys.modules["ishmn.spherical_attention"]

from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.dynamic_optimizer import DynamicWarmupAdamW
from ishmn.muon import Muon, HybridMuonAdamW, zeropower_via_newtonschulz5
from ishmn.native_cognitive_transformer import NativeSphericalBlock, NativeSphericalTransformer
from ishmn.data_loader import (
    SovereignTokenizer,
    CharacterTokenizer,
    HuggingFaceCurriculumStreamer,
    CURATED_FALLBACK_CORPUS
)

# -----------------------------------------------------------------------------
# 5. Core Cognitive Engine & Deliberation
# -----------------------------------------------------------------------------
from ishmn.universal_world_model import UniversalWorldModel
from ishmn.universal_cognitive_engine import (
    UniversalCognitiveEngine,
    ACTION_DELIBERATE,
    ACTION_MOTOR_STEP,
    ACTION_EMIT_COMMUNICATION,
    ACTION_INVOKE_SYMBOLIC_TOOL,
    ACTION_NAMES,
    safe_eval_expression
)
from ishmn.deliberative_reasoner import DeliberativeReasoner, ReasoningResult
from ishmn.continuous_self_loop import (
    ContinuousSelfLoop,
    ContinuousSelfLoop as ContinuousAutonomousCognitiveLoop,
    TriggeredEmission
)

# -----------------------------------------------------------------------------
# 6. Hybrid Grounding, Continuous Learning & Ontogeny
# -----------------------------------------------------------------------------
from ishmn.grounded_hybrid_engine import GroundedHybridEngine
from ishmn.autonomous_self_improver import AutonomousSelfImprover
from ishmn.slm_adapter import SLMManifoldAdapter
from ishmn.neurosymbolic_scratchpad import NeuroSymbolicScratchpadLogitsProcessor
from ishmn.sigreg import SIGReg, epps_pulley_statistic
from ishmn.developmental_learner import (
    DevelopmentalStage,
    ZoneOfProximalDevelopment,
    ActiveCausalPlaySimulator,
    OperationalReversibilityAudit,
    DevelopmentalManifoldLearner
)
from ishmn.collective_hyperspace import CollectiveHyperspaceBus

# Expose legacy controller net symbol if loaded
if "ishmn.network" in sys.modules and hasattr(sys.modules["ishmn.network"], "SphericalHyperspaceControllerNet"):
    SphericalHyperspaceControllerNet = sys.modules["ishmn.network"].SphericalHyperspaceControllerNet
else:
    SphericalHyperspaceControllerNet = None

__version__ = "2.3.0"

__all__ = [
    # Geometry
    "project_to_sphere",
    "project_to_tangent_space",
    "exp_map_sphere",
    "log_map_sphere",
    "geodesic_distance_sphere",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "parallel_transport_sphere",
    "analogical_transfer_sphere",
    "EPS",
    # Core
    "InfiniteSphericalMemory",
    "UniversalPerceptor",
    "RiemannianSphericalAttention",
    "RiemannianSphericalTransformerLayer",
    "LoopedRiemannianTransformer",
    "has_rust_attention",
    "fast_riemannian_so2_winding",
    "fast_tiled_riemannian_attention",
    "SphericalDynamicMoE",
    "DynamicWarmupAdamW",
    "Muon",
    "HybridMuonAdamW",
    "zeropower_via_newtonschulz5",
    "NativeSphericalBlock",
    "NativeSphericalTransformer",
    "SovereignTokenizer",
    "CharacterTokenizer",
    "HuggingFaceCurriculumStreamer",
    "CURATED_FALLBACK_CORPUS",
    "UniversalWorldModel",
    "UniversalCognitiveEngine",
    "DeliberativeReasoner",
    "ReasoningResult",
    "ContinuousAutonomousCognitiveLoop",
    "ContinuousSelfLoop",
    "TriggeredEmission",
    "GroundedHybridEngine",
    "AutonomousSelfImprover",
    "SLMManifoldAdapter",
    "NeuroSymbolicScratchpadLogitsProcessor",
    "SIGReg",
    "epps_pulley_statistic",
    "CollectiveHyperspaceBus",
    "ACTION_DELIBERATE",
    "ACTION_MOTOR_STEP",
    "ACTION_EMIT_COMMUNICATION",
    "ACTION_INVOKE_SYMBOLIC_TOOL",
    "ACTION_NAMES",
    "safe_eval_expression",
    # Ontogeny
    "DevelopmentalStage",
    "ZoneOfProximalDevelopment",
    "ActiveCausalPlaySimulator",
    "OperationalReversibilityAudit",
    "DevelopmentalManifoldLearner",
    # Neuro
    "neuro",
    "ModernHopfieldNetwork",
    "CognitiveMapGraph",
    "CriticalityController",
    "ChunkedGatedDeltaRuleLayer",
    "TwoCompartmentDendriticExpert",
    "DentateGyrusPatternSeparator",
    "ToroidalGridEncoder",
    "PredictiveGlobalWorkspace",
    "HyperspaceGlobalBus",
    "HierarchicalRiemannianPredictiveCoding",
    "RiemannianPredictiveCodingLayer",
    "DendriticCompartment",
    # Data
    "data",
    "SCIENTIFIC_INVARIANTS",
    # Legacy
    "SphericalHyperspaceControllerNet"
]
