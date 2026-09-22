"""
Neurobiological Brain Subsystems for Universal Cognitive Engine (UCE).

Integrates specialized computational neuroscience modules:
1. Attractor Network: Energy-based associative retrieval via Modern Hopfield Networks.
2. Cognitive Map: Successor representation topological planning graph.
3. Criticality: Self-organized criticality and branching ratio homeostatic controller.
4. Delta Net: Chunked gated delta-rule linear recurrent memory layer.
5. Dendritic Expert: Two-compartment somatic/apical expert networks.
6. Dentate Grid: Dentate gyrus sparse pattern separator and toroidal grid cell encoder.
7. Global Workspace: Predictive Global Workspace conscious bottleneck and global bus.
8. Predictive Coding: Hierarchical Riemannian predictive coding with dendritic compartments.
"""

from ishmn.neuro.attractor_network import ModernHopfieldNetwork
from ishmn.neuro.cognitive_map import CognitiveMapGraph
from ishmn.neuro.criticality import CriticalityController
from ishmn.neuro.delta_net import ChunkedGatedDeltaRuleLayer
from ishmn.neuro.dendritic_expert import TwoCompartmentDendriticExpert
from ishmn.neuro.dentate_grid import DentateGyrusPatternSeparator, ToroidalGridEncoder
from ishmn.neuro.global_workspace import (
    PredictiveGlobalWorkspace,
    HyperspaceGlobalBus,
    GWTTicket
)
from ishmn.neuro.predictive_coding import (
    HierarchicalRiemannianPredictiveCoding,
    RiemannianPredictiveCodingLayer,
    DendriticCompartment
)

__all__ = [
    "ModernHopfieldNetwork",
    "CognitiveMapGraph",
    "CriticalityController",
    "ChunkedGatedDeltaRuleLayer",
    "TwoCompartmentDendriticExpert",
    "DentateGyrusPatternSeparator",
    "ToroidalGridEncoder",
    "PredictiveGlobalWorkspace",
    "HyperspaceGlobalBus",
    "GWTTicket",
    "HierarchicalRiemannianPredictiveCoding",
    "RiemannianPredictiveCodingLayer",
    "DendriticCompartment",
]
