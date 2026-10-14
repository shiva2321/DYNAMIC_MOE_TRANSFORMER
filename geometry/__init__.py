"""
The `geometry` package: Non-Euclidean and Unified Hybrid Manifolds for Neuro-Symbolic Cognitive AI.

Exports:
- Manifold: Abstract base protocol.
- PoincareBall: Hyperbolic space (H^D) for hierarchies and trees.
- LorentzModel: The same hyperbolic space as a hyperboloid (no boundary, attention-friendly); see geometry/attention.py, neuron.py.
- CliffordTorus: Flat torus (T^D) for algebra, binding, and zero-drift composition.
- Hypersphere: Spherical space (S^{D-1}) for directional clustering.
- YggdrasilManifold, YggdrasilState, YggdrasilAction: The Unified Hybrid Cognitive Manifold.
"""

from geometry.base import Manifold
from geometry.poincare import PoincareBall
from geometry.lorentz import LorentzModel
from geometry.torus import CliffordTorus
from geometry.sphere import Hypersphere
from geometry.hybrid import YggdrasilManifold, YggdrasilState, YggdrasilAction
from geometry.snapped_state import SnappedRotationState

__all__ = [
    "Manifold",
    "PoincareBall",
    "LorentzModel",
    "CliffordTorus",
    "Hypersphere",
    "YggdrasilManifold",
    "YggdrasilState",
    "YggdrasilAction",
    "SnappedRotationState",
]

