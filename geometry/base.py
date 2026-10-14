"""
Base Manifold protocol and mathematical abstractions for non-Euclidean representation spaces.
"""

from abc import ABC, abstractmethod
import torch


class Manifold(ABC):
    """Abstract base class for Riemannian and product manifolds."""

    def __init__(self, dim: int):
        self.dim = int(dim)

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Projects an arbitrary ambient vector onto the manifold."""
        pass

    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the geodesic distance d(x, y) on the manifold."""
        pass

    @abstractmethod
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Riemannian exponential map: walks from x in tangent direction v."""
        pass

    @abstractmethod
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Riemannian logarithmic map: tangent vector at x pointing along geodesic to y."""
        pass
