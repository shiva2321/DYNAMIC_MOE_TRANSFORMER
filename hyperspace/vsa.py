"""
Complex Phasor Vector Symbolic Architecture (Fourier Holographic Reduced Representations - FHRR).
Supports GPU-accelerated binding, unbinding, bundling, and Hermitian cosine similarity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexPhasorVSA:
    """
    Mathematical primitives for Complex Phasor Hyperdimensional Computing (FHRR).
    All hypervectors reside on the unit complex circle: z_j = e^{i * theta_j} where theta_j in [-pi, pi).
    """

    @staticmethod
    def random_hyperspace_vector(shape: tuple, device: torch.device = None, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
        """
        Generates random unit-magnitude complex hypervectors uniformly sampled over phase [-pi, pi).
        """
        angles = (torch.rand(shape, device=device) * 2.0 * math.pi) - math.pi
        return torch.complex(torch.cos(angles), torch.sin(angles)).to(dtype)

    @staticmethod
    def project_real_to_phasor(real_vector: torch.Tensor) -> torch.Tensor:
        """
        Projects a continuous real vector x in R^D to complex phasor space z = e^{i * tanh(x) * pi}.
        Smoothly differentiable via PyTorch autograd.
        """
        angles = torch.tanh(real_vector.float()) * math.pi
        return torch.complex(torch.cos(angles), torch.sin(angles))

    @staticmethod
    def normalize(complex_tensor: torch.Tensor) -> torch.Tensor:
        """
        Projects any complex tensor back to unit modulus |z| = 1 on the complex circle.
        """
        angles = torch.angle(complex_tensor)
        return torch.complex(torch.cos(angles), torch.sin(angles))

    @staticmethod
    def project_phasor_to_real(phasor_vector: torch.Tensor) -> torch.Tensor:
        """
        Extracts the continuous real angles (phases) in [-1, 1] normalized.
        """
        angles = torch.angle(phasor_vector)
        return angles / math.pi

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binding operation (Hadamard complex product / phase addition).
        a (x) b = e^{i * (theta_a + theta_b)}
        """
        prod = a * b
        angles = torch.angle(prod)
        return torch.complex(torch.cos(angles), torch.sin(angles))

    @staticmethod
    def unbind(compound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Exact inverse unbinding via complex conjugate:
        a = (a (x) b) (x) b^(-1) = compound * conj(key)
        """
        inv_key = torch.conj(key)
        prod = compound * inv_key
        angles = torch.angle(prod)
        return torch.complex(torch.cos(angles), torch.sin(angles))

    @staticmethod
    def bundle(vectors: torch.Tensor, dim: int = -2) -> torch.Tensor:
        """
        Bundling / Superposition (vector addition projected onto the complex manifold).
        Psi = Normalize(sum_k v_k)
        """
        summed = torch.sum(vectors, dim=dim)
        angles = torch.angle(summed)
        return torch.complex(torch.cos(angles), torch.sin(angles))

    @staticmethod
    def hermitian_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculates the real part of the normalized Hermitian dot product:
        Sim(a, b) = (1 / D) * Re(a^H * b) in [-1.0, 1.0].
        Orthogonal random vectors in high dimension yield Sim ~ 0.
        Identical vectors yield Sim = 1.0.
        """
        d = a.shape[-1]
        dot = torch.sum(torch.conj(a) * b, dim=-1).real
        return dot / d

    @staticmethod
    def batch_similarity_matrix(queries: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Computes similarity between N queries and M codebook vectors.
        queries: [N, D] (complex)
        codebook: [M, D] (complex)
        Returns: [N, M] real similarity matrix
        """
        d = queries.shape[-1]
        # queries @ conj(codebook.T)
        matrix = torch.matmul(queries, torch.conj(codebook).T).real
        return matrix / d
