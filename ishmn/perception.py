"""
Universal Perception Engine for Infinite Spherical Hyperspace.

Accepts diverse data modalities (continuous vectors, discrete sequences, text-like token streams)
and projects them onto canonical representations in Infinite Spherical Hyperspace:
- Directional coordinate u in S^{D-1} (semantic orientation)
- Radial coordinate r in [0, inf) (abstraction depth / novelty scale)
- Payload feature vector v in R^{d_v}
"""

import math
from typing import Any, Union, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.geometry import project_to_sphere
from ishmn.spherical_attention import RiemannianSphericalTransformerLayer

class UniversalPerceptor(nn.Module):
    def __init__(
        self,
        hyperspace_dim: int = 8,
        payload_dim: int = 32,
        embedding_dim: int = 64,
        vocab_size: int = 1000
    ):
        super().__init__()
        self.hyperspace_dim = hyperspace_dim
        self.payload_dim = payload_dim
        self.embedding_dim = embedding_dim

        # 1. Token / Discrete Sequence Encoder (Riemannian Spherical Attention with SO(2) Winding)
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.seq_encoder = RiemannianSphericalTransformerLayer(
            d_model=embedding_dim,
            num_heads=4,
            hyperspace_dim=hyperspace_dim,
            d_ff=embedding_dim * 2
        )

        # 2. Continuous Vector Projector with arbitrary-dimension resampler
        self.sensory_dim = 32
        self.continuous_proj = nn.Sequential(
            nn.Linear(self.sensory_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 3. Hyperspace Projections from unified embedding
        self.to_direction = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, hyperspace_dim)
        )

        self.to_radius = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.to_payload = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, payload_dim)
        )

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encodes discrete sequence tokens [B, T] -> [B, embedding_dim] using RSA."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(self.token_embed.weight.device)
        embeds = self.token_embed(token_ids)
        h_seq = self.seq_encoder(embeds) # [B, T, embedding_dim]
        return h_seq.mean(dim=1)

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        device: Optional[torch.device] = None
    ) -> nn.Embedding:
        """
        Dynamically resizes token_embed to support higher vocabulary sizes (e.g. 50,257)
        while preserving all existing embedding weights.
        """
        if new_num_tokens <= self.token_embed.num_embeddings:
            return self.token_embed
        old_embed = self.token_embed
        dev = device or old_embed.weight.device
        new_embed = nn.Embedding(new_num_tokens, self.embedding_dim, device=dev)
        nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_embed.weight[:old_embed.num_embeddings].copy_(old_embed.weight)
        self.token_embed = new_embed
        return self.token_embed

    @staticmethod
    def log_radial_encode(x: Union[float, torch.Tensor], scale: float = 20.0) -> Tuple[Any, Any]:
        """Maps continuous values into scale-equivariant (sign, log_mag / scale)."""
        if isinstance(x, torch.Tensor):
            sign = torch.sign(x)
            log_mag = torch.log1p(torch.abs(x)) / scale
            return sign, log_mag
        else:
            sign = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
            log_mag = math.log1p(abs(x)) / scale
            return sign, log_mag

    @staticmethod
    def log_radial_decode(sign: Union[float, torch.Tensor], log_mag_scaled: Union[float, torch.Tensor], scale: float = 20.0) -> Any:
        """Decodes scale-equivariant (sign, log_mag / scale) back into continuous real space."""
        if isinstance(sign, torch.Tensor):
            lm = torch.clamp(log_mag_scaled * scale, max=75.0)
            return sign * torch.expm1(lm)
        else:
            lm = min(max(log_mag_scaled * scale, 0.0), 75.0)
            return sign * math.expm1(lm)

    def encode_continuous(self, vectors: torch.Tensor, use_log_radial: bool = True) -> torch.Tensor:
        """
        Encodes arbitrary continuous feature vectors [B, D_in] -> [B, embedding_dim]
        with scale-equivariant signed log-radial compression (Path B).
        rho(x) = sign(x) * ln(1 + |x|)
        """
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.dim() == 2:
            vectors = vectors.unsqueeze(1)  # [B, 1, D_in]
        elif vectors.dim() == 3 and vectors.size(1) != 1:
            vectors = vectors.mean(dim=1, keepdim=True)

        # Scale-Equivariant Signed Log-Radial Mapping:
        # Keeps numbers from 10^-6 to 10^30 smoothly bounded without saturation
        if use_log_radial:
            vectors = torch.sign(vectors) * torch.log1p(torch.abs(vectors))


        # Multi-scale Fourier / Sinusoidal continuous embedding
        half = self.sensory_dim // 2
        resampled = F.interpolate(vectors, size=half, mode='linear', align_corners=False).squeeze(1)
        freqs = torch.linspace(1.0, 8.0, half, device=vectors.device)
        sin_comp = torch.sin(resampled * freqs)
        flat = torch.cat([resampled, sin_comp], dim=-1)
        return self.continuous_proj(flat)


    def encode_subword_phases(self, data: Any, n_min: int = 2, n_max: int = 4) -> torch.Tensor:
        """
        Encodes character n-grams into continuous circular harmonic phase trajectories.
        Maps multi-character text chunks to smooth coordinates on S^{D-1}.
        """
        device = next(self.parameters()).device
        if isinstance(data, str):
            chars = [ord(c) for c in data]
        elif isinstance(data, (list, tuple)):
            chars = []
            for item in data:
                if isinstance(item, str):
                    chars.extend([ord(c) for c in item])
                elif isinstance(item, (int, float)):
                    chars.append(int(item))
                else:
                    chars.extend([ord(c) for c in str(item)])
        elif isinstance(data, torch.Tensor):
            flat_items = data.flatten().cpu().tolist()
            chars = [int(c) for c in flat_items]
        else:
            chars = [ord(c) for c in str(data)]

        if len(chars) == 0:
            chars = [32]

        # Extract n-grams and compute circular phase trajectories
        phases = []
        for n in range(n_min, n_max + 1):
            if len(chars) >= n:
                for i in range(len(chars) - n + 1):
                    gram = chars[i:i+n]
                    angle = sum((gram[j] / 256.0) * (2.0 * math.pi / (j + 1)) for j in range(n))
                    phases.append(angle % (2.0 * math.pi))
            else:
                angle = sum((c / 256.0) * (2.0 * math.pi) for c in chars)
                phases.append(angle % (2.0 * math.pi))

        if len(phases) == 0:
            phases = [0.0]

        half = self.sensory_dim // 2
        p_tensor = torch.tensor(phases, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        p_resampled = F.interpolate(p_tensor, size=half, mode='linear', align_corners=False).squeeze(0).squeeze(0)

        cos_comp = torch.cos(p_resampled)
        sin_comp = torch.sin(p_resampled)
        flat = torch.cat([cos_comp, sin_comp], dim=-1).unsqueeze(0)
        return self.continuous_proj(flat)

    def forward(
        self,
        data: Any,
        modality: str = "continuous",
        use_log_radial: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects any modality into Infinite Spherical Hyperspace coordinates:
        Returns:
            u: [B, hyperspace_dim] in S^{D-1}
            r: [B, 1] in [0, inf)
            v: [B, payload_dim]
        """
        if modality == "tokens":
            h = self.encode_tokens(data)
        elif modality == "continuous":
            h = self.encode_continuous(data, use_log_radial=use_log_radial)
        elif modality in ("subwords", "text_str"):
            h = self.encode_subword_phases(data)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Spherical mapping
        u = project_to_sphere(self.to_direction(h))
        r = F.softplus(self.to_radius(h)) + 0.5  # Positive radial abstraction
        v = torch.tanh(self.to_payload(h))

        return u, r, v

