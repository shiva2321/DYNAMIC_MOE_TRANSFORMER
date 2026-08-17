"""
Multi-Domain Streaming Dataset Generator.
Synthesizes structured language streams across multiple distinct semantic domains:
1. Domain A: Python Code & Algorithms
2. Domain B: Symbolic Mathematics & Arithmetic Proofs
3. Domain C: Natural Language & Philosophy
4. Domain D: Structured JSON / Configuration Data
"""

import random
from typing import Tuple, List, Generator
import torch

class MultiDomainStreamGenerator:
    """
    Generates multi-domain text streams with customizable domain switching schedules.
    Uses UTF-8 byte encoding (vocab_size = 256) for universal zero-dependency tokenization.
    """
    def __init__(self, seq_len: int = 64, batch_size: int = 8):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = 256 # Byte level 0-255

        # Domain Templates
        self.code_templates = [
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n",
            "class VectorSymbolicMemory:\n    def __init__(self, dim=2048):\n        self.dim = dim\n        self.keys = []\n    def bind(self, a, b):\n        return a * b\n    def bundle(self, items):\n        return sum(items)\n",
            "async def fetch_telemetry(device_id: str, timeout: float = 5.0) -> dict:\n    session = aiohttp.ClientSession()\n    response = await session.get(f'/api/v1/sensors/{device_id}')\n    return await response.json()\n",
            "def quicksort(items):\n    if len(items) <= 1:\n        return items\n    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n    middle = [x for x in items if x == pivot]\n    right = [x for x in items if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n"
        ]

        self.math_templates = [
            "Theorem: For any right triangle with legs a and b, a^2 + b^2 = c^2. Proof: Consider square of side (a+b). Total area = (a+b)^2 = a^2 + 2ab + b^2. Decomposing into 4 triangles of area ab/2 and central square c^2 yields 2ab + c^2. Equating yields a^2 + b^2 = c^2. Q.E.D.\n",
            "Evaluation: Integral of f(x) = 3*x^2 + 4*x - 5 from x=0 to x=3. Antiderivative F(x) = x^3 + 2*x^2 - 5*x. F(3) = 27 + 18 - 15 = 30. F(0) = 0. Therefore, Integral = 30.\n",
            "Matrix Properties: Given A in R^(NxN), det(A * B) = det(A) * det(B). If det(A) != 0, A is invertible with A^(-1) = adj(A) / det(A). Eigenvalues lambda satisfy det(A - lambda * I) = 0.\n",
            "Complex Algebra: z = r * e^(i * theta) = r * (cos(theta) + i * sin(theta)). Multiplication z1 * z2 = r1*r2 * e^(i*(theta1 + theta2)). Euler formula: e^(i*pi) + 1 = 0.\n"
        ]

        self.language_templates = [
            "The architecture of human thought is fundamentally compositional. We take discrete concepts, bind them into relational structures, and manipulate them within working memory. Language serves as the externalized coordination protocol for distributed human cognition across millennia.\n",
            "In modern philosophy of mind, the distinction between symbolic reasoning and connectionist learning is often seen as a false dichotomy. Distributed high-dimensional vector spaces can instantiate symbolic algebraic systems while maintaining continuous gradient-based learning dynamics.\n",
            "The city was bathed in the amber glow of the setting sun. Long shadows stretched across the ancient cobblestone alleys as the evening breeze whispered secrets through the olive trees.\n",
            "Exploration is the engine of discovery. When an agent encounters an unfamiliar landscape, novelty detection triggers the formation of new conceptual models rather than overwriting established knowledge.\n"
        ]

        self.json_templates = [
            '{"system": "hyper_space", "version": "2.0", "status": "active", "telemetry": {"nodes": 128, "resonators": 16, "superposition_ratio": 0.94}, "config": {"d_hyper": 4096, "temperature": 0.7}}\n',
            '{"expert_registry": [{"id": 0, "domain": "syntax_code", "usage": 4502}, {"id": 1, "domain": "math_proof", "usage": 3210}], "metrics": {"loss": 0.142, "orthogonality": 0.003}}\n',
            '{"event": "expert_spawned", "timestamp": "2026-08-14T12:00:00Z", "payload": {"parent_cluster": "reasoning", "seed_dim": 2048, "resonance_score": 0.28}}\n'
        ]

        self.domain_map = {
            0: ("Code", self.code_templates),
            1: ("Math", self.math_templates),
            2: ("Language", self.language_templates),
            3: ("JSON", self.json_templates)
        }

    def encode_text(self, text: str) -> List[int]:
        """Encodes string to byte token integers (0-255)."""
        return list(text.encode("utf-8"))

    def decode_tokens(self, tokens: List[int]) -> str:
        """Decodes byte token integers back to string."""
        return bytes([min(255, max(0, t)) for t in tokens]).decode("utf-8", errors="replace")

    def get_domain_batch(self, domain_id: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Generates a batch of (input_ids, targets) for a specific domain.
        Returns:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len]
            domain_name: str
        """
        domain_name, templates = self.domain_map[domain_id % len(self.domain_map)]
        
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.batch_size):
            # Pick and concatenate random templates to form a long stream
            stream_text = ""
            while len(stream_text.encode("utf-8")) < (self.seq_len + 10):
                stream_text += random.choice(templates)
                
            raw_bytes = self.encode_text(stream_text)
            # Take a random slice
            max_start = len(raw_bytes) - self.seq_len - 1
            start_idx = random.randint(0, max(0, max_start))
            
            chunk = raw_bytes[start_idx : start_idx + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                chunk = chunk + [32] * (self.seq_len + 1 - len(chunk))
                
            inp = chunk[:self.seq_len]
            tgt = chunk[1:self.seq_len + 1]
            
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            
        return (
            torch.tensor(batch_inputs, dtype=torch.long),
            torch.tensor(batch_targets, dtype=torch.long),
            domain_name
        )
