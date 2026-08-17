"""
Rich Multi-Domain and Hybrid Compositional Corpus.
Provides training and evaluation sets across:
- 4 Primary Distinct Domains: Code, Math, Philosophy, JSON
- 1 Unseen Hybrid / Compositional Domain: Mathematical Code & Symbolic Cognitive Models
"""

import random
from typing import Tuple, List, Dict
import torch

class RichMultiDomainCorpus:
    """
    Synthesizes rich domain-specific corpora with distinct train/test splits and hybrid evaluation prompts.
    """
    def __init__(self, seq_len: int = 96, batch_size: int = 8):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = 256 # Byte-level tokens (0-255)

        # Domain 0: Algorithms & Systems Code
        self.code_train = [
            "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    priority_queue = [(0, start)]\n    while priority_queue:\n        curr_dist, curr_node = heapq.heappop(priority_queue)\n        if curr_dist > distances[curr_node]:\n            continue\n        for neighbor, weight in graph[curr_node].items():\n            distance = curr_dist + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(priority_queue, (distance, neighbor))\n    return distances\n",
            "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n",
            "async def pipeline_worker(queue: asyncio.Queue, results: list):\n    while True:\n        task = await queue.get()\n        if task is None:\n            break\n        processed = await process_stream_chunk(task)\n        results.append(processed)\n        queue.task_done()\n"
        ]

        # Domain 1: Mathematics & Proofs
        self.math_train = [
            "Definition (Vector Space): A vector space V over field F is a set closed under addition and scalar multiplication satisfying associativity, commutativity of addition, existence of zero element, additive inverse, and distributivity: a*(u + v) = a*u + a*v for all a in F, u,v in V.\n",
            "Spectral Theorem: Every Hermitian matrix A in C^(nxn) is unitarily diagonalizable. That is, there exists a unitary matrix U such that U^H * A * U = Lambda, where Lambda is a diagonal matrix containing the real eigenvalues lambda_1, ..., lambda_n of A. Moreover, eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.\n",
            "Fundamental Theorem of Calculus: Let f be continuous on [a, b]. If F is an antiderivative of f on [a, b], then Integral from a to b of f(x) dx = F(b) - F(a). Furthermore, d/dx (Integral from a to x of f(t) dt) = f(x).\n"
        ]

        # Domain 2: Philosophy & Cognitive Architecture
        self.philosophy_train = [
            "The hard problem of consciousness concerns how and why physical processes in the brain give rise to subjective phenomenal experience. Functional explanations account for discrimination, integration, and reportability, yet leave open the explanatory gap regarding qualitative qualia.\n",
            "Epistemological foundationalism posits that beliefs are justified either by being basic beliefs—grounded in direct perceptual or rational intuition—or by being derived inferentially from foundational basic beliefs through valid epistemic deduction.\n",
            "Distributed cognition treats cognitive processes as emergent phenomena not bounded by individual skulls, but scaffolded across tools, external symbolic artifacts, and interpersonal communication protocols.\n"
        ]

        # Domain 3: Cloud Telemetry & JSON Schema
        self.json_train = [
            '{"event_type": "moe_routing_trace", "cluster_id": "us-east-1a", "metrics": {"active_experts": 8, "tokens_routed": 1048576, "mean_latency_ms": 1.42}, "telemetry": {"gpu_memory_used_mb": 4096, "temperature_c": 62.5}, "tags": ["prod", "vsa_hyperspace"]}\n',
            '{"schema": "http://json-schema.org/draft-07/schema#", "title": "ExpertConfig", "type": "object", "properties": {"d_model": {"type": "integer", "default": 256}, "resonance_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}}}\n',
            '{"deployment": {"id": "dep_98412", "model": "HyperTransformer-1B", "nodes": [{"node_id": 0, "status": "READY", "experts": [0, 1, 2]}, {"node_id": 1, "status": "READY", "experts": [3, 4, 5]}]}}\n'
        ]

        # Domain 4 (Unseen Hybrid Compositional Domain for Stress Testing):
        # Math + Code + Logic in Superposition
        self.hybrid_test = [
            "def power_iteration(matrix_A, num_simulations: int = 100):\n    # Spectral theorem in code: Computes dominant eigenvalue and eigenvector\n    b_k = np.random.rand(matrix_A.shape[1])\n    for _ in range(num_simulations):\n        b_k1 = np.dot(matrix_A, b_k)\n        b_k = b_k1 / np.linalg.norm(b_k1)\n    eigenvalue = np.dot(b_k.T, np.dot(matrix_A, b_k)) / np.dot(b_k.T, b_k)\n    return eigenvalue, b_k\n",
            "class EpistemicBeliefNetwork:\n    # Functional representation of justified true belief\n    def __init__(self, propositions: dict):\n        self.foundational_axioms = set(propositions.get('axioms', []))\n    def is_justified(self, claim: str) -> bool:\n        return claim in self.foundational_axioms or self.deduce_from_axioms(claim)\n",
            '{"theorem": "SpectralDecomposition", "proof_verified": true, "code_implementation": "def decompose(A): return np.linalg.eigh(A)", "philosophical_interpretation": "Orthogonal decomposition of continuous transformations into invariant eigen-spaces"}\n'
        ]

        self.domain_map = {
            0: ("Code", self.code_train),
            1: ("Math", self.math_train),
            2: ("Philosophy", self.philosophy_train),
            3: ("JSON", self.json_train),
        }

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return bytes([min(255, max(0, t)) for t in tokens]).decode("utf-8", errors="replace")

    def get_batch(self, domain_id: int, is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if domain_id == 4: # Hybrid
            templates = self.hybrid_test
            domain_name = "Hybrid-MathCode"
        else:
            domain_name, templates = self.domain_map[domain_id % 4]

        batch_inputs, batch_targets = [], []
        for _ in range(self.batch_size):
            stream_text = ""
            while len(stream_text.encode("utf-8")) < (self.seq_len + 20):
                stream_text += random.choice(templates)
            raw = self.encode(stream_text)
            start = random.randint(0, max(0, len(raw) - self.seq_len - 1))
            chunk = raw[start : start + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                chunk = chunk + [32] * (self.seq_len + 1 - len(chunk))
            batch_inputs.append(chunk[:self.seq_len])
            batch_targets.append(chunk[1:self.seq_len + 1])

        return torch.tensor(batch_inputs, dtype=torch.long), torch.tensor(batch_targets, dtype=torch.long), domain_name
