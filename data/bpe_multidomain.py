"""
Comprehensive Multi-Domain BPE Dataset using tiktoken (GPT-2 Tokenizer).
Provides extensive, structured corpora across 5 distinct domains:
1. Python Algorithms & Systems Code
2. Mathematics & Quantum Physics
3. Philosophy of Mind & Epistemology
4. Cloud Architecture & JSON Telemetry
5. Narrative Literature & Science Fiction
"""

import os
import random
from typing import Tuple, List, Dict
import torch
import tiktoken

class BPEMultiDomainCorpus:
    """
    Sub-word tokenized multi-domain dataset with train/test splits for full-length text generation.
    """
    def __init__(self, seq_len: int = 128, batch_size: int = 8):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50304 # Padded to multiple of 64 for optimal GPU tensor alignment

        # -------------------------------------------------------------
        # DOMAIN 0: Python Algorithms & Systems Code
        # -------------------------------------------------------------
        self.code_data = """
import asyncio
import heapq
from typing import List, Dict, Optional, Tuple

class MinHeapPriorityQueue:
    def __init__(self):
        self._heap = []
        self._index = 0

    def push(self, item, priority: float):
        heapq.heappush(self._heap, (priority, self._index, item))
        self._index += 1

    def pop(self):
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        return heapq.heappop(self._heap)[-1]

    def is_empty(self) -> bool:
        return len(self._heap) == 0

def dijkstra_shortest_path(graph: Dict[str, Dict[str, float]], start: str) -> Dict[str, float]:
    distances = {node: float('inf') for node in graph}
    distances[start] = 0.0
    pq = MinHeapPriorityQueue()
    pq.push(start, 0.0)

    while not pq.is_empty():
        curr_node = pq.pop()
        curr_dist = distances[curr_node]

        for neighbor, weight in graph[curr_node].items():
            new_dist = curr_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                pq.push(neighbor, new_dist)

    return distances

def binary_search_rotated(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

async def async_worker_pool(tasks: List[str], max_concurrency: int = 4) -> List[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)
    async def process_task(task_id: str):
        async with semaphore:
            await asyncio.sleep(0.01)
            return {"task_id": task_id, "status": "COMPLETED", "processed_bytes": 1024}
    results = await asyncio.gather(*(process_task(t) for t in tasks))
    return list(results)
"""

        # -------------------------------------------------------------
        # DOMAIN 1: Mathematics & Quantum Physics
        # -------------------------------------------------------------
        self.math_data = """
Spectral Theorem for Hermitian Operators:
Let H be a complex Hilbert space and let A: H -> H be a bounded self-adjoint linear operator.
There exists a unique spectral measure E on the Borel subsets of the spectrum sigma(A) such that A = Integral_sigma(A) lambda dE(lambda).
If H is finite-dimensional, this implies there exists an orthonormal basis consisting entirely of eigenvectors of A with corresponding real eigenvalues lambda_1 <= lambda_2 <= ... <= lambda_n.

Quantum Superposition and Wavefunction Collapse:
In non-relativistic quantum mechanics, a physical state is represented by a unit vector |psi> in a complex Hilbert space H.
The state can be expressed as a linear combination of orthonormal basis eigenstates |phi_k>:
|psi> = Sum_k c_k |phi_k|, where c_k in C and Sum_k |c_k|^2 = 1.
According to the Born rule, the probability of measuring eigenvalue a_k associated with operator A is given by P(a_k) = |<phi_k|psi>|^2 = |c_k|^2.
Upon measurement, the system undergoes discontinuous projection into the corresponding eigensubspace.

Fundamental Theorem of Calculus and Differential Forms:
Stokes Theorem states that for any compact, oriented k-dimensional smooth manifold M with boundary dM, and any differential (k-1)-form omega:
Integral_M d(omega) = Integral_dM omega.
This fundamental equation unifies the fundamental theorem of calculus, Green's theorem, the divergence theorem, and classical Stokes theorem under a single geometric formulation.
"""

        # -------------------------------------------------------------
        # DOMAIN 2: Philosophy of Mind & Cognitive Architecture
        # -------------------------------------------------------------
        self.philosophy_data = """
The Hard Problem of Consciousness and Phenomenal Qualia:
The distinction between the functional mechanisms of cognitive processing and subjective qualitative experience represents the central debate in modern philosophy of mind.
While easy problems encompass discrimination, categorisation, attentional focus, and deliberate control of behavior, the hard problem asks why these neurobiological physical processes are accompanied by an inner experiential life.
Functionalist theories argue that mental states are defined solely by their causal roles and relations to sensory inputs and behavioral outputs.
However, thought experiments such as the philosophical zombie and the knowledge argument suggest that complete physical descriptions leave an explanatory gap regarding phenomenal qualia.

Predictive Processing and the Free Energy Principle:
Cognitive systems can be mathematically formalized as hierarchical prediction machines engaged in minimizing variational free energy.
Under this paradigm, the brain does not passively process ascending sensory streams; rather, it continuously generates top-down generative models of the world.
Lower sensory areas compute prediction errors representing the divergence between sensory observations and descending hypotheses.
Through active inference, rational agents update their internal probabilistic representations to maintain structural homeostasis in dynamic environments.
"""

        # -------------------------------------------------------------
        # DOMAIN 3: Cloud Architecture & JSON Telemetry
        # -------------------------------------------------------------
        self.json_data = """
{
  "system_telemetry": {
    "cluster_name": "us-west-prod-hyper-01",
    "architecture": "neuro_symbolic_hyperspace",
    "status": "HEALTHY",
    "metrics": {
      "active_routing_nodes": 64,
      "branching_ratio_sigma": 1.0002,
      "mean_latency_microseconds": 84.5,
      "p99_latency_microseconds": 142.1,
      "memory_utilization_ratio": 0.42
    },
    "registered_experts": [
      {
        "expert_id": 0,
        "specialization": "python_algorithmic_syntax",
        "tokens_processed": 4819204,
        "dendritic_gain_alpha": 0.142
      },
      {
        "expert_id": 1,
        "specialization": "spectral_mathematics",
        "tokens_processed": 3910240,
        "dendritic_gain_alpha": 0.189
      },
      {
        "expert_id": 2,
        "specialization": "epistemic_philosophy",
        "tokens_processed": 5120930,
        "dendritic_gain_alpha": 0.165
      }
    ],
    "global_workspace_bus": {
      "resonance_threshold": 0.28,
      "hopfield_clean_up_iterations": 2,
      "quantum_grover_amplification": 1.5
    }
  }
}
"""

        # -------------------------------------------------------------
        # DOMAIN 4: Narrative Literature & Science Fiction
        # -------------------------------------------------------------
        self.narrative_data = """
The starship Prometheus drifted in the silent vacuum beyond the rings of Saturn.
Inside the central nexus, the quantum navigation core pulsed with a deep, rhythmic sapphire glow, calculating hyper-dimensional trajectories across warped spacetime manifolds.
Commander Elena Vance watched the distant constellations through the crystalline observation deck, listening to the gentle hum of the environmental recyclers.
For three centuries, the colony vessels had traveled in cryo-sleep, guided only by the autonomous cognitive architecture of the ship's synthetic intelligence.
Now, as the long journey neared its destination, the automated sensors detected gravitational anomalies echoing from the surface of the uncharted terrestrial exoplanet.
Elena stepped forward to the control console, activating the communication array to transmit the first expedition signal across the light-years back to Earth.
"""

        self.domain_map = {
            0: ("Python Code", self.code_data),
            1: ("Mathematics", self.math_data),
            2: ("Philosophy", self.philosophy_data),
            3: ("Cloud JSON", self.json_data),
            4: ("Sci-Fi Narrative", self.narrative_data)
        }

        # Tokenize and cache each domain as 1D tensors
        self.domain_tokens = {}
        for d_id, (name, text) in self.domain_map.items():
            toks = self.enc.encode(text)
            self.domain_tokens[d_id] = torch.tensor(toks, dtype=torch.long)
            print(f"-> Domain {d_id} ({name:<16}): {len(toks):,} BPE tokens")

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    def get_batch(self, domain_id: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        d_name, _ = self.domain_map[domain_id % len(self.domain_map)]
        data = self.domain_tokens[domain_id % len(self.domain_map)]
        
        batch_in, batch_tgt = [], []
        for _ in range(self.batch_size):
            # If data is shorter than seq_len, repeat it
            if len(data) <= self.seq_len + 2:
                reps = (self.seq_len // len(data)) + 2
                cur_data = data.repeat(reps)
            else:
                cur_data = data
                
            start = random.randint(0, len(cur_data) - self.seq_len - 1)
            chunk = cur_data[start : start + self.seq_len + 1]
            batch_in.append(chunk[:self.seq_len])
            batch_tgt.append(chunk[1:self.seq_len + 1])

        return torch.stack(batch_in), torch.stack(batch_tgt), d_name
