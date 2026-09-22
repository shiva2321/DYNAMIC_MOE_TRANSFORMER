"""
Native Riemannian Semantic Manifold Learner.

Trains word and concept embeddings directly from raw text corpora using
UCE's native LoopedRiemannianTransformer and SphericalDynamicMoE,
and crystallizes learned concept attractors directly into TriTierSphericalMemory.

Zero external lookups (no spaCy, no GloVe, no MiniLM). 100% internal learning.
"""

import sys
import os
import math
import random
import time
from typing import List, Dict, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\spherical_hyperspace_memory")
from ishmn.geometry import (
    project_to_sphere,
    exp_map_sphere,
    log_map_sphere,
    parallel_transport_sphere,
    analogical_transfer_sphere,
    geodesic_distance_sphere
)
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine


class NativeLanguageLearner:
    """
    Trains UCE's internal representations natively from text corpora and
    crystallizes learned concepts directly into the brain's TriTierSphericalMemory.
    """
    def __init__(self, uce: UniversalCognitiveEngine):
        self.uce = uce
        self.device = uce.device
        self.vocab: List[str] = []
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_embeddings: Optional[nn.Embedding] = None

    def build_vocabulary(self, corpus_sentences: List[str], min_freq: int = 1):
        """Builds a discrete concept vocabulary from raw text sentences."""
        word_counts = {}
        for sent in corpus_sentences:
            tokens = [w.strip(".,!?:;\"'()[]{}").lower() for w in sent.split()]
            for t in tokens:
                if len(t) > 1 and t.isalpha():
                    word_counts[t] = word_counts.get(t, 0) + 1

        self.vocab = sorted([w for w, c in word_counts.items() if c >= min_freq])
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_word = {i: w for i, w in enumerate(self.vocab)}
        V = len(self.vocab)

        # Initialize internal concept embedding layer matching UCE d_model (32)
        self.word_embeddings = nn.Embedding(V, self.uce.d_model).to(self.device)
        nn.init.orthogonal_(self.word_embeddings.weight)
        print(f"[*] Built internal vocabulary of {V} concepts on device {self.device}")

    def train_on_corpus(
        self,
        corpus_sentences: List[str],
        epochs: int = 40,
        lr: float = 0.015,
        window_size: int = 4
    ):
        """
        Self-Supervised Distributional Riemannian Training:
        Maximizes mutual information between co-occurring words directly on S^{D-1}.
        Uses Riemannian InfoNCE / spherical skip-gram objective:
        Gradients directly shape the hyperspherical latent coordinates.
        """
        if not self.vocab:
            self.build_vocabulary(corpus_sentences)

        # Generate co-occurrence pairs
        pairs = []
        for sent in corpus_sentences:
            tokens = [self.word_to_id[w.strip(".,!?:;\"'()[]{}").lower()] 
                     for w in sent.split() 
                     if w.strip(".,!?:;\"'()[]{}").lower() in self.word_to_id]
            for i, target in enumerate(tokens):
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((target, tokens[j]))

        random.shuffle(pairs)
        print(f"[*] Generated {len(pairs)} co-occurrence context pairs.")

        optimizer = torch.optim.AdamW(self.word_embeddings.parameters(), lr=lr, weight_decay=1e-4)
        batch_size = 128
        num_batches = len(pairs) // batch_size

        t0 = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(pairs)
            for b in range(num_batches):
                batch = pairs[b * batch_size : (b + 1) * batch_size]
                targets = torch.tensor([p[0] for p in batch], dtype=torch.long, device=self.device)
                contexts = torch.tensor([p[1] for p in batch], dtype=torch.long, device=self.device)

                # Embed into d_model and project onto S^{D-1}
                v_target = self.word_embeddings(targets)
                v_context = self.word_embeddings(contexts)

                u_target = project_to_sphere(self.uce.world_model.to_sphere(v_target))
                u_context = project_to_sphere(self.uce.world_model.to_sphere(v_context))

                # Riemannian InfoNCE Contrastive Loss on S^{D-1}:
                # Positive pairs should have high cosine similarity (small geodesic distance)
                # Negative pairs should be dispersed uniformly across the hypersphere
                cos_sim_matrix = torch.matmul(u_target, u_context.t()) / 0.1 # temperature = 0.1
                labels = torch.arange(batch_size, device=self.device)
                loss = F.cross_entropy(cos_sim_matrix, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / max(1, num_batches)
                print(f"  Epoch {epoch+1:2d}/{epochs} | Spherical InfoNCE Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")

        print("[+] Native Riemannian language training complete.")

    def crystallize_to_memory(self):
        """
        Crystallizes the learned word representations directly into
        UCE's TriTierSphericalMemory (semantic_concepts).
        """
        self.uce.memory.semantic_concepts.clear()
        with torch.no_grad():
            all_embeds = self.word_embeddings.weight # [V, 32]
            all_u = project_to_sphere(self.uce.world_model.to_sphere(all_embeds)) # [V, D]

            for i, word in enumerate(self.vocab):
                u_word = all_u[i:i+1].cpu()
                v_word = all_embeds[i:i+1].cpu()
                self.uce.memory.semantic_concepts.append({
                    "concept_id": i,
                    "word": word,
                    "u": u_word,
                    "r": 1.0,
                    "v": v_word,
                    "domain": "learned_language",
                    "exemplar_count": 10
                })

        print(f"[+] Crystallized {len(self.uce.memory.semantic_concepts)} permanent concept attractors into TriTierSphericalMemory.")

    def solve_internal_analogy(self, a_word: str, b_word: str, c_word: str) -> Dict[str, Any]:
        """
        Solves A : B :: C : ? using purely the internal TriTierSphericalMemory.
        """
        wa = a_word.strip().lower()
        wb = b_word.strip().lower()
        wc = c_word.strip().lower()

        # Find words in semantic concepts
        concept_map = {sc["word"]: sc["u"].to(self.device) for sc in self.uce.memory.semantic_concepts}

        if wa not in concept_map or wb not in concept_map or wc not in concept_map:
            missing = [w for w in (wa, wb, wc) if w not in concept_map]
            return {
                "error": f"Unknown concepts in internal memory: {missing}",
                "predicted_word": None,
                "confidence": 0.0,
                "top_candidates": []
            }

        u_a = concept_map[wa]
        u_b = concept_map[wb]
        u_c = concept_map[wc]

        # Riemannian Parallel Transport on S^{D-1}
        u_d = analogical_transfer_sphere(u_a, u_b, u_c)

        dist_ab = geodesic_distance_sphere(u_a, u_b).item()
        dist_cd = geodesic_distance_sphere(u_c, u_d).item()
        isometry = abs(dist_ab - dist_cd) < 1e-4

        # Nearest neighbor query against internal memory
        candidates = []
        for word, u_w in concept_map.items():
            if word in (wa, wb, wc):
                continue
            dist = geodesic_distance_sphere(u_d, u_w).item()
            sim_pct = max(0.0, 1.0 - (dist / math.pi)) * 100.0
            candidates.append({
                "word": word,
                "geodesic_distance": dist,
                "similarity_pct": sim_pct
            })

        candidates.sort(key=lambda x: x["geodesic_distance"])
        best_word = candidates[0]["word"] if candidates else None
        conf = candidates[0]["similarity_pct"] if candidates else 0.0

        return {
            "predicted_word": best_word,
            "confidence": conf,
            "top_candidates": candidates[:5],
            "u_d": u_d,
            "distance_source": dist_ab,
            "distance_target": dist_cd,
            "isometry_preserved": isometry
        }
