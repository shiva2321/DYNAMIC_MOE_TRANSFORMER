"""
Infinite Spherical Hyperspace Memory Module (ISHM).

Features:
- D-dimensional directional vectors on S^{D-1}
- Unbounded radial coordinates in [0, inf) representing infinite hierarchical shells
- von Mises-Fisher + Radial Gaussian attention mechanism for continuous addressing
- Differentiable Riemannian geodesic steering: controller actively navigates memory positions
- Gated erase/add payload updates
- Dynamic shell allocation for unbounded memory growth
- Consolidation & Dreaming: Shell migration, geodesic cluster compaction, and synaptic energy tracking
"""

import math
from typing import Any, Optional, Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.geometry import (
    project_to_sphere,
    project_to_tangent_space,
    exp_map_sphere,
    geodesic_distance_sphere,
    spherical_to_cartesian,
    cartesian_to_spherical,
    EPS
)
from ishmn.vsa_memory import VSAEpisodicMemory

class InfiniteSphericalMemory(nn.Module):
    def __init__(
        self,
        num_slots: int = 32,
        hyperspace_dim: int = 8,
        payload_dim: int = 32,
        initial_radius_shells: int = 4,
        max_slots: int = 128,
        device: torch.device = None
    ):
        super().__init__()
        self.num_slots = num_slots
        self.hyperspace_dim = hyperspace_dim
        self.payload_dim = payload_dim
        self.initial_radius_shells = initial_radius_shells
        self.max_slots = max_slots

        # Base template parameters for memory slots
        u_init = torch.randn(num_slots, hyperspace_dim)
        u_init = project_to_sphere(u_init)
        self.register_buffer("base_u", u_init)

        # Concentric shells: r = 1.0, 2.0, ...
        shells = torch.linspace(1.0, float(initial_radius_shells), steps=num_slots).unsqueeze(-1)
        self.register_buffer("base_r", shells)

        # Initial zero payload
        self.register_buffer("base_v", torch.zeros(num_slots, payload_dim))

        # Tri-Tier Memory Storage
        # Tier 2: Episodic Memory Store (lifelong experiential trajectories)
        self.episodic_memory: list[dict] = []
        self.max_episodic_capacity: int = 2000

        # Tier 2b: VSA Non-Parametric Buffer Memory (Surprise-Adaptive Hebbian)
        self.vsa_episodic_memory = VSAEpisodicMemory(
            hyperspace_dim=hyperspace_dim,
            payload_dim=payload_dim,
            capacity=max_slots * 4,
            device=device
        )

        # Tier 3: Semantic Concept Lattice (crystallized invariant attractors formed during sleep)
        self.semantic_concepts: list[dict] = []
        self.timestamp: int = 0

        # Vectorized Tensor Caches for GPU-accelerated retrieval (Phase 1 Optimization)
        self._ep_cache_dirty: bool = True
        self._ep_cache_u: Optional[torch.Tensor] = None
        self._ep_cache_v: Optional[torch.Tensor] = None
        self._sem_cache_dirty: bool = True
        self._sem_cache_u: Optional[torch.Tensor] = None
        self._sem_cache_v: Optional[torch.Tensor] = None

    def init_memory(self, batch_size: int, device: torch.device = None) -> dict[str, torch.Tensor]:
        """
        Initializes memory state:
        - 'u': [B, N, D] (directions on S^{D-1})
        - 'r': [B, N, 1] (radial distances >= 0)
        - 'v': [B, N, d_v] (stored features)
        - 'usage': [B, N] (access frequency)
        - 'age': [B, N] (time elapsed since creation)
        - 'energy': [B, N] (consolidation strength / retention energy)
        """
        if device is None:
            device = self.base_u.device

        u = self.base_u.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        r = self.base_r.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        v = self.base_v.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        usage = torch.zeros(batch_size, self.num_slots, device=device)
        age = torch.zeros(batch_size, self.num_slots, device=device)
        energy = torch.ones(batch_size, self.num_slots, device=device) * 0.5

        return {
            "u": u,
            "r": r,
            "v": v,
            "usage": usage,
            "age": age,
            "energy": energy
        }

    def compute_addressing_weights(
        self,
        u_mem: torch.Tensor,
        r_mem: torch.Tensor,
        u_query: torch.Tensor,
        r_query: torch.Tensor,
        kappa: torch.Tensor,
        beta_r: torch.Tensor
    ) -> torch.Tensor:
        """Computes continuous von Mises-Fisher + Radial Gaussian attention weights."""
        u_query = project_to_sphere(u_query).unsqueeze(1)  # [B, 1, D]
        r_query = r_query.unsqueeze(1)                     # [B, 1, 1]

        # Cosine alignment on S^{D-1}
        cos_sim = torch.sum(u_mem * u_query, dim=-1)      # [B, N]
        rad_dist_sq = (r_mem.squeeze(-1) - r_query.squeeze(-1)) ** 2  # [B, N]

        logits = kappa * cos_sim - beta_r * rad_dist_sq
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        weights = F.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=1.0 / u_mem.size(1))
        return weights

    def read(
        self,
        state: dict[str, torch.Tensor],
        u_query: torch.Tensor,
        r_query: torch.Tensor,
        kappa: torch.Tensor,
        beta_r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        u_mem = state["u"]
        r_mem = state["r"]
        v_mem = state["v"]
        usage = state["usage"]
        energy = state["energy"]
        age = state.get("age", torch.zeros_like(usage))

        weights = self.compute_addressing_weights(u_mem, r_mem, u_query, r_query, kappa, beta_r)
        read_payload = torch.sum(weights.unsqueeze(-1) * v_mem, dim=1)
        read_payload = torch.nan_to_num(read_payload)

        # Update access usage and boost energy for recalled slots
        new_usage = usage + weights.detach()
        new_energy = torch.clamp(energy + weights.detach() * 0.2, max=5.0)
        new_age = age + 1.0

        new_state = {
            "u": u_mem,
            "r": r_mem,
            "v": v_mem,
            "usage": new_usage,
            "energy": new_energy,
            "age": new_age
        }
        return read_payload, weights, new_state

    def write(
        self,
        state: dict[str, torch.Tensor],
        u_write: torch.Tensor,
        r_write: torch.Tensor,
        add_vector: torch.Tensor,
        erase_vector: torch.Tensor,
        write_gate: torch.Tensor,
        kappa_w: torch.Tensor,
        beta_rw: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        u_mem = state["u"]
        r_mem = state["r"]
        v_mem = state["v"]
        usage = state["usage"]
        energy = state["energy"]
        age = state.get("age", torch.zeros_like(usage))

        weights = self.compute_addressing_weights(u_mem, r_mem, u_write, r_write, kappa_w, beta_rw)
        w_gated = (weights * write_gate).unsqueeze(-1)

        erase_mat = 1.0 - w_gated * erase_vector.unsqueeze(1)
        add_mat = w_gated * add_vector.unsqueeze(1)

        new_v = v_mem * erase_mat + add_mat
        new_v = torch.nan_to_num(new_v)
        new_usage = usage + weights.detach()
        new_energy = torch.clamp(energy + (weights * write_gate).detach() * 0.4, max=5.0)

        new_state = {
            "u": u_mem,
            "r": r_mem,
            "v": new_v,
            "usage": new_usage,
            "energy": new_energy,
            "age": age + 1.0
        }
        return weights, new_state

    def steer_memory_geodesics(
        self,
        state: dict[str, torch.Tensor],
        tangent_velocity: torch.Tensor,
        radial_velocity: torch.Tensor,
        steer_weights: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        u_mem = state["u"]
        r_mem = state["r"]
        B, N, D = u_mem.shape

        vel_tangent_expanded = tangent_velocity.unsqueeze(1).expand(B, N, D)
        if steer_weights is not None:
            vel_tangent_expanded = vel_tangent_expanded * steer_weights.unsqueeze(-1)

        u_steered = exp_map_sphere(u_mem, vel_tangent_expanded, dim=-1)

        rad_delta = radial_velocity.unsqueeze(1).expand(B, N, 1)
        if steer_weights is not None:
            rad_delta = rad_delta * steer_weights.unsqueeze(-1)
        r_steered = F.softplus(r_mem + rad_delta)

        return {
            "u": u_steered,
            "r": r_steered,
            "v": state["v"],
            "usage": state["usage"],
            "energy": state["energy"],
            "age": state.get("age", torch.zeros_like(state["usage"]))
        }

    def spawn_infinite_shell(
        self,
        state: dict[str, torch.Tensor],
        u_new: torch.Tensor,
        v_new: torch.Tensor,
        shell_delta: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """Spawns a new memory slot into an outer concentric spherical hyperspace shell."""
        u_mem = state["u"]
        r_mem = state["r"]
        v_mem = state["v"]
        usage = state["usage"]
        energy = state["energy"]
        age = state.get("age", torch.zeros_like(usage))
        B, N, D = u_mem.shape

        u_new_norm = project_to_sphere(u_new).unsqueeze(1)
        max_r, _ = torch.max(r_mem, dim=1, keepdim=True)
        r_new = max_r + shell_delta
        v_new_expanded = v_new.unsqueeze(1)
        usage_new = torch.ones(B, 1, device=u_mem.device)
        energy_new = torch.ones(B, 1, device=u_mem.device) * 1.0
        age_new = torch.zeros(B, 1, device=u_mem.device)

        if N < self.max_slots:
            new_u = torch.cat([u_mem, u_new_norm], dim=1)
            new_r = torch.cat([r_mem, r_new], dim=1)
            new_v = torch.cat([v_mem, v_new_expanded], dim=1)
            new_usage = torch.cat([usage, usage_new], dim=1)
            new_energy = torch.cat([energy, energy_new], dim=1)
            new_age = torch.cat([age, age_new], dim=1)
        else:
            # Replace slot with lowest energy
            lru_idx = torch.argmin(energy, dim=1)
            new_u = u_mem.clone()
            new_r = r_mem.clone()
            new_v = v_mem.clone()
            new_usage = usage.clone()
            new_energy = energy.clone()
            new_age = age.clone()
            for b in range(B):
                idx = lru_idx[b]
                new_u[b, idx] = u_new_norm[b, 0]
                new_r[b, idx] = r_new[b, 0]
                new_v[b, idx] = v_new_expanded[b, 0]
                new_usage[b, idx] = 1.0
                new_energy[b, idx] = 1.0
                new_age[b, idx] = 0.0

        return {
            "u": new_u,
            "r": new_r,
            "v": new_v,
            "usage": new_usage,
            "energy": new_energy,
            "age": new_age
        }

    def consolidate_and_migrate_shells(
        self,
        state: dict[str, torch.Tensor],
        consolidation_rate: float = 0.1,
        core_radius: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """
        Dreaming Phase Consolidation:
        - High-energy (frequently verified) memories migrate inward towards core shells.
        - Low-energy / noisy memories decay.
        - Radial migration: r_i <- r_i - consolidation_rate * (r_i - core_radius) * sigmoid(energy_i)
        """
        r_mem = state["r"]
        energy = state["energy"]

        # Inward attraction towards core shells based on retention energy
        pull_force = torch.sigmoid(energy - 1.0).unsqueeze(-1)
        dr = -consolidation_rate * (r_mem - core_radius) * pull_force
        new_r = F.softplus(r_mem + dr)

        # Decay energy gradually over time
        new_energy = energy * 0.95

        return {
            "u": state["u"],
            "r": new_r,
            "v": state["v"],
            "usage": state["usage"],
            "energy": new_energy,
            "age": state.get("age", torch.zeros_like(new_energy))
        }

    def compact_geodesic_clusters(
        self,
        state: dict[str, torch.Tensor],
        angle_threshold: float = 0.15
    ) -> dict[str, torch.Tensor]:
        """
        Consolidates near-identical memory slots on S^{D-1} into single semantic clusters.
        Prevents redundant representations across long-term evolution.
        """
        u_mem = state["u"]
        v_mem = state["v"]
        B, N, D = u_mem.shape

        new_u = u_mem.clone()
        new_v = v_mem.clone()

        with torch.no_grad():
            for b in range(B):
                # Pairwise cosine similarity matrix
                cos_matrix = u_mem[b] @ u_mem[b].T  # [N, N]
                for i in range(N):
                    for j in range(i + 1, N):
                        if cos_matrix[i, j] > math.cos(angle_threshold):
                            # Merge slot j towards slot i along geodesic midpoint
                            mid_u = project_to_sphere(u_mem[b, i] + u_mem[b, j])
                            new_u[b, i] = mid_u
                            new_v[b, i] = 0.5 * (v_mem[b, i] + v_mem[b, j])

        return {
            "u": new_u,
            "r": state["r"],
            "v": new_v,
            "usage": state["usage"],
            "energy": state["energy"],
            "age": state.get("age", torch.zeros_like(state["usage"]))
        }

    def log_episode(
        self,
        u: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        surprise: float = 0.0,
        context: Any = None,
        domain: str = "general",
        metadata: Optional[dict] = None
    ) -> int:
        """
        Logs an experiential trace into lifelong Episodic Memory.
        """
        self.timestamp += 1
        episode = {
            "episode_id": len(self.episodic_memory),
            "timestamp": self.timestamp,
            "u": project_to_sphere(torch.nan_to_num(u.detach().cpu().view(1, -1))),
            "r": float(r.detach().cpu().item()) if isinstance(r, torch.Tensor) else float(r),
            "v": torch.nan_to_num(v.detach().cpu().view(1, -1)),
            "surprise": float(surprise),
            "context": context,
            "domain": domain,
            "metadata": metadata or {}
        }
        self.episodic_memory.append(episode)
        if len(self.episodic_memory) > self.max_episodic_capacity:
            self.episodic_memory.pop(0)
        self._ep_cache_dirty = True

        # Write into non-parametric VSA buffer with vigilance gating
        if hasattr(self, "vsa_episodic_memory"):
            try:
                self.vsa_episodic_memory.write(
                    u=episode["u"].to(self.vsa_episodic_memory.slot_u.device),
                    r=episode["r"],
                    v=episode["v"].to(self.vsa_episodic_memory.slot_v.device)
                )
            except Exception:
                pass

        return episode["episode_id"]

    def _ensure_ep_cache(self, device: torch.device):
        if self._ep_cache_dirty or self._ep_cache_u is None or self._ep_cache_u.device != device or (self._ep_cache_u is not None and self._ep_cache_u.size(0) != len(self.episodic_memory)):
            if not self.episodic_memory:
                self._ep_cache_u = None
                self._ep_cache_v = None
                self._ep_cache_dirty = False
                return
            u_list = [torch.nan_to_num(ep["u"]).to(device) for ep in self.episodic_memory]
            v_list = [torch.nan_to_num(ep["v"]).to(device) for ep in self.episodic_memory]
            self._ep_cache_u = project_to_sphere(torch.cat(u_list, dim=0))
            self._ep_cache_v = torch.cat(v_list, dim=0)
            self._ep_cache_dirty = False

    def _ensure_sem_cache(self, device: torch.device):
        if self._sem_cache_dirty or self._sem_cache_u is None or self._sem_cache_u.device != device or (self._sem_cache_u is not None and self._sem_cache_u.size(0) != len(self.semantic_concepts)):
            if not self.semantic_concepts:
                self._sem_cache_u = None
                self._sem_cache_v = None
                self._sem_cache_dirty = False
                return
            u_list = [torch.nan_to_num(sc["u"]).to(device) for sc in self.semantic_concepts]
            v_list = [torch.nan_to_num(sc["v"]).to(device) for sc in self.semantic_concepts]
            self._sem_cache_u = project_to_sphere(torch.cat(u_list, dim=0))
            self._sem_cache_v = torch.cat(v_list, dim=0)
            self._sem_cache_dirty = False

    def recall_resonant_episodes(
        self,
        u_query: torch.Tensor,
        top_k: int = 5,
        max_geodesic_distance: float = 1.8
    ) -> list[dict]:
        """
        Vectorized associative recall of past experiential episodes via geodesic proximity on S^{D-1}.
        """
        if not self.episodic_memory:
            return []

        device = u_query.device
        self._ensure_ep_cache(device)
        if self._ep_cache_u is None:
            return []

        u_q = project_to_sphere(u_query.view(1, -1).to(device))
        cos_sim = torch.mm(u_q, self._ep_cache_u.t()).squeeze(0).clamp(-1.0 + EPS, 1.0 - EPS)
        geo_dist = torch.acos(cos_sim)

        valid_mask = geo_dist <= max_geodesic_distance
        if not valid_mask.any():
            return []

        similarities = torch.clamp(1.0 - (geo_dist / math.pi), min=0.0)
        scores = torch.where(valid_mask, similarities, torch.tensor(-1.0, device=device))
        k_actual = min(top_k, int(valid_mask.sum().item()))
        top_sims, top_indices = torch.topk(scores, k=k_actual)

        recalled = []
        for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
            if sim < 0.0:
                continue
            ep = self.episodic_memory[idx]
            recalled.append({
                "episode_id": ep["episode_id"],
                "domain": ep["domain"],
                "geodesic_distance": float(geo_dist[idx].item()),
                "resonance_similarity": float(sim),
                "surprise": ep["surprise"],
                "u": ep["u"],
                "r": ep["r"],
                "v": ep["v"],
                "context": ep["context"],
                "metadata": ep["metadata"]
            })
        return recalled

    def crystallize_semantic_concepts(
        self,
        cluster_angle_threshold: float = 0.40,
        min_cluster_size: int = 2
    ) -> dict[str, Any]:
        """
        Sleep-Phase Semantic Crystallization:
        Replays lifelong episodic traces, clusters resonant trajectories on S^{D-1},
        and synthesizes permanent invariant attractor nodes in the Semantic Concept Lattice.
        """
        import numpy as np
        if len(self.episodic_memory) < min_cluster_size:
            return {"status": "insufficient_episodes", "num_crystallized": len(self.semantic_concepts)}

        u_all = torch.cat([ep["u"] for ep in self.episodic_memory], dim=0) # [M, D]
        v_all = torch.cat([ep["v"] for ep in self.episodic_memory], dim=0) # [M, d_v]
        r_all = [ep["r"] for ep in self.episodic_memory]
        domains = [ep["domain"] for ep in self.episodic_memory]
        M = u_all.size(0)

        cos_sim = torch.mm(u_all, u_all.t()).clamp(-1.0 + EPS, 1.0 - EPS)
        geo_dist = torch.acos(cos_sim)

        visited = [False] * M
        new_concepts = []

        for i in range(M):
            if visited[i]:
                continue
            cluster_indices = [i]
            for j in range(i + 1, M):
                if not visited[j] and geo_dist[i, j].item() < cluster_angle_threshold:
                    cluster_indices.append(j)

            if len(cluster_indices) >= min_cluster_size:
                for idx in cluster_indices:
                    visited[idx] = True

                cluster_u = u_all[cluster_indices]
                cluster_v = v_all[cluster_indices]
                cluster_r = [r_all[idx] for idx in cluster_indices]
                cluster_domains = [domains[idx] for idx in cluster_indices]

                prototype_u = project_to_sphere(torch.mean(cluster_u, dim=0, keepdim=True))
                prototype_v = torch.mean(cluster_v, dim=0, keepdim=True)
                prototype_r = float(np.mean(cluster_r))

                domain_counts = {}
                for d in cluster_domains:
                    domain_counts[d] = domain_counts.get(d, 0) + 1
                dominant_domain = max(domain_counts, key=domain_counts.get)

                concept_entry = {
                    "concept_id": len(self.semantic_concepts) + len(new_concepts),
                    "u": prototype_u,
                    "r": prototype_r,
                    "v": prototype_v,
                    "domain": dominant_domain,
                    "exemplar_count": len(cluster_indices)
                }
                new_concepts.append(concept_entry)

        merged_count = 0
        for nc in new_concepts:
            merged = False
            for sc in self.semantic_concepts:
                dist = geodesic_distance_sphere(nc["u"], sc["u"]).item()
                if dist < cluster_angle_threshold:
                    total_count = sc["exemplar_count"] + nc["exemplar_count"]
                    w_sc = sc["exemplar_count"] / total_count
                    w_nc = nc["exemplar_count"] / total_count
                    sc["u"] = project_to_sphere(w_sc * sc["u"] + w_nc * nc["u"])
                    sc["v"] = w_sc * sc["v"] + w_nc * nc["v"]
                    sc["r"] = w_sc * sc["r"] + w_nc * nc["r"]
                    sc["exemplar_count"] = total_count
                    merged = True
                    merged_count += 1
                    break
            if not merged:
                self.semantic_concepts.append(nc)

        self._sem_cache_dirty = True
        return {
            "status": "crystallized",
            "new_concepts_formed": len(new_concepts),
            "merged_with_existing": merged_count,
            "total_semantic_concepts": len(self.semantic_concepts)
        }

    def admit_concept(
        self,
        u_new: torch.Tensor,
        r_new: Union[float, torch.Tensor],
        v_new: torch.Tensor,
        metadata: Optional[dict] = None,
        tau_sep: float = 0.35
    ) -> dict[str, Any]:
        """
        Anti-Collapse Semantic Concept Admission Filter.
        Guarantees that newly formed concepts do not crowd into existing attractor basins.
        If geodesic distance to the nearest existing concept is < tau_sep, merges into the
        nearest concept via Fréchet barycentric update.
        Otherwise, admits the concept as an independent attractor on S^{D-1}.
        """
        u_cand = project_to_sphere(torch.nan_to_num(u_new.detach().view(1, -1)))
        v_cand = torch.nan_to_num(v_new.detach().view(1, -1))
        r_cand = float(r_new.detach().item()) if isinstance(r_new, torch.Tensor) else float(r_new)
        meta = metadata or {}

        if not self.semantic_concepts:
            concept_entry = {
                "concept_id": 0,
                "u": u_cand.cpu(),
                "r": r_cand,
                "v": v_cand.cpu(),
                "domain": meta.get("domain", "general"),
                "exemplar_count": 1,
                "metadata": meta
            }
            self.semantic_concepts.append(concept_entry)
            self._sem_cache_dirty = True
            return {"action": "admitted", "concept_id": 0, "geodesic_distance": float("inf")}

        device = u_new.device if isinstance(u_new, torch.Tensor) else torch.device("cpu")
        self._ensure_sem_cache(device)
        u_cand_dev = u_cand.to(device)

        # Vectorized geodesic distance to all existing concepts
        cos_sim = torch.mm(u_cand_dev, self._sem_cache_u.t()).squeeze(0).clamp(-1.0 + EPS, 1.0 - EPS)
        geo_dists = torch.acos(cos_sim)
        min_dist, min_idx = torch.min(geo_dists, dim=0)
        min_dist_val = min_dist.item()
        target_idx = min_idx.item()

        if min_dist_val < tau_sep:
            # Merge into existing nearest concept via Fréchet barycentric update
            sc = self.semantic_concepts[target_idx]
            n_old = sc.get("exemplar_count", 1)
            n_total = n_old + 1
            w_old = n_old / n_total
            w_new = 1.0 / n_total

            sc_u = sc["u"].to(device)
            sc_v = sc["v"].to(device)

            sc["u"] = project_to_sphere(w_old * sc_u + w_new * u_cand_dev).cpu()
            sc["v"] = (w_old * sc_v + w_new * v_cand.to(device)).cpu()
            sc["r"] = w_old * sc["r"] + w_new * r_cand
            sc["exemplar_count"] = n_total
            if meta:
                sc.setdefault("metadata", {}).update(meta)

            self._sem_cache_dirty = True
            return {"action": "merged", "concept_id": sc.get("concept_id", target_idx), "geodesic_distance": min_dist_val}
        else:
            concept_entry = {
                "concept_id": len(self.semantic_concepts),
                "u": u_cand.cpu(),
                "r": r_cand,
                "v": v_cand.cpu(),
                "domain": meta.get("domain", "general"),
                "exemplar_count": 1,
                "metadata": meta
            }
            self.semantic_concepts.append(concept_entry)
            self._sem_cache_dirty = True
            return {"action": "admitted", "concept_id": concept_entry["concept_id"], "geodesic_distance": min_dist_val}

    def query_semantic_lattice(
        self,
        u_query: torch.Tensor,
        top_k: int = 3,
        max_geodesic_distance: float = 1.5
    ) -> list[dict]:
        """
        Vectorized query of permanent invariant concept attractors in the Semantic Concept Lattice.
        """
        if not self.semantic_concepts:
            return []

        device = u_query.device
        self._ensure_sem_cache(device)
        if self._sem_cache_u is None:
            return []

        u_q = project_to_sphere(u_query.view(1, -1).to(device))
        cos_sim = torch.mm(u_q, self._sem_cache_u.t()).squeeze(0).clamp(-1.0 + EPS, 1.0 - EPS)
        geo_dist = torch.acos(cos_sim)

        valid_mask = geo_dist <= max_geodesic_distance
        if not valid_mask.any():
            return []

        similarities = torch.clamp(1.0 - (geo_dist / math.pi), min=0.0)
        scores = torch.where(valid_mask, similarities, torch.tensor(-1.0, device=device))
        k_actual = min(top_k, int(valid_mask.sum().item()))
        top_sims, top_indices = torch.topk(scores, k=k_actual)

        matched = []
        for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
            if sim < 0.0:
                continue
            sc = self.semantic_concepts[idx]
            matched.append({
                "concept_id": sc["concept_id"],
                "word": sc.get("word", ""),
                "domain": sc["domain"],
                "geodesic_distance": float(geo_dist[idx].item()),
                "semantic_similarity": float(sim),
                "u": sc["u"],
                "r": sc["r"],
                "v": sc["v"],
                "exemplar_count": sc["exemplar_count"]
            })
        return matched

    def get_concept_by_word(self, word: str) -> Optional[dict]:
        """Retrieves a concept by word string from the Semantic Concept Lattice."""
        clean = word.strip().lower()
        for sc in self.semantic_concepts:
            if sc.get("word", "").strip().lower() == clean:
                return sc
        return None

    def add_semantic_concept(
        self,
        word: str,
        u: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        r: float = 1.0,
        domain: str = "language",
        exemplar_count: int = 1
    ) -> dict:
        """
        Directly registers or updates an invariant concept attractor in the Semantic Concept Lattice.
        """
        clean = word.strip().lower()
        u_norm = project_to_sphere(u.detach().cpu().view(1, -1))
        v_tensor = v.detach().cpu().view(1, -1) if v is not None else torch.zeros(1, self.payload_dim)
        
        for sc in self.semantic_concepts:
            if sc.get("word", "").strip().lower() == clean:
                sc["u"] = u_norm
                sc["v"] = v_tensor
                sc["r"] = r
                sc["exemplar_count"] += exemplar_count
                self._sem_cache_dirty = True
                return sc
        
        new_sc = {
            "concept_id": len(self.semantic_concepts),
            "word": clean,
            "u": u_norm,
            "r": r,
            "v": v_tensor,
            "domain": domain,
            "exemplar_count": exemplar_count
        }
        self.semantic_concepts.append(new_sc)
        self._sem_cache_dirty = True
        return new_sc

    def tri_tier_recall(
        self,
        state: dict[str, torch.Tensor],
        u_query: torch.Tensor,
        r_query: torch.Tensor,
        kappa: torch.Tensor,
        beta_r: torch.Tensor,
        top_k_episodic: int = 3,
        top_k_semantic: int = 2
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        """
        Unified Tri-Tier Memory Recall:
        1. Reads active volatile Working Memory slots.
        2. Recalls resonant experiential traces from lifelong Episodic Memory.
        3. Recalls invariant conceptual attractors from the Semantic Concept Lattice.
        """
        wm_payload, wm_weights, new_state = self.read(state, u_query, r_query, kappa, beta_r)
        episodes = self.recall_resonant_episodes(u_query, top_k=top_k_episodic)
        semantics = self.query_semantic_lattice(u_query, top_k=top_k_semantic)

        device = u_query.device
        fused_payload = wm_payload.clone()

        if episodes:
            ep_vectors = torch.cat([torch.nan_to_num(ep["v"]).to(device) for ep in episodes], dim=0)
            ep_weights = torch.tensor([ep["resonance_similarity"] for ep in episodes], device=device).unsqueeze(-1)
            ep_weights = ep_weights / (ep_weights.sum() + EPS)
            ep_fused = torch.sum(ep_weights * ep_vectors, dim=0, keepdim=True)
            fused_payload = 0.7 * fused_payload + 0.3 * ep_fused

        if semantics:
            sem_vectors = torch.cat([torch.nan_to_num(sc["v"]).to(device) for sc in semantics], dim=0)
            sem_weights = torch.tensor([sc["semantic_similarity"] for sc in semantics], device=device).unsqueeze(-1)
            sem_weights = sem_weights / (sem_weights.sum() + EPS)
            sem_fused = torch.sum(sem_weights * sem_vectors, dim=0, keepdim=True)
            fused_payload = 0.8 * fused_payload + 0.2 * sem_fused

        # 4. Fast VSA Non-Parametric Buffer Recall
        if hasattr(self, "vsa_episodic_memory") and bool(self.vsa_episodic_memory.occupied.any()):
            vsa_echo, vsa_conf = self.vsa_episodic_memory(u_query, r_query)
            vsa_echo = torch.clamp(torch.nan_to_num(vsa_echo), min=-5.0, max=5.0)
            if float(vsa_conf.mean().item()) > 0.05:
                fused_payload = 0.85 * fused_payload + 0.15 * vsa_echo.to(device)

        recall_meta = {
            "wm_weights": wm_weights,
            "recalled_episodes": episodes,
            "recalled_semantics": semantics,
            "episodic_count": len(episodes),
            "semantic_count": len(semantics)
        }
        return fused_payload, recall_meta, new_state
