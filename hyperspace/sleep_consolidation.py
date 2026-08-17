"""
Biological Sleep-Phase Consolidation & Hopfield Attractor Merging Engine.
Implements:
1. Attractor Coincidence Detection: Pairwise Hermitian similarity matrix S_ij = Sim(K_i, K_j) in C^D.
2. Hopfield Barycentric Weight Merging: Merges co-resonant experts using usage-weighted barycenters:
   W_merged = (u_i * W_i + u_j * W_j) / (u_i + u_j)
   K_merged = ComplexPhasorVSA.project_real_to_phasor(u_i * K_i + u_j * K_j)
3. Dead/Inactive Expert Pruning: Removes starved experts with lifetime usage u_i < tau_prune.
4. Memory Key & Optimizer Re-indexing.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hyperspace.vsa import ComplexPhasorVSA

class SleepConsolidationEngine:
    """
    Biological Sleep Consolidation Engine for Dynamic Hyper-MoE Layers.
    Periodically consolidates redundant expert representations and prunes inactive modules.
    """
    def __init__(
        self,
        merge_similarity_threshold: float = 0.80,
        min_usage_prune_threshold: int = 5,
        min_experts_to_keep: int = 2,
    ):
        self.merge_threshold = merge_similarity_threshold
        self.prune_threshold = min_usage_prune_threshold
        self.min_experts = min_experts_to_keep

    def consolidate_moe_layer(
        self,
        moe_layer,
        layer_idx: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a sleep consolidation cycle on a single DynamicHyperMoE layer.
        """
        num_exp = moe_layer.num_experts
        if num_exp <= self.min_experts:
            return {"layer_idx": layer_idx, "merged": 0, "pruned": 0, "final_experts": num_exp}

        usage_counts = moe_layer.expert_usage_counts[:num_exp].clone()
        device = moe_layer.memory.expert_keys.device
        keys = moe_layer.memory.expert_keys # [num_exp, d_hyper] (complex)

        # 1. Compute Pairwise Similarity Matrix in C^D
        sim_matrix = ComplexPhasorVSA.batch_similarity_matrix(keys, keys) # [N, N]
        
        merged_count = 0
        pruned_count = 0
        merged_indices = set()
        prune_indices = set()

        # Find candidates for merging
        for i in range(num_exp):
            if i in merged_indices or i in prune_indices:
                continue
            for j in range(i + 1, num_exp):
                if j in merged_indices or j in prune_indices:
                    continue
                similarity = sim_matrix[i, j].item()
                if similarity >= self.merge_threshold:
                    # Merge j into i using usage-weighted barycentric interpolation
                    u_i = max(1.0, float(usage_counts[i].item()))
                    u_j = max(1.0, float(usage_counts[j].item()))
                    total_u = u_i + u_j
                    w_i = u_i / total_u
                    w_j = u_j / total_u

                    if verbose:
                        print(f"  💤 [SLEEP MERGE L{layer_idx}] Merging Expert {j} into Expert {i} (Sim: {similarity:.3f}, Weights: {w_i:.2f}/{w_j:.2f})")

                    # Merge Two-Compartment Pyramidal Expert weights
                    exp_i = moe_layer.experts[i]
                    exp_j = moe_layer.experts[j]

                    for p_i, p_j in zip(exp_i.parameters(), exp_j.parameters()):
                        p_i.data.copy_(w_i * p_i.data + w_j * p_j.data)

                    # Merge Complex Memory Address Keys
                    key_i = keys[i]
                    key_j = keys[j]
                    merged_complex = w_i * key_i + w_j * key_j
                    # Re-project to unit complex phasor
                    merged_key = ComplexPhasorVSA.normalize(merged_complex.unsqueeze(0)).squeeze(0)
                    moe_layer.memory.expert_keys.data[i] = merged_key

                    # Update usage count
                    moe_layer.expert_usage_counts[i] += usage_counts[j]
                    merged_indices.add(j)
                    merged_count += 1

        # Find candidates for pruning (inactive / starved experts)
        for i in range(num_exp):
            if i not in merged_indices:
                if usage_counts[i].item() < self.prune_threshold and (num_exp - len(merged_indices) - len(prune_indices)) > self.min_experts:
                    if verbose:
                        print(f"  ✂️ [SLEEP PRUNE L{layer_idx}] Pruning inactive Expert {i} (Usage: {usage_counts[i].item()})")
                    prune_indices.add(i)
                    pruned_count += 1

        # Reconstruct layer experts and keys
        remove_indices = merged_indices.union(prune_indices)
        if remove_indices:
            surviving_indices = [idx for idx in range(num_exp) if idx not in remove_indices]
            
            # Rebuild nn.ModuleList (moe_layer.num_experts property updates automatically)
            new_experts = nn.ModuleList([moe_layer.experts[idx] for idx in surviving_indices])
            moe_layer.experts = new_experts
            
            # Rebuild Memory Keys (memory.num_experts property updates automatically)
            surviving_keys = moe_layer.memory.expert_keys[surviving_indices]
            moe_layer.memory.expert_keys = nn.Parameter(surviving_keys, requires_grad=False)

            # Rebuild Usage Counts
            surviving_usages = usage_counts[surviving_indices]
            moe_layer.expert_usage_counts.zero_()
            moe_layer.expert_usage_counts[:len(surviving_usages)] = surviving_usages

        return {
            "layer_idx": layer_idx,
            "merged": merged_count,
            "pruned": pruned_count,
            "final_experts": moe_layer.num_experts,
        }

    def consolidate_full_model(self, model, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Executes a sleep consolidation cycle across all Transformer layers in model.
        """
        model.eval()
        if verbose:
            print("\n" + "=" * 80)
            print("  [SLEEP CONSOLIDATION ENGINE] EXECUTING BIOLOGICAL MEMORY REORGANIZATION")
            print("=" * 80)

        results = []
        for l_idx, block in enumerate(model.blocks):
            res = self.consolidate_moe_layer(block.hyper_moe, layer_idx=l_idx, verbose=verbose)
            results.append(res)

        total_merged = sum(r["merged"] for r in results)
        total_pruned = sum(r["pruned"] for r in results)
        if verbose:
            print(f"\n[SLEEP CONSOLIDATION COMPLETE] Merged: {total_merged} | Pruned: {total_pruned} | Capacity Reclaimed")
            print("=" * 80 + "\n")
        return results
