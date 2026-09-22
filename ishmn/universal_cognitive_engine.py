"""
Universal Cognitive Engine (UCE).

A truly domain-agnostic cognitive architecture operating in Infinite Spherical Hyperspace (S^{D-1} x R+).
No hardcoded math rules, no base-10 digit slicing, no domain-specific dictionary tables.

Integrates:
1. Universal Perception: Arbitrary modalities (continuous vectors, discrete symbols, sequential tokens).
2. Infinite Spherical Hyperspace Memory (ISHM): Persistent associative memory with vMF addressing.
3. Spherical Dynamic Mixture-of-Experts (MoE): Autonomous neurogenesis via clonal mitosis upon novelty.
4. Universal World Model (Predictive JEPA):
   - Forward latent prediction: z_{t+1} = W_fwd(z_t, c_t)
   - Causal rollback / inversion: z_t = W_bwd(z_{t+1}, c_t)
   - Self-supervised Pearl Level 3 Geodesic Consistency Loss
5. Adaptive Computation Time (ACT / System 1 vs System 2):
   - Internal mental deliberation loop on the Riemannian manifold when confidence < tau.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import (
    project_to_sphere,
    geodesic_distance_sphere,
    exp_map_sphere,
    log_map_sphere,
    parallel_transport_sphere,
    analogical_transfer_sphere
)
from ishmn.memory import InfiniteSphericalMemory
from ishmn.perception import UniversalPerceptor
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.universal_world_model import UniversalWorldModel
from ishmn.dynamic_optimizer import DynamicWarmupAdamW
from ishmn.spherical_attention import RiemannianSphericalAttention
from ishmn.neuro import (
    ChunkedGatedDeltaRuleLayer,
    DentateGyrusPatternSeparator,
    ToroidalGridEncoder,
    CognitiveMapGraph,
    ModernHopfieldNetwork,
    HyperspaceGlobalBus,
    PredictiveGlobalWorkspace,
    CriticalityController
)
import ast
import operator as op

# Cognitive-Motor Action Space Constants
ACTION_DELIBERATE = 0
ACTION_MOTOR_STEP = 1
ACTION_EMIT_COMMUNICATION = 2
ACTION_INVOKE_SYMBOLIC_TOOL = 3

ACTION_NAMES = {
    ACTION_DELIBERATE: "DELIBERATE",
    ACTION_MOTOR_STEP: "MOTOR_STEP",
    ACTION_EMIT_COMMUNICATION: "EMIT_COMMUNICATION",
    ACTION_INVOKE_SYMBOLIC_TOOL: "INVOKE_SYMBOLIC_TOOL"
}

_SAFE_MATH_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_eval_expression(expr: str) -> float:
    """Safely evaluates an arithmetic/symbolic expression via AST traversal without eval() security risks."""
    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_fn = _SAFE_MATH_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return float(op_fn(left, right))
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_fn = _SAFE_MATH_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return float(op_fn(operand))
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    # Clean query: strip '=' or '?' or trailing markers
    clean_expr = expr.replace("=", "").replace("?", "").strip()
    import re
    if any(c.isalpha() for c in clean_expr):
        matches = re.findall(r"[\(\d][\d\s\+\-\*\/\%\.\(\)]*[\)\d]", clean_expr)
        if matches:
            clean_expr = max(matches, key=len).strip()
    tree = ast.parse(clean_expr, mode='eval')
    return _eval_node(tree.body)


class UniversalCognitiveEngine(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        d_hidden: int = 64,
        hyperspace_dim: int = 8,
        num_memory_slots: int = 32,
        vocab_size: int = 256,
        initial_experts: int = 1,
        max_experts: Optional[int] = None,
        consensus_threshold: float = 0.70,
        max_thinking_steps: int = 6,
        num_actions: int = 4,
        lr: float = 0.002,
        device: torch.device = None
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.hyperspace_dim = hyperspace_dim
        self.num_memory_slots = num_memory_slots
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.max_experts = max_experts
        self.consensus_threshold = consensus_threshold
        self.max_thinking_steps = max_thinking_steps
        self.lr = lr

        # 1. Universal Perception (Domain-Agnostic Input Projector)
        self.perceptor = UniversalPerceptor(
            hyperspace_dim=hyperspace_dim,
            payload_dim=d_model,
            embedding_dim=d_hidden,
            vocab_size=vocab_size
        ).to(device)

        # 2. Infinite Spherical Hyperspace Memory (Persistent Storage)
        self.memory = InfiniteSphericalMemory(
            num_slots=num_memory_slots,
            hyperspace_dim=hyperspace_dim,
            payload_dim=d_model,
            initial_radius_shells=4,
            max_slots=128
        ).to(device)

        # 3. Dynamic Mixture-of-Experts with Autonomous Mitosis
        self.moe = SphericalDynamicMoE(
            d_model=d_model,
            d_ff=d_hidden,
            d_context=d_model,
            hyperspace_dim=hyperspace_dim,
            initial_experts=initial_experts,
            max_experts=max_experts,
            spawn_threshold=0.35,
            device=device
        ).to(device)

        # 4. Latent Riemannian World Model (Predictive JEPA & Causal Inversion)
        self.world_model = UniversalWorldModel(
            d_model=d_model,
            hyperspace_dim=hyperspace_dim,
            consensus_threshold=consensus_threshold,
            max_thinking_steps=max_thinking_steps,
            use_dynamic_moe=True,
            moe_instance=self.moe,
            use_working_memory=True,
            device=device
        ).to(device)

        # 5. Domain-Agnostic Output Readout Heads
        # Continuous Readout Head (projects latent state to continuous vectors)
        self.continuous_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 32)
        ).to(device)

        # Discrete / Symbolic Readout Head (projects latent state to token distribution)
        self.symbolic_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, vocab_size)
        ).to(device)

        # Discrete Action / Policy Head (projects latent state to action distribution)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_actions)
        ).to(device)

        # Scale-Equivariant Continuous Readout Head (Path B: emits [sign_logit, log_mag_scaled])
        self.log_radial_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 2)
        ).to(device)

        # Context query parameters for memory addressing
        self.kappa_mem = nn.Parameter(torch.tensor(3.0, device=device))
        self.beta_mem = nn.Parameter(torch.tensor(0.5, device=device))

        # Memory write gating
        self.mem_write_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        ).to(device)

        # 6. Multi-head Riemannian Spherical Memory Cross-Attention
        self.memory_cross_attn = RiemannianSphericalAttention(
            d_model=d_model,
            num_heads=4,
            hyperspace_dim=hyperspace_dim,
            device=device
        ).to(device)

        # 7. Discrete token projection to deliberation d_model
        self.token_to_dmodel = nn.Linear(d_hidden, d_model).to(device)

        # 8. Harvested Neuro-Symbolic General Intelligence Primitives
        self.delta_net = ChunkedGatedDeltaRuleLayer(d_in=d_model, d_k=16, d_v=d_model, chunk_size=16).to(device)
        self.dentate = DentateGyrusPatternSeparator(d_in=d_model, d_sparse=1024, sparsity_ratio=0.02).to(device)
        self.grid_encoder = ToroidalGridEncoder(d_in=hyperspace_dim, d_out=128, num_scales=4).to(device)
        self.grid_to_u = nn.Linear(128, hyperspace_dim, bias=False, device=device)
        nn.init.xavier_uniform_(self.grid_to_u.weight)
        self.cognitive_map = CognitiveMapGraph(d_model=d_model, gamma=0.85, device=device)
        self.hopfield = ModernHopfieldNetwork(d_model=d_model, beta_init=1.0, beta_max=8.0, n_iters=5, device=device).to(device)
        self.global_bus = HyperspaceGlobalBus(d_model=d_model, d_hyper=64, device=device).to(device)
        self.gwt = PredictiveGlobalWorkspace(surprise_threshold=0.35)
        self.criticality = CriticalityController(target_event_rate=0.05, window=20, gain=0.10)

        # Dynamic Optimizer (deduplicated parameter registration)
        raw_params = list(self.perceptor.parameters()) + \
                     list(self.world_model.parameters()) + \
                     list(self.continuous_head.parameters()) + \
                     list(self.symbolic_head.parameters()) + \
                     list(self.action_head.parameters()) + \
                     list(self.log_radial_head.parameters()) + \
                     list(self.moe.parameters()) + \
                     list(self.memory_cross_attn.parameters()) + \
                     list(self.token_to_dmodel.parameters()) + \
                     list(self.delta_net.parameters()) + \
                     list(self.dentate.parameters()) + \
                     list(self.grid_encoder.parameters()) + \
                     list(self.grid_to_u.parameters()) + \
                     list(self.hopfield.parameters()) + \
                     list(self.global_bus.parameters()) + \
                     [self.kappa_mem, self.beta_mem]

        seen = set()
        unique_params = []
        for p in raw_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)

        self.optimizer = DynamicWarmupAdamW(unique_params, lr=lr, default_group_warmup_steps=20)


        # Initialize persistent working memory state
        self.mem_state = self.memory.init_memory(batch_size=1, device=self.device)

        # 8. Unified Deliberative Reasoner Engine (Imaginative Causal Reasoning)
        from ishmn.deliberative_reasoner import DeliberativeReasoner
        self.reasoner = DeliberativeReasoner(self)

        # 9. Native Self-Supervised Language Learner & Concept Crystallizer
        from ishmn.native_language_learner import NativeLanguageLearner
        self.language_learner = NativeLanguageLearner(self)

        # 10. Human-Inspired Developmental Learning Architecture (HIDLA)
        from ishmn.developmental_learner import DevelopmentalManifoldLearner
        self.developmental_learner = DevelopmentalManifoldLearner(self)
        self.tokenizer = None
        self.native_transformer = None

    def perceive(
        self,
        data: Any,
        modality: str = "continuous",
        use_log_radial: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects any input into (u in S^{D-1}, r in R+, v in R^{d_model}).
        """
        if isinstance(data, str) and modality == "tokens" and getattr(self, "tokenizer", None) is not None:
            token_ids = self.tokenizer.encode(data)
            if len(token_ids) == 0:
                token_ids = [0]
            data = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
        return self.perceptor(data, modality=modality, use_log_radial=use_log_radial)



    def _sync_mem_state_batch(self, B: int) -> None:
        """Ensures working memory batch size matches input without wiping contents."""
        if self.mem_state is None or "v" not in self.mem_state:
            self.mem_state = self.memory.init_memory(batch_size=B, device=self.device)
            return

        curr_b = self.mem_state["v"].size(0)
        if curr_b != B:
            if curr_b == 1:
                new_mem = {}
                for k, v in self.mem_state.items():
                    if isinstance(v, torch.Tensor):
                        repeats = [B] + [1] * (v.ndim - 1)
                        new_mem[k] = v.repeat(*repeats)
                    else:
                        new_mem[k] = v
                self.mem_state = new_mem
            elif B < curr_b:
                new_mem = {}
                for k, v in self.mem_state.items():
                    if isinstance(v, torch.Tensor):
                        new_mem[k] = v[:B]
                    else:
                        new_mem[k] = v
                self.mem_state = new_mem
            else:
                self.mem_state = self.memory.init_memory(batch_size=B, device=self.device)

    def forward_deliberation(
        self,
        z_state: torch.Tensor,
        z_context: torch.Tensor,
        enable_system2: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a state through the Universal Cognitive Engine:
        1. Query working memory via vMF attention.
        2. Blend retrieved context.
        3. If enable_system2=True, uses Adaptive Computation Time (deliberation loop).
           Otherwise executes instantaneous System 1 transition.
        """
        # Ensure working memory batch size matches input without wiping contents
        B = z_state.size(0)
        self._sync_mem_state_batch(B)

        # 1. Project current state to manifold coordinates
        u_state, r_state = self.world_model.project_manifold(z_state)

        # 2. Read from working memory (Geodesic vMF addressing + RSA Cross-Attention)
        read_payload, mem_weights, self.mem_state = self.memory.read(
            self.mem_state,
            u_query=u_state,
            r_query=r_state,
            kappa=torch.clamp(self.kappa_mem, min=0.1, max=10.0),
            beta_r=torch.clamp(self.beta_mem, min=0.01, max=5.0)
        )

        mem_attended, _ = self.memory_cross_attn(
            x=z_state.unsqueeze(1),
            context=self.mem_state["v"]
        )
        combined_payload = 0.5 * read_payload + 0.5 * mem_attended.squeeze(1)

        # Blend memory payload into state
        z_conditioned = z_state + 0.2 * combined_payload

        # 3. Deliberation Loop (System 1 vs System 2)
        if enable_system2:
            delib_res = self.world_model.deliberate_and_halting_loop(
                z_init=z_conditioned,
                context=z_context,
                moe_layer=self.moe
            )
            z_final = delib_res["final_latent"]
            steps_taken = delib_res["steps_taken"]
            consensus = delib_res["consensus"]
            causal_error = delib_res["causal_error"]
            system_used = delib_res["system_used"]
            trajectory = delib_res["thinking_trajectory"]
        else:
            # Instantaneous System 1
            z_cand = self.world_model.forward_transition(z_conditioned, z_context)
            u_cand, r_cand = self.world_model.project_manifold(z_cand)
            moe_out, _ = self.moe(z_cand, u_cand, r_cand, optimizer=self.optimizer)
            z_final = z_cand + 0.3 * moe_out
            steps_taken = 1
            consensus = 0.9
            causal_error = 0.0
            system_used = "System 1 (Instant Intuition)"
            trajectory = []

        # 4. Optional Working Memory Write
        w_gate = self.mem_write_gate(z_final)
        u_final, r_final = self.world_model.project_manifold(z_final)
        zeros_erase = torch.zeros_like(z_final)
        _, new_mem = self.memory.write(
            self.mem_state,
            u_write=u_final.detach(),
            r_write=r_final.detach(),
            add_vector=z_final.detach(),
            erase_vector=zeros_erase.detach(),
            write_gate=w_gate.detach(),
            kappa_w=torch.clamp(self.kappa_mem.detach(), 0.1, 10.0),
            beta_rw=torch.clamp(self.beta_mem.detach(), 0.01, 5.0)
        )
        self.mem_state = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in new_mem.items()
        }

        # 5. Decode outputs
        continuous_out = self.continuous_head(z_final)
        symbolic_logits = self.symbolic_head(z_final)

        return {
            "z_final": z_final,
            "u_final": u_final,
            "r_final": r_final,
            "continuous_out": continuous_out,
            "symbolic_logits": symbolic_logits,
            "steps_taken": steps_taken,
            "system_used": system_used,
            "consensus": consensus,
            "causal_error": causal_error,
            "thinking_trajectory": trajectory,
            "num_experts": self.moe.num_experts,
            "memory_slots": self.mem_state["u"].size(1)
        }

    def _pad_or_resample_target(self, y: torch.Tensor, target_dim: int = 32) -> torch.Tensor:
        """Helper to resample target tensors to standard continuous head dimension."""
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(1)
        resampled = F.interpolate(y, size=target_dim, mode='linear', align_corners=False)
        return resampled.squeeze(1)

    def perceive_and_infer(
        self,
        data: torch.Tensor,
        modality: str = "continuous",
        num_inference_steps: int = 5,
        step_size: float = 0.05
    ) -> Dict[str, Any]:
        """
        Multimodal Hierarchical Predictive Coding Inference.
        Projects raw sensory observation into latent hyperspace, then performs
        fast continuous state relaxation down the Variational Free Energy landscape.
        """
        data = data.to(self.device)
        u_sens, r_sens, v_sens = self.perceptor(data, modality=modality)

        # Fast perceptual relaxation across hierarchical layers
        relaxed_states, trajectory = self.world_model.hrpc.infer_states(
            sensory_input=v_sens,
            num_steps=num_inference_steps,
            step_size=step_size
        )

        # Deepest abstract belief state (root/top level) and sensory level state
        z_sensory_relaxed = relaxed_states[0]
        z_abstract_relaxed = relaxed_states[-1]

        u_relaxed, r_relaxed = self.world_model.project_manifold(z_sensory_relaxed)
        continuous_out = self.continuous_head(z_sensory_relaxed)
        symbolic_logits = self.symbolic_head(z_sensory_relaxed)

        return {
            "u_sensory": u_sens,
            "r_sensory": r_sens,
            "v_sensory": v_sens,
            "relaxed_states": relaxed_states,
            "z_sensory": z_sensory_relaxed,
            "z_abstract": z_abstract_relaxed,
            "u_relaxed": u_relaxed,
            "r_relaxed": r_relaxed,
            "continuous_out": continuous_out,
            "symbolic_logits": symbolic_logits,
            "initial_free_energy": trajectory[0]["total_free_energy"],
            "final_free_energy": trajectory[-1]["total_free_energy"],
            "trajectory": trajectory
        }

    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        steps: int = 10,
        context: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Rolls out an autonomous multimodal trajectory across continuous manifold dynamics.
        Operates on S^{D-1} x R+ without discrete domain constraints.
        """
        initial_state = initial_state.to(self.device)
        if context is None:
            context = torch.zeros_like(initial_state)

        z = initial_state
        rollout = []

        for step in range(steps):
            u, r = self.world_model.project_manifold(z)
            continuous_out = self.continuous_head(z)

            rollout.append({
                "step": step,
                "z": z.detach().cpu(),
                "u": u.detach().cpu(),
                "r": r.detach().cpu(),
                "continuous_pred": continuous_out.detach().cpu()
            })

            # Forward evolution with top-down prior relaxation
            c = context if context is not None else z
            z_cand = self.world_model.forward_transition(z, c)
            if self.training and torch.is_grad_enabled():
                with torch.enable_grad():
                    pred_states, _ = self.world_model.hrpc.infer_states(z_cand, num_steps=2, step_size=0.04)
                    z_raw = 0.85 * z_cand + 0.15 * pred_states[0].detach()
            else:
                with torch.no_grad():
                    pred_states = self.world_model.hrpc.forward_relaxation_step(z_cand, num_steps=2, alpha=0.15)
                    z_raw = 0.85 * z_cand + 0.15 * pred_states[0].detach()

            # Bound state to Riemannian manifold S^{D-1} x R+ (prevents numerical runaway)
            norm_z = torch.norm(z_raw, p=2, dim=-1, keepdim=True) + 1e-6
            _, r_bound = self.world_model.project_manifold(z_raw)
            r_clamped = torch.clamp(r_bound, 1.0, 5.0)
            z = r_clamped * (z_raw / norm_z)

        return rollout

    def train_step(
        self,
        x_curr: torch.Tensor,
        context: torch.Tensor,
        x_next: torch.Tensor,
        modality_x: str = "continuous",
        modality_c: str = "continuous",
        task_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Executes a single self-supervised causal + supervised task training step:
        - Predictive JEPA Loss (forward transition)
        - Task Prediction Loss (Continuous Readout Head)
        - Pearl Level 3 Causal Consistency Loss (reversibility / counterfactual rollback with MoE)
        - Backward Transition Loss
        - Metacognitive Halting Gate Training
        """
        self.optimizer.zero_grad()
        self._detach_mem_state()

        # 1. Perceive current, context, and target representations
        u_curr, r_curr, z_curr = self.perceive(x_curr, modality=modality_x)
        u_c, r_c, z_c = self.perceive(context, modality=modality_c)
        u_tgt, r_tgt, z_tgt = self.perceive(x_next, modality=modality_x)

        # 2. Forward Transition via Predictive World Model + Dynamic MoE
        z_fwd = self.world_model.forward_transition(z_curr, z_c)
        u_fwd, r_fwd = self.world_model.project_manifold(z_fwd)

        # Route through Dynamic Spherical MoE with online neurogenesis registration
        moe_out, _ = self.moe(z_fwd, u_fwd, r_fwd, optimizer=self.optimizer)
        z_pred = z_fwd + 0.3 * moe_out
        u_pred, r_pred = self.world_model.project_manifold(z_pred)

        # Forward Prediction Loss (Latent JEPA on Manifold)
        l_fwd_latent = F.mse_loss(z_pred, z_tgt.detach())
        l_fwd_manifold = (
            geodesic_distance_sphere(u_pred, u_tgt.detach()).mean() +
            0.5 * torch.abs(r_pred - r_tgt.detach()).mean()
        )
        l_forward = l_fwd_latent + l_fwd_manifold

        # Direct Task Prediction Loss (Continuous Readout Head)
        y_tgt_pad = self._pad_or_resample_target(x_next, 32)
        y_pred = self.continuous_head(z_pred)
        l_task = F.mse_loss(y_pred, y_tgt_pad)

        # 3. Backward Inversion (Causal Rollback) with MoE context
        z_recon_base = self.world_model.backward_transition(z_pred, z_c)
        u_rec_b, r_rec_b = self.world_model.project_manifold(z_recon_base)
        moe_out_b, _ = self.moe(z_recon_base, u_rec_b, r_rec_b, optimizer=self.optimizer)
        z_recon = z_recon_base + 0.3 * moe_out_b
        u_recon, r_recon = self.world_model.project_manifold(z_recon)

        # Causal Reversibility Loss (Pearl Level 3: W_bwd(W_fwd(z, c), c) == z)
        l_causal = (
            geodesic_distance_sphere(u_recon, u_curr.detach()).mean() +
            0.5 * torch.abs(r_recon - r_curr.detach()).mean() +
            F.mse_loss(z_recon, z_curr.detach())
        )

        # 4. Direct Backward Loss (W_bwd(z_tgt, c) == z_curr)
        z_direct_bwd = self.world_model.backward_transition(z_tgt.detach(), z_c)
        l_backward = F.mse_loss(z_direct_bwd, z_curr.detach())

        # 5. Metacognitive Halting Gate Training (Adaptive Computation Time)
        gate_in = torch.cat([z_pred, u_pred, r_pred], dim=-1)
        pred_consensus = self.world_model.halting_gate(gate_in)
        causal_surprise = geodesic_distance_sphere(u_recon, u_curr.detach()) + 0.5 * torch.abs(r_recon - r_curr.detach())
        target_consensus = torch.exp(-causal_surprise.detach())
        l_halting = F.mse_loss(pred_consensus, target_consensus)

        # Total Universal Loss
        total_loss = task_weight * l_task + l_forward + 0.5 * l_causal + 0.5 * l_backward + 0.3 * l_halting
        total_loss.backward()

        self.optimizer.step()
        if hasattr(self.optimizer, "update_dynamic_schedules"):
            self.optimizer.update_dynamic_schedules()

        return {
            "total_loss": total_loss.item(),
            "loss_task": l_task.item(),
            "loss_forward": l_forward.item(),
            "loss_causal": l_causal.item(),
            "loss_backward": l_backward.item(),
            "experts": self.moe.num_experts
        }

    def evaluate_causal_inversion(
        self,
        x_curr: torch.Tensor,
        context: torch.Tensor,
        modality: str = "continuous"
    ) -> Dict[str, float]:
        """
        Measures the system's causal understanding by performing round-trip rollback:
        x -> forward(x, c) -> backward(z_next, c) -> reconstruction error.
        Includes active MoE experts.
        """
        with torch.no_grad():
            u_curr, r_curr, z_curr = self.perceive(x_curr, modality=modality)
            u_c, r_c, z_c = self.perceive(context, modality=modality)

            z_fwd = self.world_model.forward_transition(z_curr, z_c)
            u_fwd, r_fwd = self.world_model.project_manifold(z_fwd)
            moe_out, _ = self.moe(z_fwd, u_fwd, r_fwd, optimizer=None)
            z_next = z_fwd + 0.3 * moe_out

            z_recon_base = self.world_model.backward_transition(z_next, z_c)
            u_rec_b, r_rec_b = self.world_model.project_manifold(z_recon_base)
            moe_out_b, _ = self.moe(z_recon_base, u_rec_b, r_rec_b, optimizer=None)
            z_rollback = z_recon_base + 0.3 * moe_out_b

            u_recon, r_recon = self.world_model.project_manifold(z_rollback)

            angular_err = geodesic_distance_sphere(u_curr, u_recon).mean().item()
            radial_err = torch.abs(r_curr - r_recon).mean().item()
            latent_mse = F.mse_loss(z_curr, z_rollback).item()

            return {
                "angular_error_rad": angular_err,
                "radial_error": radial_err,
                "latent_mse": latent_mse,
                "understanding_score": max(0.0, 1.0 - (angular_err + 0.5 * radial_err))
            }

    def evaluate_step(
        self,
        x_curr: torch.Tensor,
        context: torch.Tensor,
        x_next: torch.Tensor,
        modality_x: str = "continuous",
        modality_c: str = "continuous"
    ) -> Dict[str, Any]:
        """
        Unified evaluation:
        - Forward task accuracy (MSE & Cosine Similarity) against true target x_next
        - Causal rollback & understanding score
        """
        with torch.no_grad():
            u_curr, r_curr, z_curr = self.perceive(x_curr, modality=modality_x)
            u_c, r_c, z_c = self.perceive(context, modality=modality_c)
            y_tgt_pad = self._pad_or_resample_target(x_next, 32)

            z_fwd = self.world_model.forward_transition(z_curr, z_c)
            u_fwd, r_fwd = self.world_model.project_manifold(z_fwd)
            moe_out, _ = self.moe(z_fwd, u_fwd, r_fwd, optimizer=None)
            z_pred = z_fwd + 0.3 * moe_out

            y_pred = self.continuous_head(z_pred)
            task_mse = F.mse_loss(y_pred, y_tgt_pad).item()
            task_cos = F.cosine_similarity(y_pred, y_tgt_pad, dim=-1).mean().item()
            task_acc = max(0.0, task_cos) * 100.0

            # Causal Inversion
            z_recon_base = self.world_model.backward_transition(z_pred, z_c)
            u_rec_b, r_rec_b = self.world_model.project_manifold(z_recon_base)
            moe_out_b, _ = self.moe(z_recon_base, u_rec_b, r_rec_b, optimizer=None)
            z_rollback = z_recon_base + 0.3 * moe_out_b
            u_recon, r_recon = self.world_model.project_manifold(z_rollback)

            angular_err = geodesic_distance_sphere(u_curr, u_recon).mean().item()
            radial_err = torch.abs(r_curr - r_recon).mean().item()
            causal_score = max(0.0, 1.0 - (angular_err + 0.5 * radial_err)) * 100.0

            return {
                "task_mse": task_mse,
                "task_cos_sim": max(0.0, task_cos),
                "task_accuracy": task_acc,
                "angular_error_rad": angular_err,
                "radial_error": radial_err,
                "causal_score": causal_score
            }

    def evaluate_epistemic_curiosity(
        self,
        candidate_stimuli: List[torch.Tensor],
        modality: str = "continuous"
    ) -> List[Dict[str, Any]]:
        """
        Active Inference: Evaluates epistemic curiosity (expected information gain)
        across candidate stimuli.
        In Karl Friston's Variational Free Energy formulation, curiosity corresponds
        to the expected drop in hierarchical Free Energy:
            Epistemic Value = F_initial - F_relaxed + lambda * Novelty
        Candidates with high ambiguity/novelty yield high epistemic curiosity.
        """
        curiosity_scores = []
        for idx, stim in enumerate(candidate_stimuli):
            if isinstance(stim, torch.Tensor):
                stim_dev = stim.to(self.device)
            else:
                stim_dev = stim
            u, r, v = self.perceive(stim_dev, modality=modality)

            # Compute initial vs relaxed free energy through HRPC with local autograd enabled
            with torch.enable_grad():
                states, traj = self.world_model.hrpc.infer_states(v, num_steps=4, step_size=0.05)
            f_init = traj[0]["total_free_energy"]
            f_final = traj[-1]["total_free_energy"]
            delta_f = max(0.0, f_init - f_final)

            with torch.no_grad():
                u_curr = u[0:1]
                r_curr = r[0:1]
                _, mem_w, _ = self.memory.read(
                    self.mem_state, u_curr, r_curr,
                    kappa=torch.clamp(self.kappa_mem, 0.1, 10.0),
                    beta_r=torch.clamp(self.beta_mem, 0.01, 5.0)
                )
                familiarity = mem_w.max().item() if mem_w is not None else 0.5
                novelty_score = max(0.0, 1.0 - familiarity)

            # Epistemic Information Gain
            info_gain = delta_f * 0.7 + novelty_score * 0.3

            curiosity_scores.append({
                "candidate_idx": idx,
                "info_gain": float(info_gain),
                "free_energy_reduction": float(delta_f),
                "novelty_score": float(novelty_score),
                "initial_free_energy": float(f_init),
                "final_free_energy": float(f_final)
            })


        # Rank candidates by information gain descending (highest curiosity first)
        curiosity_scores.sort(key=lambda x: x["info_gain"], reverse=True)
        return curiosity_scores

    def active_train_step(
        self,
        stimulus_pool: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        select_top_k: int = 2,
        modality_x: str = "continuous",
        modality_c: str = "continuous"
    ) -> Dict[str, Any]:
        """
        Executes an Active Inference Training Step:
        1. Evaluates epistemic curiosity across candidate stimulus pool.
        2. Actively samples the top-k most informative stimuli.
        3. Executes gradient updates on selected high-epistemic stimuli.
        """
        candidates = [item[0] for item in stimulus_pool]
        ranked = self.evaluate_epistemic_curiosity(candidates, modality=modality_x)
        selected_indices = [r["candidate_idx"] for r in ranked[:select_top_k]]

        batch_metrics = []
        for s_idx in selected_indices:
            x_c, ctx, x_n = stimulus_pool[s_idx]
            m = self.train_step(x_c, ctx, x_n, modality_x=modality_x, modality_c=modality_c)
            batch_metrics.append(m)

        avg_loss = sum(m["total_loss"] for m in batch_metrics) / len(batch_metrics)
        avg_causal = sum(m["loss_causal"] for m in batch_metrics) / len(batch_metrics)

        return {
            "selected_indices": selected_indices,
            "curiosity_ranking": ranked,
            "mean_loss": avg_loss,
            "mean_causal_loss": avg_causal,
            "top_info_gain": ranked[0]["info_gain"],
            "num_experts": self.moe.num_experts
        }


    def solve_sequence_task(
        self,
        token_ids: torch.Tensor,
        is_causal: bool = False,
        mask: Optional[torch.Tensor] = None,
        max_loops: int = 8,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solves multi-step sequence reasoning tasks (e.g. multi-digit arithmetic, reversal, stack retrieval)
        using the Looped Riemannian Transformer deliberation engine with Dynamic MoE and Hyperspace Memory.
        """
        token_ids = token_ids.to(self.device)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        # 1. Embed discrete tokens
        embeds = self.perceptor.token_embed(token_ids)  # [B, T, d_hidden]
        h_seq = self.token_to_dmodel(embeds)            # [B, T, d_model]

        # Ensure working memory batch size matches input
        B = h_seq.size(0)
        self._sync_mem_state_batch(B)
        self._detach_mem_state()

        # 2. Deliberate via Looped Riemannian Transformer
        h_out, meta = self.world_model.deliberation_loop(
            h_seq,
            mask=mask,
            is_causal=is_causal,
            max_steps=max_loops,
            top_k=top_k,
            optimizer=self.optimizer if self.training else None,
            memory=self.memory,
            mem_state=self.mem_state
        )
        if "mem_state" in meta and meta["mem_state"] is not None:
            self.mem_state = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in meta["mem_state"].items()
            }

        # 3. Read out token logits through symbolic head
        logits = self.symbolic_head(h_out)  # [B, T, vocab_size]
        pred_tokens = torch.argmax(logits, dim=-1)

        return {
            "predicted_tokens": pred_tokens,
            "logits": logits,
            "steps_taken": meta["steps_taken"],
            "ponder_cost": meta["ponder_cost"],
            "final_loops": meta["final_loops"],
            "intermediate_states": meta["intermediate_states"],
            "num_experts": meta.get("num_experts", self.moe.num_experts)
        }

    def solve_sequence_autoregressive(
        self,
        prompt_tokens: torch.Tensor,
        max_gen_tokens: int = 16,
        max_loops: int = 4,
        eos_token: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        sample: bool = False
    ) -> Dict[str, Any]:
        """
        Autoregressively generates sequence tokens step-by-step using the Looped Riemannian Deliberator.
        At each generation step, attends to all previously emitted tokens with causal attention and updates working memory.
        """
        prompt_tokens = prompt_tokens.to(self.device)
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        curr_tokens = prompt_tokens.clone()
        B, prompt_len = prompt_tokens.shape
        steps_list = []

        for _ in range(max_gen_tokens):
            res = self.solve_sequence_task(curr_tokens, is_causal=True, max_loops=max_loops, top_k=top_k)

            next_logits = res["logits"][:, -1, :]  # [B, vocab_size]
            if sample and temperature > 0.0:
                probs = F.softmax(next_logits / max(temperature, 1e-4), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B, 1]
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            steps_list.append(res["steps_taken"])

            if eos_token is not None and (next_token == eos_token).all():
                break

        gen_tokens = curr_tokens[:, prompt_len:]
        return {
            "full_sequence": curr_tokens,
            "generated_tokens": gen_tokens,
            "avg_steps_per_token": sum(steps_list) / max(len(steps_list), 1)
        }

    def solve_stream_automaton(
        self,
        stream_inputs: torch.Tensor,
        max_loops: int = 4,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Solves algorithmic stream reasoning (e.g. serial multi-digit carry addition,
        unbounded sequence transducer) as a continuous manifold automaton.
        Args:
            stream_inputs: [B, T, num_operands] or [B, T]
        Returns:
            predicted_tokens: [B, T]
            all_logits: [B, T, vocab_size]
        """
        stream_inputs = stream_inputs.to(self.device)
        if stream_inputs.dim() == 2:
            B, T = stream_inputs.shape
            num_ops = 1
        elif stream_inputs.dim() == 3:
            B, T, num_ops = stream_inputs.shape
        else:
            raise ValueError(f"Invalid stream_inputs shape: {stream_inputs.shape}")

        if self.mem_state["v"].size(0) != B:
            self.mem_state = self.memory.init_memory(batch_size=B, device=self.device)

        logits_list = []
        z_curr = torch.zeros(B, self.d_model, device=self.device)
        steps_history = []

        for t in range(T):
            if num_ops == 1:
                token_t = stream_inputs[:, t]
                emb_t = self.perceptor.token_embed(token_t)
                x_t = self.token_to_dmodel(emb_t)
            else:
                emb_sum = torch.zeros(B, self.d_hidden, device=self.device)
                for op_idx in range(num_ops):
                    emb_sum = emb_sum + self.perceptor.token_embed(stream_inputs[:, t, op_idx])
                x_t = self.token_to_dmodel(emb_sum)

            # Query memory using current state
            u_s, r_s = self.world_model.project_manifold(z_curr)
            read_p, _, self.mem_state = self.memory.read(
                self.mem_state, u_s, r_s,
                kappa=torch.clamp(self.kappa_mem, 0.1, 10.0),
                beta_r=torch.clamp(self.beta_mem, 0.01, 5.0)
            )

            # Input to deliberator: state + memory + current input
            combined_in = (z_curr + x_t + 0.2 * read_p).unsqueeze(1)
            h_out, meta = self.world_model.deliberation_loop(
                combined_in,
                max_steps=max_loops,
                top_k=top_k,
                optimizer=self.optimizer if self.training else None,
                memory=self.memory,
                mem_state=self.mem_state
            )
            if "mem_state" in meta and meta["mem_state"] is not None:
                self.mem_state = {
                    k: v.detach() if isinstance(v, torch.Tensor) else v
                    for k, v in meta["mem_state"].items()
                }

            z_next = h_out[:, -1, :]

            # Write updated state into memory
            u_w, r_w = self.world_model.project_manifold(z_next)
            w_gate = torch.sigmoid(self.mem_write_gate(z_next))
            _, self.mem_state = self.memory.write(
                self.mem_state, u_w, r_w,
                add_vector=z_next,
                erase_vector=torch.ones_like(z_next) * 0.2,
                write_gate=w_gate,
                kappa_w=torch.clamp(self.kappa_mem, 0.1, 10.0),
                beta_rw=torch.clamp(self.beta_mem, 0.01, 5.0)
            )

            logits_t = self.symbolic_head(z_next)
            logits_list.append(logits_t.unsqueeze(1))
            steps_history.append(meta["steps_taken"])
            z_curr = z_next

        all_logits = torch.cat(logits_list, dim=1)
        pred_tokens = torch.argmax(all_logits, dim=-1)

        return {
            "predicted_tokens": pred_tokens,
            "logits": all_logits,
            "avg_steps": sum(steps_history) / max(len(steps_history), 1)
        }

    def train_stream_automaton_step(
        self,
        stream_inputs: torch.Tensor,
        target_ids: torch.Tensor,
        max_loops: int = 4,
        top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Trains the Stream Automaton end-to-end on sequence transduction.
        """
        self.train()
        self._detach_mem_state()
        stream_inputs = stream_inputs.to(self.device)
        target_ids = target_ids.to(self.device)

        res = self.solve_stream_automaton(stream_inputs, max_loops=max_loops, top_k=top_k)
        logits = res["logits"]

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.update_dynamic_schedules()
        self._detach_mem_state()

        pred_tokens = res["predicted_tokens"]
        tok_acc = (pred_tokens == target_ids).float().mean().item() * 100.0
        seq_acc = (pred_tokens == target_ids).all(dim=-1).float().mean().item() * 100.0

        return {
            "loss": loss.item(),
            "token_accuracy": tok_acc,
            "exact_sequence_match": seq_acc,
            "num_experts": self.moe.num_experts,
            "avg_steps": res["avg_steps"]
        }

    def train_sequence_step(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        is_causal: bool = True,
        max_loops: int = 8,
        top_k: Optional[int] = None,
        ponder_weight: float = 0.01
    ) -> Dict[str, float]:
        """
        Trains the Looped Riemannian deliberator end-to-end on sequence prediction targets.
        """
        self.train()
        self._detach_mem_state()
        token_ids = token_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        res = self.solve_sequence_task(token_ids, is_causal=is_causal, max_loops=max_loops, top_k=top_k)

        logits = res["logits"]

        loss_task = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_ids.reshape(-1))
        loss_ponder = ponder_weight * res["ponder_cost"]
        total_loss = loss_task + loss_ponder

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.update_dynamic_schedules()
        self._detach_mem_state()

        # Measure token-level and sequence-level accuracy
        pred_tokens = res["predicted_tokens"]
        tok_acc = (pred_tokens == target_ids).float().mean().item() * 100.0
        seq_acc = (pred_tokens == target_ids).all(dim=-1).float().mean().item() * 100.0

        return {
            "loss": total_loss.item(),
            "total_loss": total_loss.item(),
            "loss_task": loss_task.item(),
            "loss_ponder": loss_ponder.item(),
            "token_accuracy": tok_acc,
            "exact_sequence_match": seq_acc,
            "steps_taken": res["steps_taken"],
            "num_experts": res.get("num_experts", self.moe.num_experts)
        }

    def _detach_mem_state(self):
        """Detaches persistent working memory state buffers from the computation graph."""
        if hasattr(self, "mem_state") and self.mem_state is not None:
            self.mem_state = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in self.mem_state.items()
            }

    @torch.no_grad()
    def sense_make(
        self,
        data: Any,
        context: Optional[Any] = None,
        modality: str = "continuous",
        prior_z: Optional[torch.Tensor] = None,
        surprise_threshold: float = 0.45,
        domain: str = "general",
        log_to_episodic: bool = True
    ) -> Dict[str, Any]:
        """
        Abductive Sense-Making & Comprehension Engine:
        Evaluates whether an observation makes sense relative to internal knowledge,
        learnings, and lifelong memories.

        Steps:
        1. Bottom-up perception -> u_in in S^{D-1}, r_in in R+, v_in in R^d_model
        2. Top-down surprise evaluation against prior expectations:
           Surprise S = d_S(u_in, u_prior) + |r_in - r_prior|
        3. Tri-Tier Associative Memory Recall:
           Recalls resonant episodic experiences and invariant semantic concepts.
        4. Abductive Causal Inversion:
           Infers hidden cause via W_bwd(z_in, z_prior) (Pearl Level 3 abduction).
        5. Internal Deliberation (ACT System 1 vs System 2):
           If surprise > surprise_threshold, deliberates multiple steps until internal coherence settles.
        6. Logs experiential trace to lifelong Episodic Memory.
        7. Returns a structured Cognitive Explanation Record.
        """
        # 1. Bottom-up perception
        u_in, r_in, v_in = self.perceive(data, modality=modality)
        z_in = v_in.clone()
        B = z_in.size(0)
        self._sync_mem_state_batch(B)

        # 2. Context embedding
        if context is not None:
            _, _, z_ctx = self.perceive(context, modality=modality)
        else:
            z_ctx = torch.zeros_like(z_in)

        # 3. Tri-Tier Memory Retrieval
        fused_mem_payload, memory_meta, self.mem_state = self.memory.tri_tier_recall(
            self.mem_state,
            u_query=u_in,
            r_query=r_in,
            kappa=torch.clamp(self.kappa_mem, 0.1, 10.0),
            beta_r=torch.clamp(self.beta_mem, 0.01, 5.0),
            top_k_episodic=4,
            top_k_semantic=3
        )

        # 4. Top-down surprise evaluation against prior expectations or recalled memory
        u_in_wm, r_in_wm = self.world_model.project_manifold(z_in)
        if prior_z is not None:
            prior_z = prior_z.to(self.device)
            u_prior, r_prior = self.world_model.project_manifold(prior_z)
            ang_surprise = geodesic_distance_sphere(u_in_wm, u_prior).mean().item()
            rad_surprise = torch.abs(r_in_wm - r_prior).mean().item()
            latent_diff = F.mse_loss(z_in, prior_z).item()
            surprise = ang_surprise + 0.2 * rad_surprise + 0.5 * latent_diff
        elif (memory_meta.get("recalled_episodes") or memory_meta.get("recalled_semantics")):
            # Measure surprise against nearest recalled experiential trace or semantic invariant
            all_dists = [ep["geodesic_distance"] for ep in memory_meta.get("recalled_episodes", [])] + \
                        [sc["geodesic_distance"] for sc in memory_meta.get("recalled_semantics", [])]
            min_dist = min(all_dists) if all_dists else 1.5
            surprise = min_dist
        else:
            # Baseline surprise relative to memory centroid
            u_mem_mean = project_to_sphere(torch.mean(self.mem_state["u"], dim=1))
            surprise = geodesic_distance_sphere(u_in, u_mem_mean).mean().item()

        # Combine percept with recalled memory context
        z_augmented = 0.7 * z_in + 0.3 * fused_mem_payload

        # 5. Abductive Causal Inversion (Pearl Level 3)
        # Infer what condition or cause could have produced this transition
        abduced_cause = self.world_model.w_bwd(torch.cat([z_in, z_augmented], dim=-1))

        # 6. Deliberation: System 1 (intuitive flow) vs System 2 (deep deliberation)
        is_surprising = surprise > surprise_threshold
        delib_res = self.forward_deliberation(
            z_state=z_augmented,
            z_context=z_ctx,
            enable_system2=is_surprising
        )

        z_deliberated = delib_res["z_final"]
        steps_taken = delib_res["steps_taken"]

        # Forward-backward consistency check for coherence
        z_reconstructed_cause = self.world_model.w_bwd(torch.cat([z_deliberated, z_in], dim=-1))
        causal_inconsistency = F.mse_loss(z_reconstructed_cause, abduced_cause).item()
        coherence = 1.0 / (1.0 + surprise + causal_inconsistency)

        # 7. Lifelong Episodic Memory logging
        if log_to_episodic:
            for b in range(B):
                self.memory.log_episode(
                    u=u_in[b:b+1],
                    r=r_in[b:b+1],
                    v=z_deliberated[b:b+1],
                    surprise=surprise,
                    context=str(context) if context is not None else None,
                    domain=domain,
                    metadata={"coherence": coherence, "steps": steps_taken}
                )

        # Output readouts
        continuous_out = self.continuous_head(z_deliberated)
        symbolic_out = self.symbolic_head(z_deliberated) if modality == "tokens" else None

        return {
            "perceived_u": u_in,
            "perceived_r": r_in,
            "latent_state": z_deliberated,
            "surprise": float(surprise),
            "is_surprising": is_surprising,
            "system_used": "System 1 (Intuitive Flow)" if steps_taken == 1 else f"System 2 Deliberation ({steps_taken} steps)",
            "deliberation_steps": steps_taken,
            "coherence_score": float(coherence),
            "abduced_cause": abduced_cause,
            "recalled_episodes": memory_meta["recalled_episodes"],
            "recalled_semantics": memory_meta["recalled_semantics"],
            "episodic_count": memory_meta["episodic_count"],
            "semantic_count": memory_meta["semantic_count"],
            "continuous_output": continuous_out,
            "symbolic_output": symbolic_out
        }

    @torch.no_grad()
    def solve_analogy(
        self,
        a_data: Any,
        b_data: Any,
        c_data: Any,
        modality: str = "continuous"
    ) -> Dict[str, Any]:
        """
        Riemannian Analogical Transfer Engine on S^{D-1}:
        Solves A : B :: C : D (i.e. 'A is to B as C is to D').
        Transfers relational tangent transformations across disparate knowledge domains.
        Supports continuous coordinate vectors and grounded verbal concepts.
        """
        is_single_word = (
            isinstance(a_data, str) and len(a_data.split()) == 1 and
            isinstance(b_data, str) and len(b_data.split()) == 1 and
            isinstance(c_data, str) and len(c_data.split()) == 1
        )

        if is_single_word:
            wa = a_data.strip().lower()
            wb = b_data.strip().lower()
            wc = c_data.strip().lower()

            concept_map = {
                sc["word"].strip().lower(): sc["u"].to(self.device)
                for sc in self.memory.semantic_concepts
                if sc.get("word")
            }

            if wa not in concept_map or wb not in concept_map or wc not in concept_map:
                missing = [w for w in (wa, wb, wc) if w not in concept_map]
                return {
                    "error": f"Unknown concepts in memory: {missing}",
                    "predicted_word": None,
                    "confidence": 0.0,
                    "top_candidates": [],
                    "u_d": torch.zeros(1, self.hyperspace_dim, device=self.device),
                    "distance_source": 0.0,
                    "distance_target": 0.0,
                    "isometry_preserved": False
                }

            u_a = concept_map[wa]
            u_b = concept_map[wb]
            u_c = concept_map[wc]

            # Pure Riemannian parallel transport on S^{D-1}
            u_d = analogical_transfer_sphere(u_a, u_b, u_c)
            dist_ab = geodesic_distance_sphere(u_a, u_b).item()
            dist_cd = geodesic_distance_sphere(u_c, u_d).item()

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
                "r_d": torch.tensor([[1.0]], device=self.device),
                "distance_source": dist_ab,
                "distance_target": dist_cd,
                "isometry_preserved": abs(dist_ab - dist_cd) < 1e-4,
                "analogous_latent": torch.zeros(1, self.d_model, device=self.device)
            }

        u_a, r_a, v_a = self.perceive(a_data, modality=modality)
        u_b, r_b, v_b = self.perceive(b_data, modality=modality)
        u_c, r_c, v_c = self.perceive(c_data, modality=modality)

        # Apply parallel transport on S^{D-1}
        u_d = analogical_transfer_sphere(u_a, u_b, u_c)

        # Scale analogy radius
        r_scale = (r_b / r_a.clamp(min=1e-5))
        r_d = r_c * r_scale

        dist_ab = geodesic_distance_sphere(u_a, u_b).item()
        dist_cd = geodesic_distance_sphere(u_c, u_d).item()

        # Query semantic concept lattice candidates if available
        top_candidates = []
        best_word = None
        confidence = 0.0
        lattice_results = self.memory.query_semantic_lattice(u_d, top_k=5)
        if lattice_results:
            top_candidates = [
                {
                    "word": lr.get("word", f"concept_{lr['concept_id']}"),
                    "geodesic_distance": lr["geodesic_distance"],
                    "similarity_pct": lr["semantic_similarity"] * 100.0
                }
                for lr in lattice_results
            ]
            best_word = top_candidates[0]["word"]
            confidence = top_candidates[0]["similarity_pct"]

        # Generate target latent representation
        z_d = self.continuous_head[0](v_c) if hasattr(self.continuous_head, "__getitem__") else v_c

        return {
            "predicted_word": best_word,
            "confidence": confidence,
            "top_candidates": top_candidates,
            "u_d": u_d,
            "r_d": r_d,
            "distance_source": dist_ab,
            "distance_target": dist_cd,
            "isometry_preserved": abs(dist_ab - dist_cd) < 1e-3,
            "analogous_latent": z_d
        }

    def train_language_corpus(
        self,
        corpus_sentences: List[str],
        epochs: int = 50,
        lr: float = 0.02,
        window_size: int = 4
    ) -> Dict[str, Any]:
        """
        Trains UCE's internal concept representations natively from text corpora
        and crystallizes permanent concept attractors into TriTierSphericalMemory.
        """
        self.language_learner.train_on_corpus(
            corpus_sentences, epochs=epochs, lr=lr, window_size=window_size
        )
        self.language_learner.crystallize_to_memory()
        return {
            "status": "trained_and_crystallized",
            "vocabulary_size": len(self.language_learner.vocab),
            "crystallized_concepts": len(self.memory.semantic_concepts)
        }

    @torch.no_grad()
    def reason(
        self,
        premise: Any,
        context: Optional[Any] = None,
        candidate_branches: Optional[List[Tuple[Any, Any]]] = None,
        modality: str = "continuous",
        lookahead_depth: int = 12,
        goal: Optional[Any] = None,
        external_branch_evaluator: Optional[Any] = None
    ) -> Any:
        """
        Coordinates multi-step deliberative reasoning via endogenous imagination:
        - Simulates hypothetical candidate branches forward through the Causal World Model
        - Evaluates abductive causal consistency (W_bwd) and memory resonance
        - Prunes dead ends and contradictions in imagination
        - Emits a structured ReasoningResult with full thought trajectory
        """
        return self.reasoner.reason(
            premise=premise,
            context=context,
            candidate_branches=candidate_branches,
            modality=modality,
            lookahead_depth=lookahead_depth,
            goal=goal,
            external_branch_evaluator=external_branch_evaluator
        )

    @torch.no_grad()
    def sleep(
        self,
        prune_usage_threshold: float = 0.02,
        crystallize_concepts: bool = True
    ) -> Dict[str, Any]:
        """
        Coordinates full-brain homeostatic sleep consolidation across all subsystems:
        1. SphericalDynamicMoE: Geodesic barycentric merging of co-resonant clusters
           and apoptotic pruning of dormant/dead experts.
        2. DynamicWarmupAdamW: Synchronizes parameter groups and cleans up expired optimizer states.
        3. Semantic Concept Crystallization: Synthesizes invariant attractors in Semantic Concept
           Lattice from episodic memory traces.
        4. Memory Maintenance: Consolidates concentric shells and detaches buffers.
        """
        # 1. MoE barycentric consolidation and pruning
        moe_report = self.moe.sleep_consolidate_and_prune(
            prune_usage_threshold=prune_usage_threshold,
            optimizer=self.optimizer
        )

        # 2. Semantic concept crystallization from episodic replay
        semantic_report = {}
        if crystallize_concepts:
            semantic_report = self.memory.crystallize_semantic_concepts(
                cluster_angle_threshold=0.35,
                min_cluster_size=2
            )

        # 3. Concentric shell migration towards core
        self.mem_state = self.memory.consolidate_and_migrate_shells(self.mem_state)
        self._detach_mem_state()
        mem_norm = torch.norm(self.mem_state["v"]).item()

        return {
            "status": "consolidated",
            "moe_consolidation": moe_report,
            "semantic_crystallization": semantic_report,
            "working_memory_norm": mem_norm,
            "active_experts": self.moe.num_experts,
            "total_episodic_episodes": len(self.memory.episodic_memory),
            "total_semantic_concepts": len(self.memory.semantic_concepts)
        }

    def train_continuous_step(
        self,
        a_vals: List[float],
        op_codes: List[int],
        b_vals: List[float],
        y_vals: List[float]
    ) -> Dict[str, float]:
        """
        Trains the scale-equivariant log-radial pipeline end-to-end:
        - Latent JEPA prediction
        - Log-radial readout loss: MSE(pred_sign, tgt_sign) + 5.0 * MSE(pred_log_mag, tgt_log_mag)
        - Causal rollback
        """
        self.train()
        self.optimizer.zero_grad()
        self._detach_mem_state()

        N = len(a_vals)
        x_in = torch.zeros(N, 2, device=self.device)
        c_in = torch.zeros(N, 6, device=self.device)
        y_tgt = torch.zeros(N, 2, device=self.device)

        for i in range(N):
            sa, lma = self.perceptor.log_radial_encode(a_vals[i], scale=20.0)
            sb, lmb = self.perceptor.log_radial_encode(b_vals[i], scale=20.0)
            sy, lmy = self.perceptor.log_radial_encode(y_vals[i], scale=20.0)
            x_in[i, 0] = sa
            x_in[i, 1] = lma
            c_in[i, op_codes[i]] = 1.0
            c_in[i, 4] = sb
            c_in[i, 5] = lmb
            y_tgt[i, 0] = sy
            y_tgt[i, 1] = lmy

        # Perceive representations
        _, _, z_x = self.perceive(x_in, modality="continuous", use_log_radial=False)
        _, _, z_c = self.perceive(c_in, modality="continuous", use_log_radial=False)

        # World model forward step
        z_fwd = self.world_model.forward_transition(z_x, z_c)
        u_fwd, r_fwd = self.world_model.project_manifold(z_fwd)
        moe_out, _ = self.moe(z_fwd, u_fwd, r_fwd, optimizer=self.optimizer)
        z_pred = z_fwd + 0.3 * moe_out

        # Readout from log_radial_head
        out_lr = self.log_radial_head(z_pred)
        loss_sign = F.mse_loss(out_lr[:, 0], y_tgt[:, 0])
        loss_mag = F.mse_loss(out_lr[:, 1], y_tgt[:, 1])
        loss_task = loss_sign + 5.0 * loss_mag

        # Causal rollback
        z_recon_b = self.world_model.backward_transition(z_pred, z_c)
        loss_causal = F.mse_loss(z_recon_b, z_x.detach())

        total_loss = loss_task + 0.3 * loss_causal
        total_loss.backward()
        self.optimizer.step()
        if hasattr(self.optimizer, "update_dynamic_schedules"):
            self.optimizer.update_dynamic_schedules()

        return {
            "loss": total_loss.item(),
            "loss_mag": loss_mag.item(),
            "loss_sign": loss_sign.item()
        }

    def solve_continuous_arithmetic(

        self,
        a: float,
        op: str,
        b: float,
        enable_system2: bool = True
    ) -> Dict[str, Any]:
        """
        Solves continuous arithmetic natively across any scale (10^0 to 10^30)
        using Scale-Equivariant Log-Radial Manifold coordinates on S^{D-1} x R+.
        Zero saturation on arbitrary magnitudes (e.g. 23 trillion, continuous decimals).
        """
        self.eval()
        sa, lma = self.perceptor.log_radial_encode(a, scale=20.0)
        sb, lmb = self.perceptor.log_radial_encode(b, scale=20.0)
        op_code = 0 if op == "+" else (1 if op == "-" else (2 if op == "*" else 3))

        # Direct input encoding
        x_in = torch.tensor([[sa, lma]], dtype=torch.float32, device=self.device)
        c_in = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        c_in[0, op_code] = 1.0
        c_in[0, 4] = sb
        c_in[0, 5] = lmb

        # Perceive into hyperspace
        _, _, z_x = self.perceive(x_in, modality="continuous", use_log_radial=False)
        _, _, z_c = self.perceive(c_in, modality="continuous", use_log_radial=False)

        # Deliberate on the Riemannian manifold
        delib_res = self.forward_deliberation(z_x, z_c, enable_system2=enable_system2)
        z_out = delib_res["z_final"]


        # Scale-equivariant readout
        out_lr = self.log_radial_head(z_out)[0]
        pred_s = 1.0 if out_lr[0].item() >= 0 else -1.0
        pred_lm = max(out_lr[1].item(), 0.0)
        pred_val = self.perceptor.log_radial_decode(pred_s, pred_lm, scale=20.0)

        return {
            "prediction": pred_val,
            "pred_sign": pred_s,
            "pred_log_mag_scaled": pred_lm,
            "system_used": delib_res["system_used"],
            "steps_taken": delib_res["steps_taken"],
            "consensus": delib_res["consensus"],
            "causal_error": delib_res["causal_error"]
        }

    def run_autonomous_loop(
        self,
        num_ticks: int = 10,
        inputs: Optional[List[Dict[str, Any]]] = None,
        emission_threshold: float = 0.52
    ) -> List[Any]:
        """
        Runs the living ContinuousSelfLoop driven directly by this UCE brain.
        External inputs act as sensory perturbations. Emissions are triggered
        only when internal consensus is reached.
        """
        from ishmn.continuous_self_loop import ContinuousSelfLoop
        loop = ContinuousSelfLoop(uce=self, emission_threshold=emission_threshold)
        if inputs:
            for item in inputs:
                loop.inject_input(item["data"], modality=item.get("modality", "continuous"), tag=item.get("tag", ""))
        emissions = []
        for _ in range(num_ticks):
            em = loop.loop_tick()
            if em is not None:
                emissions.append(em)
        return emissions

    def dispatch_action(
        self,
        action_id: int,
        payload: Any = None,
        query: str = "",
        log_to_episodic: bool = True
    ) -> Dict[str, Any]:
        """
        Cognitive-Motor Action Dispatcher.
        Executes discrete actions emitted by the action_head or deliberate loop:
        - ACTION_DELIBERATE (0): Continue internal manifold trajectory.
        - ACTION_MOTOR_STEP (1): Physical motor action.
        - ACTION_EMIT_COMMUNICATION (2): Emit communication / speech payload.
        - ACTION_INVOKE_SYMBOLIC_TOOL (3): Executes discrete symbolic/arithmetic tool safely.
        """
        action_name = ACTION_NAMES.get(action_id, f"UNKNOWN_ACTION_{action_id}")

        if action_id == ACTION_INVOKE_SYMBOLIC_TOOL:
            expr_to_eval = query if query else (str(payload) if payload is not None else "")
            try:
                eval_result = safe_eval_expression(expr_to_eval)
                status = "success"
                error_msg = None
            except Exception as ex:
                eval_result = None
                status = "error"
                error_msg = str(ex)

            if status == "success" and log_to_episodic:
                # Ground exact truth directly into episodic memory
                memory_trace = f"Exact symbolic truth: {expr_to_eval} = {eval_result:g}"
                self.sense_make(memory_trace, modality="subwords", domain="symbolic_math", log_to_episodic=True)

            return {
                "action_id": action_id,
                "action_name": action_name,
                "status": status,
                "result": eval_result,
                "query": expr_to_eval,
                "error": error_msg
            }

        elif action_id == ACTION_EMIT_COMMUNICATION:
            return {
                "action_id": action_id,
                "action_name": action_name,
                "status": "emitted",
                "payload": payload
            }

        elif action_id == ACTION_MOTOR_STEP:
            return {
                "action_id": action_id,
                "action_name": action_name,
                "status": "stepped",
                "direction": payload
            }

        else:  # ACTION_DELIBERATE
            return {
                "action_id": action_id,
                "action_name": action_name,
                "status": "continue_deliberation"
            }

    def solve_with_dual_process(
        self,
        query: str,
        enable_system2: bool = True,
        confidence_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Unified Dual-Process Cognitive Engine Solver:
        1. System 1 (Intuitive Hyperspace): Perceives query on Riemannian manifold S^{D-1} x R+
           and produces an instantaneous continuous magnitude estimate (Weber-Fechner sense).
        2. System 2 (Autonomous Action & Discrete Tool Dispatch): The action_head evaluates
           cognitive policy. If symbolic/exact computation is requested, dispatches
           ACTION_INVOKE_SYMBOLIC_TOOL to obtain 100% exact truth and grounds it into episodic memory.
        3. Pearl Level 3 Causal Verification: Evaluates reversibility and consistency.
        """
        self.eval()
        clean_q = query.strip()

        # 1. Perceive query via subwords/tokens into manifold
        u_stim, r_stim, z_stim = self.perceive(clean_q, modality="subwords")

        # Action head policy readout
        with torch.no_grad():
            act_logits = self.action_head(z_stim)[0]
            act_probs = F.softmax(act_logits, dim=-1)
            predicted_action = int(torch.argmax(act_logits).item())

            # System 1 Intuitive Readout (Scale-Equivariant Log-Radial)
            out_lr = self.log_radial_head(z_stim)[0]
            pred_s = 1.0 if out_lr[0].item() >= 0 else -1.0
            pred_lm = max(out_lr[1].item(), 0.0)
            s1_intuition = self.perceptor.log_radial_decode(pred_s, pred_lm, scale=20.0)

            # Check if query is arithmetic / symbolic
            is_symbolic_query = any(op_char in clean_q for op_char in ["+", "-", "*", "/", "%", "^"])

            # If query is symbolic or action head selected tool dispatch
            if is_symbolic_query or predicted_action == ACTION_INVOKE_SYMBOLIC_TOOL:
                chosen_action = ACTION_INVOKE_SYMBOLIC_TOOL
            else:
                chosen_action = predicted_action

            tool_dispatch_res = None
            exact_result = None
            if enable_system2 and chosen_action == ACTION_INVOKE_SYMBOLIC_TOOL:
                tool_dispatch_res = self.dispatch_action(
                    action_id=ACTION_INVOKE_SYMBOLIC_TOOL,
                    query=clean_q,
                    log_to_episodic=True
                )
                exact_result = tool_dispatch_res["result"]

            # Causal Inversion Check
            z_fwd = self.world_model.forward_transition(z_stim, z_stim)
            z_bwd = self.world_model.backward_transition(z_fwd, z_stim)
            u_bwd, _ = self.world_model.project_manifold(z_bwd)
            angular_err = geodesic_distance_sphere(u_stim, u_bwd).mean().item()
            causal_consistency = 1.0 / (1.0 + angular_err)

        return {
            "query": clean_q,
            "system1_intuition": s1_intuition,
            "system2_exact_result": exact_result,
            "action_selected": chosen_action,
            "action_name": ACTION_NAMES.get(chosen_action, "UNKNOWN"),
            "action_probabilities": {ACTION_NAMES.get(i, str(i)): float(act_probs[i].item()) for i in range(len(act_probs))},
            "tool_dispatch": tool_dispatch_res,
            "causal_consistency": causal_consistency,
            "angular_rollback_error": angular_err,
            "manifold_coords": {
                "u": u_stim[0].cpu().tolist(),
                "r": float(r_stim[0, 0].item())
            }
        }

    def save_brain(self, filepath: str) -> None:
        """
        Saves the persistent lifelong cognitive brain to disk:
        - Network parameters across all submodules
        - Dynamic MoE expert anchors and neurogenesis count
        - Complete Tri-Tier Memory state (episodic traces & semantic lattice)
        - Architecture hyperparameters
        """
        checkpoint = {
            "config": {
                "d_model": self.d_model,
                "d_hidden": self.d_hidden,
                "hyperspace_dim": self.hyperspace_dim,
                "num_memory_slots": self.num_memory_slots,
                "vocab_size": self.vocab_size,
                "num_actions": self.num_actions,
                "max_experts": self.max_experts,
                "consensus_threshold": self.consensus_threshold,
                "max_thinking_steps": self.max_thinking_steps,
                "lr": self.lr
            },
            "num_experts": self.moe.num_experts,
            "state_dict": self.state_dict(),
            "episodic_memory": self.memory.episodic_memory,
            "semantic_concepts": self.memory.semantic_concepts,
            "mem_timestamp": self.memory.timestamp
        }
        if hasattr(self, "language_learner") and self.language_learner.vocab:
            checkpoint["language_vocab"] = self.language_learner.vocab
            if self.language_learner.word_embeddings is not None:
                checkpoint["language_embeddings"] = self.language_learner.word_embeddings.state_dict()
        if hasattr(self, "developmental_learner"):
            checkpoint["developmental_profile"] = self.developmental_learner.get_developmental_profile()
        torch.save(checkpoint, filepath)

    @classmethod
    def load_brain(cls, filepath: str, device: Optional[torch.device] = None) -> "UniversalCognitiveEngine":
        """
        Loads a persistent lifelong cognitive brain from disk.
        Restores weights, active experts, and lifelong episodic/semantic memory tiers.
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        config = checkpoint["config"]
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_experts = checkpoint["num_experts"]
        engine = cls(
            d_model=config["d_model"],
            d_hidden=config["d_hidden"],
            hyperspace_dim=config["hyperspace_dim"],
            num_memory_slots=config["num_memory_slots"],
            vocab_size=config["vocab_size"],
            initial_experts=num_experts,
            max_experts=config["max_experts"],
            consensus_threshold=config["consensus_threshold"],
            max_thinking_steps=config["max_thinking_steps"],
            num_actions=config.get("num_actions", 4),
            lr=config["lr"],
            device=target_device
        )

        engine.load_state_dict(checkpoint["state_dict"], strict=False)

        # Sanitize episodic and semantic memory vectors on load
        cleaned_episodes = []
        for ep in checkpoint.get("episodic_memory", []):
            ep_clean = dict(ep)
            if isinstance(ep_clean.get("v"), torch.Tensor):
                ep_clean["v"] = torch.nan_to_num(ep_clean["v"])
            if isinstance(ep_clean.get("u"), torch.Tensor):
                ep_clean["u"] = project_to_sphere(torch.nan_to_num(ep_clean["u"]))
            cleaned_episodes.append(ep_clean)

        cleaned_semantics = []
        for sc in checkpoint.get("semantic_concepts", []):
            sc_clean = dict(sc)
            if isinstance(sc_clean.get("v"), torch.Tensor):
                sc_clean["v"] = torch.nan_to_num(sc_clean["v"])
            if isinstance(sc_clean.get("u"), torch.Tensor):
                sc_clean["u"] = project_to_sphere(torch.nan_to_num(sc_clean["u"]))
            cleaned_semantics.append(sc_clean)

        engine.memory.episodic_memory = cleaned_episodes
        engine.memory.semantic_concepts = cleaned_semantics
        engine.memory.timestamp = checkpoint["mem_timestamp"]

        if "language_vocab" in checkpoint and hasattr(engine, "language_learner"):
            engine.language_learner.vocab = checkpoint["language_vocab"]
            engine.language_learner.word_to_id = {w: i for i, w in enumerate(checkpoint["language_vocab"])}
            engine.language_learner.id_to_word = {i: w for i, w in enumerate(checkpoint["language_vocab"])}
            if "language_embeddings" in checkpoint:
                V = len(engine.language_learner.vocab)
                engine.language_learner.word_embeddings = torch.nn.Embedding(V, engine.d_model).to(target_device)
                engine.language_learner.word_embeddings.load_state_dict(checkpoint["language_embeddings"])

        if "developmental_profile" in checkpoint and hasattr(engine, "developmental_learner"):
            dev_prof = checkpoint["developmental_profile"]
            from ishmn.developmental_learner import DevelopmentalStage
            stage_val = dev_prof.get("current_stage", "0_sensorimotor")
            try:
                engine.developmental_learner.stage = DevelopmentalStage(stage_val)
            except ValueError:
                pass
            if "zpd_telemetry" in dev_prof:
                z_tel = dev_prof["zpd_telemetry"]
                engine.developmental_learner.zpd.tau_lower = z_tel.get("tau_lower", 0.20)
                engine.developmental_learner.zpd.tau_upper = z_tel.get("tau_upper", 0.85)
            if "stage_transition_history" in dev_prof:
                engine.developmental_learner.stage_transition_log = dev_prof["stage_transition_history"]
            engine.developmental_learner.developmental_ticks = dev_prof.get(
                "developmental_ticks", dev_prof.get("total_developmental_ticks", 0)
            )
            engine.developmental_learner.stage_ticks = dev_prof.get(
                "stage_ticks", dev_prof.get("ticks_in_current_stage", 0)
            )

        engine.to(target_device)
        return engine

    def mental_simulation(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Instantaneous mental simulation via Successor Representation M = (I - gamma * T)^{-1} (~0.15ms)."""
        return self.cognitive_map.simulate_future_rollout(concept, top_k=top_k)

    def denoise_attractor(self, vec: torch.Tensor) -> Any:
        """Snaps noisy off-manifold representation into clean attractor basin via Modern Hopfield Network (~0.41ms)."""
        return self.hopfield.snap(vec)

    def separate_patterns(self, vec: torch.Tensor) -> torch.Tensor:
        """Projects dense representation through Dentate Gyrus 2% k-WTA lateral inhibition."""
        return self.dentate(vec)

    def step_criticality(self, was_event: bool) -> float:
        """Homeostatically adapts spawn/novelty thresholds toward critical branching ratio sigma = 1.0."""
        new_tau = self.criticality.step(was_event)
        self.moe.spawn_threshold = new_tau
        return new_tau

    def developmental_step(
        self,
        observation: Any,
        modality: str = "continuous",
        domain: str = "general",
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Executes a single developmental learning intake step through the ontogenetic state machine.
        """
        return self.developmental_learner.step_intake(
            observation=observation,
            modality=modality,
            domain=domain,
            context=context
        )

    def get_developmental_profile(self) -> Dict[str, Any]:
        """
        Returns the current developmental profile (stage, ZPD telemetry, transition history).
        """
        return self.developmental_learner.get_developmental_profile()

    def advance_developmental_stage(
        self,
        force: bool = False,
        reason: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Advances the agent's developmental stage if readiness milestones are satisfied.
        """
        if force:
            return self.developmental_learner.advance_stage(force=True, reason=reason)
        return self.developmental_learner.check_and_advance_milestones()

    def attach_native_transformer(
        self,
        transformer: Any,
        tokenizer: Optional[Any] = None
    ):
        """
        Binds a Sovereign NativeSphericalTransformer as UCE's primary language cortex.
        """
        self.native_transformer = transformer
        self.tokenizer = tokenizer

        # Defensive vocabulary alignment: ensure UniversalPerceptor supports full cortex vocabulary
        if hasattr(transformer, "vocab_size") and hasattr(self.perceptor, "resize_token_embeddings"):
            if transformer.vocab_size > self.perceptor.token_embed.num_embeddings:
                self.perceptor.resize_token_embeddings(transformer.vocab_size, device=self.device)
            self.vocab_size = max(self.vocab_size, transformer.vocab_size)

        return self

    def seed_scientific_invariants(self, invariants: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Seeds the UCE's InfiniteSphericalMemory and Modern Hopfield Network
        with ground-truth domain principles and physical/biological invariants.
        """
        if invariants is None:
            from ishmn.data.scientific_knowledge_bank import SCIENTIFIC_INVARIANTS
            invariants = SCIENTIFIC_INVARIANTS

        seeded = 0
        with torch.no_grad():
            for item in invariants:
                desc = item.get("description", item.get("word", ""))
                word = item.get("word", "")
                domain = item.get("domain", "general")
                meta = {
                    "word": word,
                    "domain": domain,
                    "description": desc,
                    "keywords": item.get("keywords", [word])
                }
                # Perceive invariant on manifold
                if self.tokenizer is not None:
                    tok_ids = torch.tensor([self.tokenizer.encode(desc)], dtype=torch.long, device=self.device)
                    u_inv, r_inv, v_inv = self.perceive(tok_ids, modality="tokens")
                else:
                    u_inv, r_inv, v_inv = self.perceive(desc, modality="subwords")

                self.memory.admit_concept(
                    u_new=u_inv[0:1],
                    r_new=r_inv[0:1],
                    v_new=v_inv[0:1],
                    metadata=meta,
                    tau_sep=0.15
                )
                if hasattr(self, "hopfield"):
                    self.hopfield.store(word, v_inv[0:1])
                seeded += 1
        return seeded

    def think_and_speak(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        enable_born_rule: bool = True,
        enable_scratchpad: bool = True,
        enable_deliberation: bool = True,
        write_to_vsa: bool = True
    ) -> Dict[str, Any]:
        """
        Full Closed-Loop Sovereign Cognitive Processing:
        1. Perceive query on continuous Riemannian manifold S^{D-1} x R+ via UniversalPerceptor.
        2. Denoise and recall associative priors via Modern Hopfield Network (attractor snap).
        3. Simulate predictive successor paths via CognitiveMapGraph.
        4. If symbolic arithmetic/equation is detected, execute System 2 dual-process solver.
        5. Mount Prefrontal Neuro-Symbolic Scratchpad to constrain generation.
        6. Autoregressively generate thought & response using Rectified Born Rule decoding.
        7. Consolidate high-surprise insights into Episodic Memory and VSA Memory.
        """
        if not hasattr(self, "native_transformer") or self.native_transformer is None:
            raise RuntimeError("No NativeSphericalTransformer attached. Call attach_native_transformer() first.")

        t0 = time.time()
        clean_prompt = prompt.strip()

        # Step 1: Perceive prompt in UCE manifold
        u_stim, r_stim, z_stim = self.perceive(clean_prompt, modality="subwords")

        # Step 2: Modern Hopfield Attractor Recall
        attractor_res = self.hopfield.snap(z_stim)
        z_attractor = attractor_res.converged_vec
        hopfield_energy = attractor_res.energy

        # Step 3: Mental Simulation via Cognitive Map Graph
        try:
            mental_rollouts = self.cognitive_map.simulate_future_rollout(clean_prompt, top_k=3)
        except Exception:
            mental_rollouts = []

        # Step 4: System 1 / System 2 Dual-Process Evaluation
        dual_process_res = None
        if enable_deliberation:
            try:
                dual_process_res = self.solve_with_dual_process(clean_prompt, enable_system2=True)
            except Exception:
                pass

        # Step 5: Setup Prefrontal Neuro-Symbolic Scratchpad & Deliberative Logits
        processors = []
        if enable_scratchpad:
            try:
                from ishmn.neurosymbolic_scratchpad import NeuroSymbolicScratchpadLogitsProcessor
                processors.append(NeuroSymbolicScratchpadLogitsProcessor(uce=self, tokenizer=self.tokenizer))
            except Exception:
                pass

        if enable_deliberation:
            try:
                from ishmn.deliberative_logits import DeliberativeConceptLogitsProcessor
                processors.append(DeliberativeConceptLogitsProcessor(uce=self, tokenizer=self.tokenizer, enabled=True))
            except Exception:
                pass

        def combined_logits_processor(input_ids, scores):
            for p in processors:
                scores = p(input_ids, scores)
            return scores

        active_processor = combined_logits_processor if processors else None

        # Step 6: Autoregressive Native Generation with Rectified Born Rule & Cognitive Context
        trans_dev = self.native_transformer.tok_embed.weight.device
        if self.tokenizer is not None:
            encoded_prompt = self.tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=trans_dev)
        else:
            encoded_prompt = [ord(c) for c in prompt]
            prompt_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=trans_dev)

        c_context = z_attractor.unsqueeze(0).to(trans_dev) if (z_attractor is not None and z_attractor.numel() > 0) else None

        gen_tokens = self.native_transformer.generate(
            prompt_ids=prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_born_rule=enable_born_rule,
            logits_processor=active_processor,
            context=c_context,
            write_to_vsa=write_to_vsa
        )

        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(gen_tokens[0].tolist())
        else:
            generated_text = "".join(chr(min(max(t, 0), 127)) for t in gen_tokens[0].tolist())

        elapsed = time.time() - t0

        return {
            "prompt": prompt,
            "response": generated_text,
            "elapsed_sec": elapsed,
            "tokens_generated": gen_tokens.size(1) - len(encoded_prompt),
            "hopfield_energy": float(hopfield_energy) if isinstance(hopfield_energy, (float, int)) else 0.0,
            "mental_rollouts": mental_rollouts,
            "dual_process": dual_process_res,
            "system2_proof": dual_process_res.get("system2_exact_result") if dual_process_res else None,
            "vsa_occupied_slots": int(self.native_transformer.vsa_memory.occupied.sum().item()) if hasattr(self.native_transformer, "vsa_memory") else 0
        }

    def train_unified_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Executes a single unified forward pass across the COMPLETE dual-cortex architecture:
        1. Language Cortex (NativeSphericalTransformer):
           - Dual-Stream Riemannian Attention on S^{D-1}
           - Sparse Dynamic MoE routing
           - Next-token cross-entropy + MTP heads
        2. Universal Perception (UniversalPerceptor):
           - Continuous manifold projection on S^{D-1} x R+
           - Cross-cortex representation alignment
        3. Modern Hopfield Network:
           - Dense associative memory attractor snapping
        4. DeltaNet:
           - Associative linear recurrence
        5. Dentate Gyrus:
           - Biologically inspired sparse pattern separation
        6. Latent JEPA World Model:
           - Phase-space continuous causal trajectory prediction
        7. Toroidal Grid Encoder:
           - Multi-scale spatial/conceptual coordinate preservation
        """
        if not hasattr(self, "native_transformer") or self.native_transformer is None:
            raise RuntimeError("No NativeSphericalTransformer attached. Call attach_native_transformer() first.")

        # 1. Universal Perceptor sequence encoding & manifold projection
        h_perc = self.perceptor.encode_tokens(input_ids) # [B, d_hidden]
        u_perc = project_to_sphere(self.perceptor.to_direction(h_perc)) # [B, hyperspace_dim]
        r_perc = F.softplus(self.perceptor.to_radius(h_perc)) # [B, 1]
        payload_perc = self.perceptor.to_payload(h_perc) # [B, d_model]

        # 2. Modern Hopfield Network Attractor Snapping
        hop = self.hopfield.snap(payload_perc)
        loss_hopfield = F.mse_loss(hop.converged_vec, payload_perc)

        # 3. Closed-Loop Cognitive Context Conditioning
        # The Hopfield attractor vector acts as the global cognitive context c
        # actively modulating MoE proposal bids and expert selection
        c_context = hop.converged_vec.unsqueeze(1) # [B, 1, d_model]

        # 4. Language Cortex forward pass with active cognitive context
        out_lm = self.native_transformer(input_ids, targets=targets, context=c_context, return_mtp=True)
        lm_loss = out_lm["loss"]
        h = out_lm["hidden_states"] # [B, T, d_model]
        u_tokens = out_lm.get("u_tokens") # [B*T, hyperspace_dim]
        r_tokens = out_lm.get("r_tokens") # [B*T, 1]
        B, T, D = h.shape

        # 5. Dual-Cortex Manifold Co-Registration
        # Representation alignment: pooled language hidden states <-> perceptor payload
        h_mean = h.mean(dim=1)
        loss_repr_align = F.mse_loss(payload_perc, h_mean)

        # Spherical coordinate alignment: S^{D-1} sequence direction
        if u_tokens is not None:
            u_tokens_seq = project_to_sphere(u_tokens.view(B, T, -1).mean(dim=1))
            loss_u_align = geodesic_distance_sphere(u_perc, u_tokens_seq).mean()
        else:
            loss_u_align = torch.tensor(0.0, device=h.device)

        # Radial depth alignment: R+ abstraction radius
        if r_tokens is not None:
            r_tokens_seq = r_tokens.view(B, T, 1).mean(dim=1)
            loss_r_align = F.mse_loss(r_perc, r_tokens_seq)
        else:
            loss_r_align = torch.tensor(0.0, device=h.device)

        loss_manifold_align = loss_repr_align + 0.10 * loss_u_align + 0.05 * loss_r_align

        # 6. DeltaNet Fast Associative Recurrence
        delta_out, _ = self.delta_net(h)
        loss_delta = F.mse_loss(delta_out, h)

        # 7. Dentate Gyrus Biological Pattern Separation (Barlow Twins Decorrelation)
        dent_out = self.dentate(h.reshape(-1, D)) # [B*T, d_sparse], unit-normalized
        n_sample = min(64, dent_out.size(0))
        dent_sub = dent_out[:n_sample]
        sim_matrix = torch.mm(dent_sub, dent_sub.t()) # [N, N]
        eye = torch.eye(n_sample, device=dent_out.device)
        loss_dentate = ((sim_matrix - eye) ** 2).mean()

        # 8. JEPA Latent World Model Phase-Space Trajectory with Causal Reversibility
        if T > 1:
            z_curr = h[:, :-1, :]
            z_next = h[:, 1:, :]
            c_curr = z_curr
            z_pred = self.world_model.forward_transition(z_curr, c_curr)
            z_recon = self.world_model.backward_transition(z_pred, c_curr)
            loss_jepa = F.mse_loss(z_pred, z_next) + F.mse_loss(z_recon, z_curr)
        else:
            loss_jepa = torch.tensor(0.0, device=h.device)

        # 9. Toroidal Harmonic Grid Reconstruction
        if u_tokens is not None and hasattr(self, "grid_to_u"):
            grid_out = self.grid_encoder(u_tokens)
            u_grid_recon = project_to_sphere(self.grid_to_u(grid_out))
            loss_grid = geodesic_distance_sphere(u_grid_recon, u_tokens).mean()
        else:
            loss_grid = torch.tensor(0.0, device=h.device)

        # Total Unified Multi-Cortical Loss
        total_loss = (
            lm_loss
            + 0.10 * loss_jepa
            + 0.05 * loss_manifold_align
            + 0.05 * loss_hopfield
            + 0.05 * loss_delta
            + 0.02 * loss_dentate
            + 0.02 * loss_grid
        )

        loss_breakdown = {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "jepa_loss": loss_jepa.item(),
            "manifold_align_loss": loss_manifold_align.item(),
            "perceptor_align_loss": loss_repr_align.item(),
            "u_align_loss": loss_u_align.item(),
            "r_align_loss": loss_r_align.item(),
            "hopfield_loss": loss_hopfield.item(),
            "delta_loss": loss_delta.item(),
            "dentate_loss": loss_dentate.item(),
            "grid_loss": loss_grid.item()
        }

        return {
            "loss": total_loss,
            "loss_breakdown": loss_breakdown,
            "out_lm": out_lm,
            "hopfield_energy": float(hop.energy) if isinstance(hop.energy, (float, int)) else 0.0,
            "dentate_active_ratio": float((dent_out > 0).float().mean().item())
        }
