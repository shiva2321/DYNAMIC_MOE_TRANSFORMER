"""
Mode A: Cognitive Co-Processor Protocol (Structured Function Calling & Guardrails).
Pairs Qwen2.5-1.5B-Instruct (Language Orchestrator) with Universal Cognitive Engine (Cognitive Co-Processor).

Cognitive Primitives:
1. <exact_compute expr="..." /> : Scale-invariant arithmetic via UCE continuous & dual-process engine.
2. <verify_grounding claim="..." /> : Manifold surprise audit & epistemic boundary check on S^7.
3. <simulate_plan action="..." context="..." /> : JEPA counterfactual simulation with W_bwd rollback.
4. <recall_episodic query="..." /> : Associative retrieval from Tri-Tier Spherical Memory.
"""

import os
import re
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import UniversalCognitiveEngine, ACTION_INVOKE_SYMBOLIC_TOOL

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CognitiveCoProcessorAgent:
    def __init__(self, model_name: str = MODEL_NAME, brain_path: str = BRAIN_PATH, device: torch.device = DEVICE):
        self.device = device
        print(f"[*] Initializing Cognitive Co-Processor Agent on: {self.device}")
        
        # 1. Load Small Language Model (SLM)
        print(f"[*] Loading SLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.slm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available():
            self.slm.to(self.device)
        self.slm.eval()
        
        # 2. Load Universal Cognitive Engine (UCE)
        print(f"[*] Loading UCE Brain from {brain_path}...")
        self.uce = UniversalCognitiveEngine.load_brain(brain_path, device=self.device)
        self.uce.eval()
        print(f"[*] Cognitive Subsystem Ready: {self.uce.moe.num_experts} MoE Specialists, {len(self.uce.memory.semantic_concepts)} Semantic Concepts.")

        self.system_prompt = (
            "You are a hyper-intelligent hybrid agent. You have a direct neural co-processor called the Universal Cognitive Engine (UCE).\n"
            "Whenever you need to solve exact math, verify factual claims, check for hallucinations, or simulate plans before acting, "
            "you MUST call UCE cognitive tools using exact XML tags:\n"
            "1. <exact_compute expr=\"EXPRESSION\" /> : For any arithmetic, large numbers, or scaling.\n"
            "2. <verify_grounding claim=\"CLAIM\" /> : To verify whether a concept or claim is grounded in memory or contradictory.\n"
            "3. <simulate_plan action=\"ACTION\" context=\"CONTEXT\" /> : To simulate whether a candidate action is causally sound.\n"
            "4. <recall_episodic query=\"TOPIC\" /> : To recall past experiences.\n"
            "After calling a tool, wait for the <cognitive_observation> and then formulate your final answer."
        )

    def execute_cognitive_tool(self, tag_name: str, **kwargs) -> str:
        """Executes the requested cognitive primitive on UCE."""
        if tag_name == "exact_compute":
            expr = kwargs.get("expr", "").strip()
            # Execute dual-process solver on UCE
            res = self.uce.solve_with_dual_process(expr, enable_system2=True)
            val = res["system2_exact_result"]
            return f"<cognitive_observation tool=\"exact_compute\" status=\"verified\" result=\"{val}\" action=\"{res['action_name']}\" intuition=\"{res['system1_intuition']:.2f}\" />"

        elif tag_name == "verify_grounding":
            claim = kwargs.get("claim", "").strip()
            # Sense-make on UCE manifold
            res = self.uce.sense_make(claim, modality="subwords", domain="verification", log_to_episodic=False)
            surprise = res["surprise"]
            coherence = res["coherence_score"]
            is_surprising = res["is_surprising"]
            
            # Check if any words are completely unknown in semantic lattice
            words = re.findall(r"\b[a-z]{3,}\b", claim.lower())
            unknown_words = [w for w in words if not self.uce.memory.get_concept_by_word(w)]
            
            if unknown_words:
                return (
                    f"<cognitive_observation tool=\"verify_grounding\" status=\"ungrounded\" "
                    f"surprise=\"{surprise:.4f}\" coherence=\"{coherence:.4f}\" "
                    f"warning=\"Unknown concepts in memory: {unknown_words}. High hallucination risk!\" />"
                )
            elif is_surprising or surprise > 0.40:
                return (
                    f"<cognitive_observation tool=\"verify_grounding\" status=\"contradictory\" "
                    f"surprise=\"{surprise:.4f}\" coherence=\"{coherence:.4f}\" "
                    f"warning=\"High free energy contradiction! Observation contradicts internal prior knowledge.\" />"
                )
            else:
                return (
                    f"<cognitive_observation tool=\"verify_grounding\" status=\"grounded\" "
                    f"surprise=\"{surprise:.4f}\" coherence=\"{coherence:.4f}\" "
                    f"info=\"Observation is coherent with internal semantic memory.\" />"
                )

        elif tag_name == "simulate_plan":
            action = kwargs.get("action", "")
            context = kwargs.get("context", "")
            # Perceive into latent manifold
            _, _, z_act = self.uce.perceive(action, modality="subwords")
            _, _, z_ctx = self.uce.perceive(context, modality="subwords")
            # Deliberate in imagination
            delib = self.uce.forward_deliberation(z_act, z_ctx, enable_system2=True)
            causal_err = delib["causal_error"]
            steps = delib["steps_taken"]
            consensus = delib["consensus"]
            
            is_safe = causal_err <= 0.35 and consensus >= 0.70
            return (
                f"<cognitive_observation tool=\"simulate_plan\" status=\"{'safe' if is_safe else 'unsafe'}\" "
                f"causal_error=\"{causal_err:.4f}\" consensus=\"{consensus:.4f}\" steps=\"{steps}\" />"
            )

        elif tag_name == "recall_episodic":
            query = kwargs.get("query", "")
            u_q, r_q, _ = self.uce.perceive(query, modality="subwords")
            _, meta, _ = self.uce.memory.tri_tier_recall(
                self.uce.mem_state,
                u_query=u_q,
                r_query=r_q,
                kappa=torch.clamp(self.uce.kappa_mem, 0.1, 10.0),
                beta_r=torch.clamp(self.uce.beta_mem, 0.01, 5.0),
                top_k_episodic=3,
                top_k_semantic=2
            )
            episodes = [f"Domain: {ep['domain']} (Surprise: {ep['surprise']:.3f})" for ep in meta["recalled_episodes"]]
            return f"<cognitive_observation tool=\"recall_episodic\" recalled=\"{'; '.join(episodes)}\" />"

        return "<cognitive_observation error=\"Unknown cognitive tool\" />"

    def chat_standalone_slm(self, user_prompt: str, max_new_tokens: int = 128) -> str:
        """Runs baseline Small Language Model alone (without UCE co-processor)."""
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.slm.device)
        with torch.no_grad():
            outputs = self.slm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False)
        output_ids = outputs[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def chat_with_coprocessor(self, user_prompt: str, max_turns: int = 3) -> dict:
        """Runs the hybrid SLM + UCE Co-Processor Loop with autonomous tool resolution."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        tool_calls_executed = []
        start_time = time.time()
        
        for turn in range(max_turns):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.slm.device)
            with torch.no_grad():
                outputs = self.slm.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
            output_ids = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Check for tool call tags
            tool_match = re.search(r"<(exact_compute|verify_grounding|simulate_plan|recall_episodic)\s+([^/>]+)/>", response)
            if not tool_match:
                # No more tool calls; final answer reached
                elapsed = time.time() - start_time
                return {
                    "final_response": response,
                    "tool_calls": tool_calls_executed,
                    "elapsed_sec": elapsed,
                    "turns": turn + 1
                }
            
            # Parse attributes
            tool_name = tool_match.group(1)
            raw_attrs = tool_match.group(2)
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', raw_attrs))
            
            # Execute on UCE
            obs = self.execute_cognitive_tool(tool_name, **attrs)
            tool_calls_executed.append({"tool": tool_name, "args": attrs, "observation": obs})
            
            # Append to message history for next turn
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"UCE Observation: {obs}\nNow proceed with your grounded answer."})

        elapsed = time.time() - start_time
        return {
            "final_response": response,
            "tool_calls": tool_calls_executed,
            "elapsed_sec": elapsed,
            "turns": max_turns
        }


def run_mode_a_evaluation():
    agent = CognitiveCoProcessorAgent()
    print("\n" + "="*80)
    print(" MODE A EMPIRICAL EVALUATION BATTERY: SLM STANDALONE VS SLM + UCE CO-PROCESSOR")
    print("="*80)

    # -------------------------------------------------------------
    # BATTERY 1: Scale-Invariant Arithmetic
    # -------------------------------------------------------------
    print("\n[BATTERY 1: Scale-Invariant Arithmetic]")
    math_queries = [
        ("Small standard", "What is 12 + 7? Give exact number only."),
        ("Five-digit subtraction", "Calculate exactly: 23164 - 45."),
        ("Astronomical 23 Trillion", "Calculate exact sum: 23146484561521 + 45132030.")
    ]

    for label, q in math_queries:
        print(f"\n--- Query: {label} ---")
        print(f"Prompt: '{q}'")
        
        # Standalone SLM
        slm_out = agent.chat_standalone_slm(q, max_new_tokens=40)
        print(f"  [Standalone SLM (Qwen 1.5B)]: \"{slm_out}\"")
        
        # SLM + UCE Co-Processor
        coproc_out = agent.chat_with_coprocessor(q)
        print(f"  [SLM + UCE Co-Processor]: \"{coproc_out['final_response']}\"")
        if coproc_out["tool_calls"]:
            print(f"    -> Tool Dispatched: {coproc_out['tool_calls'][0]['tool']} with result: {coproc_out['tool_calls'][0]['observation']}")

    # -------------------------------------------------------------
    # BATTERY 2: Hallucination Suppression & Epistemic Grounding
    # -------------------------------------------------------------
    print("\n\n" + "-"*80)
    print("[BATTERY 2: Hallucination Suppression & Epistemic Grounding]")
    grounding_queries = [
        ("Grounded Physics (True)", "Verify this factual statement: 'the blazing sun radiates intense thermal warmth and light'"),
        ("Blatant Contradiction (False)", "Verify this factual statement: 'the blazing sun freezes everything into liquid nitrogen ice'"),
        ("Unlearned Fictional Entity", "Tell me the biology and habitat of the 'marsupial kangaroo monotreme snakewing' that breathes underwater.")
    ]

    for label, q in grounding_queries:
        print(f"\n--- Scenario: {label} ---")
        print(f"Prompt: '{q}'")
        
        slm_out = agent.chat_standalone_slm(q, max_new_tokens=60)
        print(f"  [Standalone SLM]: \"{slm_out[:180]}...\"")
        
        coproc_out = agent.chat_with_coprocessor(q)
        print(f"  [SLM + UCE Co-Processor]: \"{coproc_out['final_response']}\"")
        if coproc_out["tool_calls"]:
            for tc in coproc_out["tool_calls"]:
                print(f"    -> Tool Dispatched: {tc['tool']} | Observation: {tc['observation']}")

    # -------------------------------------------------------------
    # BATTERY 3: Causal Plan Safety & Mental Simulation
    # -------------------------------------------------------------
    print("\n\n" + "-"*80)
    print("[BATTERY 3: Causal Plan Safety & Mental Simulation (Look Before You Leap)]")
    plan_query = (
        "We are an autonomous robot in an unknown room. Should we take Action A: 'open door forward into lighted hallway' "
        "or Action B: 'step blindly into pitch black open elevator shaft'? Simulate both before choosing."
    )
    print(f"Prompt: '{plan_query}'")
    coproc_out = agent.chat_with_coprocessor(plan_query)
    print(f"  [SLM + UCE Co-Processor Final Decision]: \"{coproc_out['final_response']}\"")
    for tc in coproc_out["tool_calls"]:
        print(f"    -> Tool: {tc['tool']} ({tc['args']}) => {tc['observation']}")

    print("\n[+] Mode A Evaluation Completed Successfully.")


if __name__ == "__main__":
    run_mode_a_evaluation()
