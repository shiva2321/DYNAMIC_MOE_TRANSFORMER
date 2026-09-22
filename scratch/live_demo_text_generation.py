"""
Live demonstration of GroundedHybridEngine text generation with Riemannian manifold deliberation at Layer 14.
"""
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import GroundedHybridEngine

BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("LIVE GROUNDED HYBRID ENGINE DEMONSTRATION (QWEN2.5-1.5B + UCE MANIFOLD)")
    print("=" * 80)
    
    t0 = time.time()
    engine = GroundedHybridEngine(brain_path=BRAIN_PATH, device=DEVICE)
    print(f"[+] Loaded engine in {time.time() - t0:.2f}s.")
    
    prompt = "In physics, what happens when light passes from air into water?"
    print(f"\n[PROMPT]: \"{prompt}\"\n")
    
    print("-" * 80)
    print("1. GENERATING WITH LAYER 14 RIEMANNIAN DELIBERATION (SANDWICH ACTIVE):")
    print("-" * 80)
    
    res = engine.generate_grounded(prompt, max_new_tokens=60, enable_sandwich=True)
    
    print("\n[GENERATED TEXT]:")
    print(res["text"])
    print("\n[COGNITIVE TELEMETRY]:")
    print(f"  - Generation Time          : {res['elapsed_sec']:.2f}s")
    print(f"  - Tokens Generated         : {res['tokens_generated']}")
    print(f"  - System 1 (Intuitive)     : {res['system1_tokens']} tokens ({res['system1_tokens']/res['tokens_generated']*100:.1f}%)")
    print(f"  - System 2 (Deliberated)   : {res['system2_tokens']} tokens ({res['system2_tokens']/res['tokens_generated']*100:.1f}%)")
    print(f"  - Average Thinking Ticks   : {res['avg_thinking_ticks']:.2f} ticks/token")
    print(f"  - Mean Causal Error (W_bwd): {res['mean_causal_error']:.4f}")
    print(f"  - Mean Manifold Surprise   : {res['mean_surprise']:.4f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
