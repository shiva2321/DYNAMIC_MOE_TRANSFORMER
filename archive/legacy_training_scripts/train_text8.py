"""
Training & Evaluation of Hyperspace 2.0 on Real-World Wikipedia text8 Dataset.
Evaluates:
1. Bits-Per-Character (BPC) & Cross-Entropy Loss
2. Dynamic Expert Spawning across Wikipedia Topic Transitions
3. Self-Organized Criticality Stability (sigma = 1.0) on Natural Language
4. Comparison against Dense Transformer Baseline
5. Real-time Autoregressive Generation with Token-Level Attribution
"""

import sys
import time
import math
import argparse
import os
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from model.baselines import DenseTransformerLM
from model.attribution import AttributionTracer

class Text8Dataset:
    """Loads and streams character chunks from text8 dataset."""
    def __init__(self, filepath: str = "e:/universal_substrait/data/text8.txt", seq_len: int = 128, batch_size: int = 8):
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            self.raw_text = f.read()
            
        # Create character vocabulary (27 characters: a-z and space)
        self.chars = sorted(list(set(self.raw_text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Pre-tokenize full buffer to byte tensor
        self.encoded_data = torch.tensor([self.char_to_idx[c] for c in self.raw_text if c in self.char_to_idx], dtype=torch.long)
        
        # Split 90% train, 10% test
        n_train = int(len(self.encoded_data) * 0.9)
        self.train_data = self.encoded_data[:n_train]
        self.test_data = self.encoded_data[n_train:]
        
        print(f"Loaded text8 dataset: {len(self.encoded_data):,} characters (Vocab size: {self.vocab_size})")
        print(f"Train split: {len(self.train_data):,} chars | Test split: {len(self.test_data):,} chars")

    def encode(self, text: str) -> list:
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, tokens: list) -> str:
        return "".join([self.idx_to_char.get(t, " ") for t in tokens])

    def get_batch(self, split: str = "train") -> tuple:
        data = self.train_data if split == "train" else self.test_data
        max_start = len(data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (self.batch_size,))
        
        batch_inputs = torch.stack([data[s : s + self.seq_len] for s in starts])
        batch_targets = torch.stack([data[s + 1 : s + self.seq_len + 1] for s in starts])
        return batch_inputs, batch_targets

def sync_optimizer(optimizer, model):
    existing = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing.add(p)
    new_params = [p for p in model.parameters() if p not in existing and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300, help="Training steps on text8")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--d_hyper", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("================================================================================")
    print("  [EXPERIMENT] TRAINING HYPERSPACE 2.0 ON REAL-WORLD WIKIPEDIA (text8)")
    print("================================================================================")
    print(f"Device: {device} | Total Steps: {args.steps} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}\n")

    dataset = Text8Dataset(seq_len=args.seq_len, batch_size=args.batch_size)

    # 1. Initialize Competitor Models
    print("\n-> Initializing Dense Transformer Baseline...")
    model_dense = DenseTransformerLM(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff * 2,
        max_seq_len=args.seq_len + 32
    ).to(device)

    print("-> Initializing Hyperspace 2.0 Neuro-Cognitive Model...")
    model_hyperspace2 = HyperTransformerLM(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        d_hyper=args.d_hyper,
        top_k=2,
        spawn_threshold=0.28,
        max_experts=16,
        max_seq_len=args.seq_len + 32
    ).to(device)

    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=args.lr)
    opt_hyper2 = torch.optim.AdamW(model_hyperspace2.parameters(), lr=args.lr)

    print("\n--- Beginning Training on text8 Wikipedia Stream ---")
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        inputs, targets = dataset.get_batch("train")
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Train Dense
        model_dense.train()
        opt_dense.zero_grad()
        _, loss_dense, _ = model_dense(inputs, targets=targets)
        loss_dense.backward()
        torch.nn.utils.clip_grad_norm_(model_dense.parameters(), 1.0)
        opt_dense.step()

        # 2. Train Hyperspace 2.0
        model_hyperspace2.train()
        opt_hyper2.zero_grad()
        prev_exp = model_hyperspace2.blocks[0].hyper_moe.num_experts
        _, loss_hyper2, telem_h2 = model_hyperspace2(inputs, targets=targets, allow_spawning=True)
        
        # Dynamic Spawning check
        curr_exp = model_hyperspace2.blocks[0].hyper_moe.num_experts
        if curr_exp != prev_exp:
            sync_optimizer(opt_hyper2, model_hyperspace2)
            print(f"[SPAWN @ Step {step:3d}] text8 Topic Shift Detected | Hyperspace 2.0 Experts: {curr_exp}")

        loss_hyper2.backward()
        torch.nn.utils.clip_grad_norm_(model_hyperspace2.parameters(), 1.0)
        opt_hyper2.step()

        if step % 50 == 0 or step == 1 or step == args.steps:
            # Compute Bits Per Character (BPC): BPC = loss / ln(2)
            bpc_dense = loss_dense.item() / math.log(2.0)
            bpc_hyper = loss_hyper2.item() / math.log(2.0)
            sigma = telem_h2[0]["branching_ratio"] if telem_h2 else 1.0
            temp = telem_h2[0]["routing_temperature"] if telem_h2 else 10.0
            exp_counts = "/".join(str(b.hyper_moe.num_experts) for b in model_hyperspace2.blocks)
            print(f"Step {step:4d}/{args.steps} | Dense BPC: {bpc_dense:.3f} | Hyper 2.0 BPC: {bpc_hyper:.3f} (Loss: {loss_hyper2.item():.3f}) | Experts: [{exp_counts}] | Sigma: {sigma:.3f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({args.steps / total_time:.1f} steps/sec)\n")

    # ----------------------------------------------------------------------
    # TEST SPLIT EVALUATION: Out-of-Sample Test Set BPC
    # ----------------------------------------------------------------------
    print("================================================================================")
    print("  [EVALUATION] OUT-OF-SAMPLE TEST SET BITS-PER-CHARACTER (BPC)")
    print("================================================================================")
    
    model_dense.eval()
    model_hyperspace2.eval()
    
    test_loss_dense, test_loss_hyper = 0.0, 0.0
    eval_batches = 20
    
    with torch.no_grad():
        for _ in range(eval_batches):
            inp_t, tgt_t = dataset.get_batch("test")
            inp_t, tgt_t = inp_t.to(device), tgt_t.to(device)
            
            _, l_d, _ = model_dense(inp_t, targets=tgt_t)
            _, l_h, _ = model_hyperspace2(inp_t, targets=tgt_t)
            
            test_loss_dense += l_d.item()
            test_loss_hyper += l_h.item()

    mean_l_dense = test_loss_dense / eval_batches
    mean_l_hyper = test_loss_hyper / eval_batches
    
    test_bpc_dense = mean_l_dense / math.log(2.0)
    test_bpc_hyper = mean_l_hyper / math.log(2.0)
    
    print(f"Dense Transformer Test Loss:       {mean_l_dense:.4f} -> Test BPC: {test_bpc_dense:.4f}")
    print(f"Hyperspace 2.0 Test Loss:          {mean_l_hyper:.4f} -> Test BPC: {test_bpc_hyper:.4f}")
    print(f"BPC Advantage of Hyperspace 2.0:   {test_bpc_dense - test_bpc_hyper:+.4f} BPC")

    # ----------------------------------------------------------------------
    # AUTOREGRESSIVE GENERATION WITH ATTRIBUTION ON REAL WIKIPEDIA
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [GENERATION] AUTOREGRESSIVE GENERATION & TOKEN ATTRIBUTION (text8)")
    print("================================================================================")
    
    sample_prompts = [
        "the history of philosophy begins with ",
        "anarchism is a political philosophy which ",
        "in modern mathematics the concept of "
    ]
    
    for prompt_text in sample_prompts:
        prompt_tokens = dataset.encode(prompt_text)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        trace = AttributionTracer.trace_generation_attribution(
            model_hyperspace2, prompt_tensor, dataset, max_new_tokens=48, temperature=0.7
        )
        AttributionTracer.print_colored_trace(trace, title=f"text8 Prompt: '{prompt_text}'")

if __name__ == "__main__":
    main()
