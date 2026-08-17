"""
Canonical Training Entry Point for Universal Substrait Dynamic MoE.
Usage:
    python train.py --steps 3000 --batch-size 4 --accum-steps 3 --lr 6e-4
"""

import argparse
from train_scaled_production_engine import run_scaled_pretraining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Substrait Canonical Trainer")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per GPU")
    parser.add_argument("--accum-steps", type=int, default=3, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--out-dir", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()

    run_scaled_pretraining(
        total_steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size * args.accum_steps,
        max_lr=args.lr,
        out_dir=args.out_dir
    )
