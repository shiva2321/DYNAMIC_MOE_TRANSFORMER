import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES
from ishmn.evaluation_harness import evaluate_mmoe_comprehensive

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'artifacts/checkpoints/native_moe_acc80_plus.pt'
    print(f"Loading golden checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    step = ckpt.get('step')
    comp_acc = ckpt.get('composite_accuracy')
    saved_res = ckpt.get('eval_results', {}).get('domain_accuracies', {})
    print(f"Checkpoint Milestone Step: {step}")
    print(f"Recorded Composite Accuracy: {comp_acc:.2f}%")
    print("Recorded Domain Breakdown:")
    for domain, acc in saved_res.items():
        print(f"  - {domain:15s}: {acc:.2f}%")

    cfg = MODEL_PROFILES['nano']
    tokenizer = SovereignTokenizer('gpt2')

    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg['d_model'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        hyperspace_dim=cfg['hyperspace_dim'],
        d_ff=cfg['d_ff'],
        max_seq_len=64 + 32,
        mtp_future_tokens=cfg['mtp_future_tokens'],
        vsa_capacity=cfg['vsa_capacity'],
        num_experts=cfg['num_experts'],
        top_k=cfg['top_k'],
        enable_shared_expert=cfg['enable_shared_expert'],
        enable_aux_loss_free_balancing=cfg['enable_aux_loss_free_balancing'],
        enable_vsa_memory=True,
        enable_world_model=True,
        device=device
    ).to(device)

    uce = UniversalCognitiveEngine(
        d_model=cfg['d_model'],
        d_hidden=cfg['d_ff'],
        hyperspace_dim=cfg['hyperspace_dim'],
        num_memory_slots=32,
        vocab_size=tokenizer.vocab_size,
        initial_experts=cfg['num_experts'],
        device=device
    ).to(device)
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    def test_checkpoint(path):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        ckpt = torch.load(path, map_location=device, weights_only=False)
        step = ckpt.get('step')
        print(f"\nEvaluating: {path} (Step {step})")
        if 'uce_state_dict' in ckpt:
            uce.load_state_dict(ckpt['uce_state_dict'])
        elif 'extra_meta' in ckpt and 'uce_state_dict' in ckpt['extra_meta']:
            uce.load_state_dict(ckpt['extra_meta']['uce_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])

        model.eval()
        uce.eval()
        eval_res = evaluate_mmoe_comprehensive(model=model, uce=uce, tokenizer=tokenizer, device=device)
        domains = eval_res['domain_accuracies']
        print("=" * 60)
        print(f" 1. Mathematical & Numeric Reasoning  : {domains['math']:.2f}%")
        print(f" 2. Causal & Physical Sensemaking     : {domains['science']:.2f}%")
        print(f" 3. Algorithmic Logic & Syntax        : {domains['code']:.2f}%")
        print(f" 4. Held-Out Curriculum Validation    : {domains['curriculum']:.2f}%")
        print("-" * 60)
        print(f" >>> COMPOSITE EVALUATION ACCURACY    : {eval_res['composite_accuracy']:.2f}%")
        print("=" * 60)
        domain_map = {
            'code': 'algorithmic_code',
            'curriculum': 'curriculum_validation',
            'science': 'causal_sensemaking',
            'math': 'math_reasoning'
        }
        for dom in ['math', 'science', 'code', 'curriculum']:
            dom_key = domain_map[dom]
            items = eval_res['domains'][dom_key]['details']
            print(f"\n[{dom.upper()} Breakout]")
            for it in items:
                if dom == 'science':
                    ok = "PASS" if it.get("discriminated") else "FAIL"
                    premise = it.get("premise", "")[:35]
                    fact = it.get("fact", "")[:25]
                    contra = it.get("contradiction", "")[:25]
                    print(f"  [{ok}] Premise: '{premise}' | Fact: '{fact}' vs '{contra}'")
                elif dom == 'math':
                    ok = "PASS" if it.get("matched") else "FAIL"
                    tgt = it.get("target")
                    pred = it.get("pred")
                    prompt = it.get("prompt", "")[:35]
                    print(f"  [{ok}] Target: '{tgt}' | Pred: '{pred}' | Prompt: '{prompt}'")
                else:
                    ok = "PASS" if it.get("is_correct") else "FAIL"
                    tgt = it.get("target")
                    best = it.get("best", "")
                    prompt = it.get("prompt", "")[:35]
                    print(f"  [{ok}] Target: '{tgt}' | Picked: '{best}' | Prompt: '{prompt}'")
        return eval_res

    res_96 = test_checkpoint('artifacts/checkpoints/native_moe_acc96_plus.pt')
    res_golden = test_checkpoint('artifacts/checkpoints/native_moe_acc80_plus.pt')

if __name__ == '__main__':
    main()
