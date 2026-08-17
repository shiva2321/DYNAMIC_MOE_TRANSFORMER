"""
Multi-Domain Dataset Downloader & Inspector for Hyperspace 2.0.
Downloads, caches, and verifies diverse datasets across 7 domains:
1. Python Code (algorithms, systems, asyncio)
2. Mathematics (derivations, quantum mechanics, algebra)
3. Encyclopedic Prose (history, world knowledge)
4. Conversational Dialogue (multi-turn assistant chat)
5. Literature (creative sci-fi & narrative stories)
6. Biomedical Sciences (cellular biology, pathology, pharmacology)
7. Structured Cloud & JSON Telemetry (Kubernetes, AWS IAM, Prometheus)
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Optional, List, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.dataset_hub import MultiDomainDatasetHub, DOMAIN_METADATA, DOMAIN_LIST, CACHE_DIR

def try_download_hf_dataset(domain_key: str, max_samples: int = 500) -> Optional[str]:
    """
    Attempts to download real samples from Hugging Face datasets in streaming mode.
    Returns aggregated text if successful, None otherwise.
    """
    try:
        from datasets import load_dataset
        
        texts = []
        if domain_key == "code":
            # Code dataset
            ds = load_dataset("codeparrot/github-code-clean", split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_samples:
                    break
                code_text = row.get("code", "")
                if 100 < len(code_text) < 4000:
                    texts.append(code_text)
                    
        elif domain_key == "math":
            # GSM8K / Math reasoning
            ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_samples:
                    break
                q, a = row.get("question", ""), row.get("answer", "")
                texts.append(f"Problem: {q}\nSolution: {a}\n")
                
        elif domain_key == "encyclopedia":
            # WikiText-2
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_samples:
                    break
                t = row.get("text", "").strip()
                if len(t) > 80:
                    texts.append(t)
                    
        elif domain_key == "dialogue":
            # DailyDialog
            ds = load_dataset("daily_dialog", split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_samples:
                    break
                dialogue_turns = row.get("dialog", [])
                formatted = []
                for turn_idx, turn in enumerate(dialogue_turns):
                    speaker = "User" if turn_idx % 2 == 0 else "Assistant"
                    formatted.append(f"{speaker}: {turn.strip()}")
                texts.append("\n".join(formatted) + "\n")
                
        elif domain_key == "literature":
            # TinyStories / Literature
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_samples:
                    break
                story = row.get("text", "").strip()
                if len(story) > 100:
                    texts.append(story)
                    
        if texts:
            return "\n\n".join(texts)
    except Exception as e:
        print(f"  [HF Note] Could not stream '{domain_key}' directly from HF: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Download and cache multi-domain datasets for Hyperspace 2.0")
    parser.add_argument("--fetch_hf", action="store_true", help="Attempt streaming download from Hugging Face Hub")
    parser.add_argument("--max_samples", type=int, default=150, help="Max samples per domain if downloading from HF")
    args = parser.parse_args()

    print("================================================================================")
    print("  [DATASET HUB] MULTI-DOMAIN INGESTION & CACHE MANAGER")
    print("================================================================================")
    print(f"Target Cache Directory: {CACHE_DIR}")
    print(f"Total Available Domains: {len(DOMAIN_LIST)}\n")

    if args.fetch_hf:
        print("-> Attempting online fetch for missing/enhanced datasets from Hugging Face...")
        for domain_key in DOMAIN_LIST:
            print(f"  * Fetching domain [{domain_key}] ({DOMAIN_METADATA[domain_key]['name']})...")
            text_data = try_download_hf_dataset(domain_key, max_samples=args.max_samples)
            if text_data:
                cache_file = os.path.join(CACHE_DIR, f"{domain_key}.txt")
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text_data)
                print(f"    [SAVED] {len(text_data):,} characters cached for '{domain_key}'.")
            else:
                print(f"    [INFO] Using verified high-density corpus for '{domain_key}'.")

    # Initialize Hub and inspect loaded stats
    hub = MultiDomainDatasetHub(seq_len=128, batch_size=8, use_bpe=True)
    domains_info = hub.get_all_domains()

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Domain ID':<10} | {'Domain Name':<35} | {'Train Tokens':<14} | {'Eval Tokens':<12}")
    print("--------------------------------------------------------------------------------")
    total_train_tok = 0
    total_eval_tok = 0
    for d in domains_info:
        print(f"Domain {d['id']:<3} | {d['name']:<35} | {d['train_token_count']:<14,d} | {d['eval_token_count']:<12,d}")
        total_train_tok += d['train_token_count']
        total_eval_tok += d['eval_token_count']
    print("--------------------------------------------------------------------------------")
    print(f"TOTAL      | {len(domains_info)} Domains Active{'':<20} | {total_train_tok:<14,d} | {total_eval_tok:<12,d}")
    print("================================================================================\n")

    print("[SUCCESS] All 7 domains ready for multi-dataset training and lifelong experiments.")

if __name__ == "__main__":
    main()
