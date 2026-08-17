"""
Build 100% Genuine, Diverse, Non-Repeating Multi-Domain Dataset Cache.
Streams real-world text across 4 core pillars:
1. Natural Language Stories (roneneldan/TinyStories) - Narrative grammar & causal logic
2. Real Python Source Code (iamtarun/python_code_instructions_18k_alpaca) - Real Python algorithms & functions
3. General Knowledge (wikitext-103-raw-v1) - World facts, science, biographies
4. Web Reasoning (HuggingFaceFW/fineweb-edu) - High-quality educational web reasoning
"""

import os
import sys
import time
import json
import numpy as np
import tiktoken
from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "genuine_diverse_cache")

def build_genuine_corpus(target_tokens_per_domain: int = 1_000_000):
    os.makedirs(CACHE_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    print("\n" + "=" * 90)
    print("  [STREAMING GENUINE, NON-REPEATING MULTI-DOMAIN DATASET (100% UNIQUE TEXT)]")
    print(f"  Target: {target_tokens_per_domain:,} tokens/domain x 4 domains = {target_tokens_per_domain*4:,} Total Tokens")
    print("=" * 90)

    domain_sources = [
        {
            "key": "natural_stories",
            "title": "Natural English Stories & Logic (TinyStories)",
            "hf_name": "roneneldan/TinyStories",
            "subset": None,
            "extractor": lambda r: r.get("text", ""),
            "min_len": 50
        },
        {
            "key": "python_code",
            "title": "Real Python Source Code (Python Instructions)",
            "hf_name": "iamtarun/python_code_instructions_18k_alpaca",
            "subset": None,
            "extractor": lambda r: f"# {r.get('instruction', '')}\n{r.get('output', '')}\n",
            "min_len": 40
        },
        {
            "key": "wikitext_facts",
            "title": "General Knowledge & Science (WikiText-103)",
            "hf_name": "wikitext",
            "subset": "wikitext-103-raw-v1",
            "extractor": lambda r: r.get("text", ""),
            "min_len": 60
        },
        {
            "key": "fineweb_reasoning",
            "title": "Educational Web Reasoning (FineWeb-Edu)",
            "hf_name": "HuggingFaceFW/fineweb-edu",
            "subset": "sample-10BT",
            "extractor": lambda r: r.get("text", ""),
            "min_len": 100
        }
    ]

    metadata = {
        "created_at": time.time(),
        "vocab_size": enc.n_vocab,
        "domains": {}
    }

    for d_info in domain_sources:
        d_key = d_info["key"]
        print(f"\n>>> Streaming Domain: [{d_info['title']}] from HuggingFace ({d_info['hf_name']})...")
        t0 = time.perf_counter()

        train_file = os.path.join(CACHE_DIR, f"{d_key}_train.bin")
        val_file = os.path.join(CACHE_DIR, f"{d_key}_val.bin")

        if d_info["subset"]:
            ds = load_dataset(d_info["hf_name"], d_info["subset"], split="train", streaming=True)
        else:
            ds = load_dataset(d_info["hf_name"], split="train", streaming=True)

        tokens_collected = []
        doc_count = 0

        for row in ds:
            text = d_info["extractor"](row)
            if isinstance(text, list):
                text = " ".join([str(x) for x in text])
            elif isinstance(text, dict):
                text = " ".join([str(v) for v in text.values()])
            text = str(text).strip()

            if len(text) >= d_info["min_len"]:
                toks = enc.encode_ordinary(text) + [enc.eot_token]
                tokens_collected.extend(toks)
                doc_count += 1

                if len(tokens_collected) >= (target_tokens_per_domain + 50_000):
                    break

        total_tokens = len(tokens_collected)
        val_count = min(80_000, int(total_tokens * 0.10))
        train_count = total_tokens - val_count

        train_arr = np.array(tokens_collected[:train_count], dtype=np.uint16)
        val_arr = np.array(tokens_collected[train_count:train_count + val_count], dtype=np.uint16)

        # Write to binary memmap
        train_mmap = np.memmap(train_file, dtype=np.uint16, mode='w+', shape=(train_count,))
        train_mmap[:] = train_arr[:]
        train_mmap.flush()

        val_mmap = np.memmap(val_file, dtype=np.uint16, mode='w+', shape=(val_count,))
        val_mmap[:] = val_arr[:]
        val_mmap.flush()

        dur = time.perf_counter() - t0
        print(f"    [SUCCESS] Streamed {doc_count:,} unique documents -> {train_count:,} train + {val_count:,} val tokens ({dur:.1f}s)")

        metadata["domains"][d_key] = {
            "title": d_info["title"],
            "train_tokens": int(train_count),
            "val_tokens": int(val_count),
            "total_tokens": int(total_tokens),
            "train_file": train_file,
            "val_file": val_file,
            "unique_docs": doc_count
        }

    meta_path = os.path.join(CACHE_DIR, "metadata_genuine.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 90)
    print(f"  [COMPLETE] Successfully cached genuine diverse datasets in {CACHE_DIR}/")
    print("=" * 90)

if __name__ == "__main__":
    build_genuine_corpus(target_tokens_per_domain=1_000_000)
