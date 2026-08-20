import os
import json
import numpy as np
import tiktoken

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "corpus_10domain")

def inspect_samples():
    enc = tiktoken.get_encoding("gpt2")
    meta_path = os.path.join(DATA_DIR, "metadata_10domain.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\n" + "=" * 95)
    print("  [REAL-WORLD DATASET AUDIT & EXCERPT INSPECTOR]")
    print("=" * 95)

    for d_id, info in meta["domains"].items():
        train_file = os.path.join(DATA_DIR, info["train_file"])
        data = np.memmap(train_file, dtype=np.uint16, mode='r')
        
        # Decode first 120 tokens
        sample_tokens = data[100:220].tolist()
        decoded_text = enc.decode(sample_tokens).replace("\n", " ")
        if len(decoded_text) > 140:
            decoded_text = decoded_text[:140] + "..."

        print(f"\n• Domain: [{d_id}] ({info['train_tokens']:,} unique tokens on disk)")
        print(f"  Title:  {info['title']}")
        print(f"  Sample: \"{decoded_text}\"")

if __name__ == "__main__":
    inspect_samples()
