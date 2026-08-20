import os
import shutil
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "corpus_10domain")
os.makedirs(TARGET_DIR, exist_ok=True)

DOMAINS_10 = [
    {
        "id": "fineweb_edu",
        "title": "Educational Web Reasoning (FineWeb-Edu)",
        "src_train": "data/real_blend_cache/fineweb_edu_train.bin",
        "src_val": "data/real_blend_cache/fineweb_edu_val.bin",
    },
    {
        "id": "github_code",
        "title": "Systems & Multi-Language Code (GitHub Clean)",
        "src_train": "data/real_blend_cache/github_code_train.bin",
        "src_val": "data/real_blend_cache/github_code_val.bin",
    },
    {
        "id": "openweb_math",
        "title": "Formal Mathematics & Logic (OpenWebMath)",
        "src_train": "data/real_blend_cache/openweb_math_train.bin",
        "src_val": "data/real_blend_cache/openweb_math_val.bin",
    },
    {
        "id": "pubmed_biomedical",
        "title": "Biomedical & Clinical Science (PubMed)",
        "src_train": "data/real_blend_cache/pubmed_biomedical_train.bin",
        "src_val": "data/real_blend_cache/pubmed_biomedical_val.bin",
    },
    {
        "id": "freelaw_legal",
        "title": "Legal Contracts & Jurisprudence (FreeLaw)",
        "src_train": "data/real_blend_cache/freelaw_legal_train.bin",
        "src_val": "data/real_blend_cache/freelaw_legal_val.bin",
    },
    {
        "id": "arxiv_physics",
        "title": "Theoretical Physics & Relativity (arXiv)",
        "src_train": "data/real_blend_cache/arxiv_physics_train.bin",
        "src_val": "data/real_blend_cache/arxiv_physics_val.bin",
    },
    {
        "id": "financial_market",
        "title": "Financial Markets & SEC Reports (Macroeconomics)",
        "src_train": "data/real_blend_cache/financial_market_train.bin",
        "src_val": "data/real_blend_cache/financial_market_val.bin",
    },
    {
        "id": "gutenberg_literature",
        "title": "Classic Literature & Philosophy (Project Gutenberg)",
        "src_train": "data/real_blend_cache/gutenberg_literature_train.bin",
        "src_val": "data/real_blend_cache/gutenberg_literature_val.bin",
    },
    {
        "id": "python_code",
        "title": "Python Algorithms & AST Syntax (Alpaca)",
        "src_train": "data/scaled_real_corpus/python_code_train.bin",
        "src_val": "data/scaled_real_corpus/python_code_val.bin",
    },
    {
        "id": "wikitext_facts",
        "title": "Encyclopedic Factual Knowledge (WikiText-103)",
        "src_train": "data/scaled_real_corpus/wikitext_facts_train.bin",
        "src_val": "data/scaled_real_corpus/wikitext_facts_val.bin",
    }
]

def setup_10domain_dataset():
    print("\n" + "=" * 90)
    print("  [ASSEMBLING & VALIDATING 10 DIVERSE KNOWLEDGE DOMAINS]")
    print(f"  Destination Directory: {TARGET_DIR}")
    print("=" * 90)

    metadata = {
        "dataset_name": "10-Domain Orthogonal Real-World Corpus",
        "tokenizer": "gpt2",
        "vocab_size": 50304,
        "domains": {}
    }

    total_tokens = 0

    for d in DOMAINS_10:
        d_id = d["id"]
        train_src = os.path.join(PROJECT_ROOT, d["src_train"])
        val_src = os.path.join(PROJECT_ROOT, d["src_val"])
        
        train_dst = os.path.join(TARGET_DIR, f"{d_id}_train.bin")
        val_dst = os.path.join(TARGET_DIR, f"{d_id}_val.bin")

        shutil.copyfile(train_src, train_dst)
        shutil.copyfile(val_src, val_dst)

        train_data = np.memmap(train_dst, dtype=np.uint16, mode='r')
        val_data = np.memmap(val_dst, dtype=np.uint16, mode='r')

        train_tokens = len(train_data)
        val_tokens = len(val_data)
        total_tokens += (train_tokens + val_tokens)

        metadata["domains"][d_id] = {
            "title": d["title"],
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_file": f"{d_id}_train.bin",
            "val_file": f"{d_id}_val.bin"
        }

        print(f"  • {d_id:<22} | Train Tokens: {train_tokens:>9,} | Val Tokens: {val_tokens:>7,} | {d['title']}")

    meta_path = os.path.join(TARGET_DIR, "metadata_10domain.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("-" * 90)
    print(f"  Total Corpus Volume: {total_tokens:,} tokens ({total_tokens*2 / (1024*1024):.2f} MB)")
    print(f"  Saved metadata to: {meta_path}\n")

if __name__ == "__main__":
    setup_10domain_dataset()
