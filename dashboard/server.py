"""
Hyperspace 2.0 Web Studio Server.
Provides real-time interactive generation, multi-domain exploration,
token-by-token expert attribution, and live experiment telemetry.
"""

import sys
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from model.nanogpt import HyperTransformerLM
from data.dataset_hub import MultiDomainDatasetHub, DOMAIN_LIST, DOMAIN_METADATA
from hyperspace.vsa import ComplexPhasorVSA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hub = MultiDomainDatasetHub(seq_len=256, batch_size=4, use_bpe=True)

def load_dynamic_checkpoint(ckpt_path: str, device: torch.device) -> HyperTransformerLM:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    state_dict = state.get("model_state", state)

    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 6)
    d_ff = config.get("d_ff", 1024)
    d_hyper = config.get("d_hyper", 2048)
    top_k = config.get("top_k", 2)
    max_experts = config.get("max_experts", 16)
    spawn_threshold = config.get("spawn_threshold", 0.25)
    seq_len = config.get("seq_len", 256)
    max_seq_len = seq_len + 32

    model = HyperTransformerLM(
        vocab_size=50304,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=top_k,
        max_experts=max_experts,
        spawn_threshold=spawn_threshold,
        max_seq_len=max_seq_len,
    ).to(device)

    for b_idx, block in enumerate(model.blocks):
        exp_keys = [k for k in state_dict.keys() if k.startswith(f"blocks.{b_idx}.hyper_moe.experts.")]
        expert_ids = set()
        for k in exp_keys:
            parts = k.split(".")
            exp_id = int(parts[4])
            expert_ids.add(exp_id)

        target_num_experts = max(len(expert_ids), 2)
        while block.hyper_moe.num_experts < target_num_experts:
            seed = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(seed, label="loaded_expert")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[Studio Server] Loaded dynamic checkpoint from {ckpt_path}")
    return model

# Prefer scaled production model
ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "scaled_production_hyperspace.pt")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "benchmark_hyperspace_moe.pt")

if os.path.exists(ckpt_path):
    model = load_dynamic_checkpoint(ckpt_path, device=device)
else:
    model = HyperTransformerLM(vocab_size=hub.vocab_size, d_model=384, n_layers=6, n_heads=6, d_ff=1024, d_hyper=2048, top_k=2).to(device)
    model.eval()

class HyperspaceStudioHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(PROJECT_ROOT, "dashboard", "index.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
                
        elif self.path == "/api/domains":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            domains_data = hub.get_all_domains()
            self.wfile.write(json.dumps(domains_data).encode("utf-8"))
            
        elif self.path == "/api/benchmark_results":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            json_file = os.path.join(PROJECT_ROOT, "experiments", "experiment_benchmark_results.json")
            if os.path.exists(json_file):
                with open(json_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(json.dumps({"status": "no_benchmark_data"}).encode("utf-8"))

        elif self.path.startswith("/plots/"):
            filename = os.path.basename(self.path)
            plot_file = os.path.join(PROJECT_ROOT, "experiments", "plots", filename)
            if os.path.exists(plot_file):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(plot_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))
            prompt = req.get("prompt", "def quicksort(array):")
            max_new = int(req.get("max_new_tokens", 40))
            temp = float(req.get("temperature", 0.75))
            top_k_filter = int(req.get("top_k", 30))

            prompt_tokens = hub.encode(prompt)
            curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

            generated_tokens = []
            last_sigma = 1.0000
            last_temp = 10.0

            for _ in range(max_new):
                idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                with torch.no_grad():
                    logits, _, telemetries = model(idx_cond, allow_spawning=False)

                last_logits = logits[:, -1, :] / temp
                if top_k_filter is not None:
                    v, _ = torch.topk(last_logits, min(top_k_filter, last_logits.size(-1)))
                    last_logits[last_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(last_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)

                if telemetries:
                    top_exp = telemetries[0]["top_indices"][0, -1, 0].item()
                    top_w = telemetries[0]["top_weights"][0, -1, 0].item()
                    last_sigma = telemetries[0]["branching_ratio"]
                    last_temp = telemetries[0]["routing_temperature"]
                else:
                    top_exp = 0
                    top_w = 1.0

                token_str = hub.decode([next_tok.item()])
                generated_tokens.append({
                    "char": token_str,
                    "expert": top_exp,
                    "weight": round(top_w, 3),
                })

            resp = {
                "prompt": prompt,
                "tokens": generated_tokens,
                "sigma": round(last_sigma, 4),
                "temp": round(last_temp, 2),
                "active_experts": model.blocks[0].hyper_moe.num_experts,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, HyperspaceStudioHandler)
    print(f"Hyperspace 2.0 Multi-Domain Studio running at http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server(8080)
