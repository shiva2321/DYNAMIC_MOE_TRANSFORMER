"""
OpenAI-Compatible Streaming REST API Server for Hyperspace 2.0.
Implements:
1. POST /v1/chat/completions (Standard OpenAI schema + Server-Sent Events (SSE) streaming)
2. GET /v1/models (Model metadata and active expert count)
3. POST /v1/sleep/consolidate (Triggers biological sleep consolidation on live model)
4. GET /v1/telemetry/experts (Live expert routing distribution and SOC branching ratio)
"""

import os
import sys
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from data.dataset_hub import MultiDomainDatasetHub
from hyperspace.vsa import ComplexPhasorVSA
from hyperspace.sleep_consolidation import SleepConsolidationEngine

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hub = MultiDomainDatasetHub(seq_len=256, batch_size=4, use_bpe=True)

# Load model
def load_production_model():
    ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "scaled_production_hyperspace.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "benchmark_hyperspace_moe.pt")

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
        expert_ids = set(int(k.split(".")[4]) for k in exp_keys)
        target_num_experts = max(len(expert_ids), 2)
        while block.hyper_moe.num_experts < target_num_experts:
            seed = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(seed, label="loaded_expert")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_production_model()
consolidation_engine = SleepConsolidationEngine()

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Hyperspace 2.0 OpenAI-Compatible Gateway", version="2.0.0")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "hyperspace-2.0"
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.75
        top_p: Optional[float] = 0.9
        max_tokens: Optional[int] = 60
        stream: Optional[bool] = False

    @app.get("/v1/models")
    async def list_models():
        total_exp = sum(b.hyper_moe.num_experts for b in model.blocks)
        return {
            "object": "list",
            "data": [
                {
                    "id": "hyperspace-2.0",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "universal-substrait",
                    "metadata": {
                        "parameters": "63.12M",
                        "layers": len(model.blocks),
                        "active_experts_total": total_exp,
                        "architecture": "DynamicHyperMoE-FHRR-TwoCompartment",
                    }
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # Format messages into prompt
        prompt = ""
        for m in req.messages:
            if m.role == "system":
                prompt += f"System: {m.content}\n"
            elif m.role == "user":
                prompt += f"User: {m.content}\n"
            elif m.role == "assistant":
                prompt += f"Assistant: {m.content}\n"
        prompt += "Assistant:"

        prompt_tokens = hub.encode(prompt)
        curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        max_new = req.max_tokens or 50
        temp = req.temperature or 0.75

        if req.stream:
            async def event_generator():
                nonlocal curr_ids
                call_id = f"chatcmpl-{int(time.time()*1000)}"
                
                # Initial role chunk
                init_chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "hyperspace-2.0",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(init_chunk)}\n\n"

                for _ in range(max_new):
                    idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                    with torch.no_grad():
                        logits, _, telemetries = model(idx_cond, allow_spawning=False)
                    last_logits = logits[:, -1, :] / temp
                    probs = F.softmax(last_logits, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                    curr_ids = torch.cat([curr_ids, next_tok], dim=1)

                    tok_str = hub.decode([next_tok.item()])

                    chunk = {
                        "id": call_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "hyperspace-2.0",
                        "choices": [{"index": 0, "delta": {"content": tok_str}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)

                end_chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "hyperspace-2.0",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            # Non-streaming full generation
            for _ in range(max_new):
                idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
                with torch.no_grad():
                    logits, _, _ = model(idx_cond, allow_spawning=False)
                last_logits = logits[:, -1, :] / temp
                probs = F.softmax(last_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)

            full_output = hub.decode(curr_ids[0, len(prompt_tokens):].tolist())
            return {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "hyperspace-2.0",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_output},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": max_new,
                    "total_tokens": len(prompt_tokens) + max_new
                }
            }

    @app.post("/v1/sleep/consolidate")
    async def trigger_sleep_consolidation():
        results = consolidation_engine.consolidate_full_model(model, verbose=True)
        return {"status": "success", "layers_consolidated": results}

    @app.get("/v1/telemetry/experts")
    async def get_expert_telemetry():
        layer_stats = []
        for l_idx, block in enumerate(model.blocks):
            layer_stats.append({
                "layer": l_idx,
                "num_experts": block.hyper_moe.num_experts,
                "usages": block.hyper_moe.expert_usage_counts[:block.hyper_moe.num_experts].tolist()
            })
        return {"status": "active", "layers": layer_stats}

def main():
    if not FASTAPI_AVAILABLE:
        print("[ERROR] FastAPI / Uvicorn not installed. Please install via: pip install fastapi uvicorn")
        sys.exit(1)
    print("=" * 80)
    print("  [STARTING] HYPERSPACE 2.0 OPENAI-COMPATIBLE STREAMING API SERVER")
    print("  Endpoints:")
    print("    POST http://localhost:8000/v1/chat/completions (OpenAI SSE Stream)")
    print("    GET  http://localhost:8000/v1/models")
    print("    POST http://localhost:8000/v1/sleep/consolidate")
    print("    GET  http://localhost:8000/v1/telemetry/experts")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
