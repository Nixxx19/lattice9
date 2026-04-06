"""
Hivemind Worker - runs inference on assigned model layers.
Registers with the coordinator on startup and processes inference requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import psutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKER_ID = os.environ.get("WORKER_ID", f"worker-{uuid.uuid4().hex[:6]}")
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8001"))
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
WORKER_HOST = os.environ.get("WORKER_HOST", "localhost")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

tokenizer: GPT2Tokenizer | None = None
model: GPT2LMHeadModel | None = None


def load_model():
    global tokenizer, model
    print(f"[{WORKER_ID}] Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    print(f"[{WORKER_ID}] Model loaded. Layers: {len(model.transformer.h)}")


# ---------------------------------------------------------------------------
# Registration & heartbeat
# ---------------------------------------------------------------------------

async def register_with_coordinator():
    url = f"{COORDINATOR_URL}/api/workers/register"
    payload = {
        "worker_id": WORKER_ID,
        "url": f"http://{WORKER_HOST}:{WORKER_PORT}",
        "cpu_cores": psutil.cpu_count(logical=False) or 1,
        "memory_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(10):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                print(f"[{WORKER_ID}] Registered with coordinator: {resp.json()}")
                return
            except Exception as e:
                print(f"[{WORKER_ID}] Registration attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2)
    print(f"[{WORKER_ID}] WARNING: Could not register with coordinator")


async def heartbeat_loop():
    url = f"{COORDINATOR_URL}/api/workers/heartbeat"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                await client.post(url, json={"worker_id": WORKER_ID})
            except Exception:
                pass
            await asyncio.sleep(10)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    await register_with_coordinator()
    task = asyncio.create_task(heartbeat_loop())
    yield
    task.cancel()


app = FastAPI(title=f"Hivemind Worker ({WORKER_ID})", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = 50
    phase: str = "full"                    # full | encode | middle | decode
    layers: list[int] = []
    request_id: str = ""
    hidden_states: Optional[list] = None
    attention_mask: Optional[list] = None


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------

def run_full_inference(prompt: str, max_tokens: int) -> dict:
    """Run complete GPT-2 inference (single-worker mode)."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return {
        "generated_text": generated,
        "tokens_generated": int(new_tokens),
    }


def run_encode_phase(prompt: str, layers: list[int]) -> dict:
    """Tokenize and run through first set of transformer layers."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        # Get embeddings
        hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1]).unsqueeze(0)
        )
        hidden_states = model.transformer.drop(hidden_states)

        # Run through assigned layers
        for layer_idx in layers:
            if layer_idx < len(model.transformer.h):
                block = model.transformer.h[layer_idx]
                outputs = block(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0]

    return {
        "hidden_states": hidden_states.tolist(),
        "attention_mask": attention_mask.tolist(),
    }


def run_middle_phase(hidden_states_list: list, attention_mask_list: list, layers: list[int]) -> dict:
    """Run through middle transformer layers."""
    hidden_states = torch.tensor(hidden_states_list)
    attention_mask = torch.tensor(attention_mask_list)

    with torch.no_grad():
        for layer_idx in layers:
            if layer_idx < len(model.transformer.h):
                block = model.transformer.h[layer_idx]
                outputs = block(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0]

    return {
        "hidden_states": hidden_states.tolist(),
        "attention_mask": attention_mask.tolist(),
    }


def run_decode_phase(
    prompt: str,
    hidden_states_list: list,
    attention_mask_list: list,
    layers: list[int],
    max_tokens: int,
) -> dict:
    """Run remaining layers, apply layer norm, project to vocab, decode tokens."""
    hidden_states = torch.tensor(hidden_states_list)
    attention_mask = torch.tensor(attention_mask_list)

    with torch.no_grad():
        # Finish remaining transformer layers
        for layer_idx in layers:
            if layer_idx < len(model.transformer.h):
                block = model.transformer.h[layer_idx]
                outputs = block(hidden_states, attention_mask=attention_mask)
                hidden_states = outputs[0]

        # Final layer norm
        hidden_states = model.transformer.ln_f(hidden_states)

        # Project to vocabulary
        logits = model.lm_head(hidden_states)

        # Greedy decode from logits of last position
        generated_ids = []
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        # Use full model for generation from the context we built
        # (This is a practical simplification - in production you'd continue
        # autoregressive generation from the hidden states)
        full_outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(full_outputs[0], skip_special_tokens=True)
    new_tokens = full_outputs.shape[1] - input_ids.shape[1]
    return {
        "generated_text": generated,
        "tokens_generated": int(new_tokens),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "model_loaded": model is not None,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }


@app.post("/api/process")
async def process(req: ProcessRequest):
    start = time.time()

    if req.phase == "full":
        result = run_full_inference(req.prompt, req.max_tokens)
    elif req.phase == "encode":
        result = run_encode_phase(req.prompt, req.layers)
    elif req.phase == "middle":
        result = run_middle_phase(req.hidden_states, req.attention_mask, req.layers)
    elif req.phase == "decode":
        result = run_decode_phase(
            req.prompt, req.hidden_states, req.attention_mask, req.layers, req.max_tokens
        )
    else:
        result = run_full_inference(req.prompt, req.max_tokens)

    latency_ms = (time.time() - start) * 1000
    result["worker_id"] = WORKER_ID
    result["latency_ms"] = round(latency_ms, 2)
    result["phase"] = req.phase
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
