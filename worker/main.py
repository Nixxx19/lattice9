from __future__ import annotations

import asyncio
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

from sharding import model_memory_mb, shard_model_inplace


WORKER_ID = os.environ.get("WORKER_ID", f"worker-{uuid.uuid4().hex[:6]}")
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8001"))
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
WORKER_HOST = os.environ.get("WORKER_HOST", "localhost")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2-large")
SHED_WEIGHTS = os.environ.get("SHED_WEIGHTS", "false").lower() in ("1", "true", "yes")


tokenizer: Optional[GPT2Tokenizer] = None
model: Optional[GPT2LMHeadModel] = None
assigned_layers: list[int] = []
is_first_in_chain: bool = False
is_last_in_chain: bool = False


def load_full_model() -> None:
    global tokenizer, model
    print(f"[{WORKER_ID}] loading {MODEL_NAME}")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"[{WORKER_ID}] full model loaded ({model_memory_mb(model):.1f} MB)")


def apply_assignment(layers: list[int], is_first: bool, is_last: bool) -> None:
    global assigned_layers, is_first_in_chain, is_last_in_chain
    assigned_layers = layers
    is_first_in_chain = is_first
    is_last_in_chain = is_last
    if SHED_WEIGHTS:
        shard_model_inplace(model, layers, is_first, is_last)
        print(
            f"[{WORKER_ID}] sharded to layers {layers} "
            f"(first={is_first}, last={is_last}) -> {model_memory_mb(model):.1f} MB"
        )
    else:
        print(
            f"[{WORKER_ID}] assigned layers {layers} "
            f"(first={is_first}, last={is_last}); full model kept for failure recovery"
        )


async def register_with_coordinator() -> None:
    url = f"{COORDINATOR_URL}/api/workers/register"
    payload = {
        "worker_id": WORKER_ID,
        "url": f"http://{WORKER_HOST}:{WORKER_PORT}",
        "cpu_cores": psutil.cpu_count(logical=False) or 1,
        "memory_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
        "total_layers": len(model.transformer.h),
        "model_name": MODEL_NAME,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(10):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                print(f"[{WORKER_ID}] registered")
                assignment = data.get("assignment", {})
                if assignment.get("layers"):
                    apply_assignment(
                        assignment["layers"],
                        assignment.get("is_first", False),
                        assignment.get("is_last", False),
                    )
                return
            except Exception as e:
                print(f"[{WORKER_ID}] registration attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2)
    print(f"[{WORKER_ID}] could not register")


async def heartbeat_loop() -> None:
    url = f"{COORDINATOR_URL}/api/workers/heartbeat"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                await client.post(url, json={"worker_id": WORKER_ID})
            except Exception:
                pass
            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_full_model()
    await register_with_coordinator()
    task = asyncio.create_task(heartbeat_loop())
    yield
    task.cancel()


app = FastAPI(title=f"Hivemind Worker ({WORKER_ID})", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    layers: list[int]
    is_first: bool = False
    is_last: bool = False
    input_ids: Optional[list[list[int]]] = None
    hidden_states: Optional[list] = None
    request_id: str = ""


def _run_assigned_layers(hidden_states: torch.Tensor, layers: list[int]) -> torch.Tensor:
    for layer_idx in layers:
        if layer_idx >= len(model.transformer.h):
            continue
        block = model.transformer.h[layer_idx]
        hidden_states = block(hidden_states)[0]
    return hidden_states


def _embed(input_ids_list: list[list[int]]) -> torch.Tensor:
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(positions)
    return model.transformer.drop(hidden_states)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "model_loaded": model is not None,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }


@app.post("/api/tokenize")
async def tokenize(payload: dict):
    text = payload.get("text", "")
    ids = tokenizer(text, return_tensors="pt")["input_ids"].tolist()
    return {"input_ids": ids, "eos_token_id": tokenizer.eos_token_id}


@app.post("/api/decode")
async def decode_text(payload: dict):
    ids = payload.get("input_ids", [])
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return {"text": text}


@app.post("/api/process")
async def process(req: ProcessRequest):
    start = time.time()

    with torch.no_grad():
        if req.is_first:
            if req.input_ids is None:
                return {"error": "is_first requires input_ids"}
            hidden_states = _embed(req.input_ids)
        else:
            if req.hidden_states is None:
                return {"error": "non-first phase requires hidden_states"}
            hidden_states = torch.tensor(req.hidden_states)

        hidden_states = _run_assigned_layers(hidden_states, req.layers)

        if req.is_last:
            hidden_states = model.transformer.ln_f(hidden_states)
            last_logits = model.lm_head(hidden_states[:, -1, :])
            payload = {"logits": last_logits.tolist()}
        else:
            payload = {"hidden_states": hidden_states.tolist()}

    latency_ms = (time.time() - start) * 1000
    payload["worker_id"] = WORKER_ID
    payload["latency_ms"] = round(latency_ms, 2)
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
