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
from transformers import AutoModelForCausalLM, AutoTokenizer

from sharding import model_memory_mb, shard_model_inplace


WORKER_ID = os.environ.get("WORKER_ID", f"worker-{uuid.uuid4().hex[:6]}")
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8001"))
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
WORKER_HOST = os.environ.get("WORKER_HOST", "localhost")
MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SHED_WEIGHTS = os.environ.get("SHED_WEIGHTS", "false").lower() in ("1", "true", "yes")


tokenizer: Optional[AutoTokenizer] = None
model = None
assigned_layers: list[int] = []
is_first_in_chain: bool = False
is_last_in_chain: bool = False


def load_full_model() -> None:
    global tokenizer, model
    print(f"[{WORKER_ID}] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    print(f"[{WORKER_ID}] full model loaded ({model_memory_mb(model):.1f} MB), {len(model.model.layers)} layers")


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
        "total_layers": len(model.model.layers),
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
    hb_url = f"{COORDINATOR_URL}/api/workers/heartbeat"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                resp = await client.post(hb_url, json={"worker_id": WORKER_ID})
                if resp.status_code >= 400:
                    print(f"[{WORKER_ID}] heartbeat rejected ({resp.status_code}), re-registering")
                    await register_with_coordinator()
            except Exception as e:
                print(f"[{WORKER_ID}] heartbeat failed: {e}, re-registering")
                try:
                    await register_with_coordinator()
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


app = FastAPI(title=f"Plasma-Mesh Worker ({WORKER_ID})", version="3.0.0", lifespan=lifespan)
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


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _run_assigned_layers(
    hidden_states: torch.Tensor,
    layers: list[int],
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    for layer_idx in layers:
        if layer_idx >= len(model.model.layers):
            continue
        block = model.model.layers[layer_idx]
        outputs = block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]
    return hidden_states


def _embed(input_ids_list: list[list[int]]) -> torch.Tensor:
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    return model.model.embed_tokens(input_ids)


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

    with torch.no_grad():
        if req.is_first:
            if req.input_ids is None:
                return {"error": "is_first requires input_ids"}
            hidden_states = _embed(req.input_ids)
        else:
            if req.hidden_states is None:
                return {"error": "non-first phase requires hidden_states"}
            hidden_states = torch.tensor(req.hidden_states, dtype=torch.float32)

        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = _causal_mask(seq_len, hidden_states.dtype)

        hidden_states = _run_assigned_layers(hidden_states, req.layers, position_ids, attention_mask)

        if req.is_last:
            hidden_states = model.model.norm(hidden_states)
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
