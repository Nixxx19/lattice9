"""
Hivemind Coordinator - manages workers and distributes inference across them.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scheduler import Scheduler, Strategy


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    worker_id: str
    url: str
    cpu_cores: int = 1
    memory_mb: int = 1024


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


class InferResponse(BaseModel):
    request_id: str
    prompt: str
    result: str
    tokens_generated: int
    total_latency_ms: float
    worker_trace: list[dict]


class HeartbeatRequest(BaseModel):
    worker_id: str


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
job_history: list[dict] = []
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0)
    yield
    await http_client.aclose()


app = FastAPI(title="Hivemind Coordinator", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "coordinator", "workers": len(scheduler.workers)}


@app.post("/api/workers/register")
async def register_worker(req: RegisterRequest):
    worker = scheduler.register_worker(req.worker_id, req.url, req.cpu_cores, req.memory_mb)
    return {"status": "registered", "worker": worker.to_dict()}


@app.post("/api/workers/heartbeat")
async def worker_heartbeat(req: HeartbeatRequest):
    scheduler.heartbeat(req.worker_id)
    return {"status": "ok"}


@app.get("/api/workers")
async def list_workers():
    workers = [w.to_dict() for w in scheduler.workers.values()]
    stats = scheduler.get_throughput_stats()
    return {"workers": workers, "stats": stats}


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": job_history[-50:]}


@app.post("/api/infer")
async def infer(req: InferRequest):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    assignments = scheduler.get_worker_assignments()

    if not assignments:
        raise HTTPException(status_code=503, detail="No workers available")

    worker_trace: list[dict] = []

    # Phase 1 – tokenize + encode via the first worker's assigned layers
    # Phase 2 – propagate hidden states through subsequent workers
    # Phase 3 – decode from the last worker
    #
    # For simplicity with real GPT-2, we send the prompt to worker-0 which
    # runs its assigned layers, then pass intermediate tensors down the chain.
    # The last worker decodes tokens.

    current_payload: dict = {
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "phase": "full" if len(assignments) == 1 else "encode",
        "layers": assignments[0]["layers"],
        "request_id": request_id,
    }

    for idx, assignment in enumerate(assignments):
        is_last = idx == len(assignments) - 1
        if len(assignments) > 1:
            if idx == 0:
                current_payload["phase"] = "encode"
            elif is_last:
                current_payload["phase"] = "decode"
            else:
                current_payload["phase"] = "middle"

        current_payload["layers"] = assignment["layers"]
        worker_url = assignment["url"]
        worker_id = assignment["worker_id"]

        step_start = time.time()
        try:
            resp = await http_client.post(f"{worker_url}/api/process", json=current_payload)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Worker {worker_id} at {worker_url} failed: {str(e)}",
            )
        step_ms = (time.time() - step_start) * 1000

        scheduler.record_job(worker_id, step_ms)
        worker_trace.append({
            "worker_id": worker_id,
            "url": worker_url,
            "phase": current_payload.get("phase", "full"),
            "layers": assignment["layers"],
            "latency_ms": round(step_ms, 2),
        })

        # Pass output forward as input to next worker
        if not is_last:
            current_payload = {
                "hidden_states": result.get("hidden_states"),
                "attention_mask": result.get("attention_mask"),
                "prompt": req.prompt,
                "max_tokens": req.max_tokens,
                "request_id": request_id,
            }

    total_ms = (time.time() - start) * 1000
    generated_text = result.get("generated_text", "")
    tokens_generated = result.get("tokens_generated", 0)

    job_record = {
        "request_id": request_id,
        "prompt": req.prompt,
        "result": generated_text,
        "tokens_generated": tokens_generated,
        "total_latency_ms": round(total_ms, 2),
        "worker_trace": worker_trace,
        "timestamp": time.time(),
    }
    job_history.append(job_record)

    return InferResponse(
        request_id=request_id,
        prompt=req.prompt,
        result=generated_text,
        tokens_generated=tokens_generated,
        total_latency_ms=round(total_ms, 2),
        worker_trace=worker_trace,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
