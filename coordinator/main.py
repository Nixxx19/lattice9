from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from scheduler import Scheduler, Strategy


class RegisterRequest(BaseModel):
    worker_id: str
    url: str
    cpu_cores: int = 1
    memory_mb: int = 1024
    total_layers: int = 0
    model_name: str = ""


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    deterministic: bool = False
    temperature: float = 0.9
    top_p: float = 0.92
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3


class InferResponse(BaseModel):
    request_id: str
    prompt: str
    result: str
    tokens_generated: int
    total_latency_ms: float
    worker_trace: list[dict]


class HeartbeatRequest(BaseModel):
    worker_id: str


DEFAULT_MODEL = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=0)
active_model: str = DEFAULT_MODEL
job_history: list[dict] = []
http_client: httpx.AsyncClient | None = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, tokenizer
    http_client = httpx.AsyncClient(timeout=120.0)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    yield
    await http_client.aclose()


app = FastAPI(title="Plasma-Mesh Coordinator", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "coordinator", "workers": len(scheduler.workers)}


@app.get("/metrics")
async def metrics():
    lines: list[str] = []
    stats = scheduler.get_throughput_stats()

    lines.append("# TYPE plasma_mesh_workers_total gauge")
    lines.append(f"plasma_mesh_workers_total {stats['workers_total']}")
    lines.append("# TYPE plasma_mesh_workers_active gauge")
    lines.append(f"plasma_mesh_workers_active {stats['workers_active']}")
    lines.append("# TYPE plasma_mesh_jobs_completed_total counter")
    lines.append(f"plasma_mesh_jobs_completed_total {len(job_history)}")
    lines.append("# TYPE plasma_mesh_pipeline_calls_total counter")
    lines.append(f"plasma_mesh_pipeline_calls_total {stats['total_jobs']}")
    lines.append("# TYPE plasma_mesh_avg_call_latency_ms gauge")
    lines.append(f"plasma_mesh_avg_call_latency_ms {stats['avg_latency_ms']}")

    lines.append("# TYPE plasma_mesh_worker_jobs_processed counter")
    lines.append("# TYPE plasma_mesh_worker_avg_latency_ms gauge")
    lines.append("# TYPE plasma_mesh_worker_layers_assigned gauge")
    for w in scheduler.workers.values():
        labels = f'worker_id="{w.worker_id}"'
        lines.append(f"plasma_mesh_worker_jobs_processed{{{labels}}} {w.jobs_processed}")
        lines.append(f"plasma_mesh_worker_avg_latency_ms{{{labels}}} {w.avg_latency_ms}")
        lines.append(f"plasma_mesh_worker_layers_assigned{{{labels}}} {len(w.assigned_layers)}")

    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/plain; version=0.0.4",
    )


@app.post("/api/workers/register")
async def register_worker(req: RegisterRequest):
    global active_model
    if req.total_layers and scheduler.total_layers != req.total_layers:
        scheduler.total_layers = req.total_layers
        for w in scheduler.workers.values():
            w.assigned_layers = []
        scheduler._reassign_layers()
    if req.model_name:
        active_model = req.model_name
    worker = scheduler.register_worker(req.worker_id, req.url, req.cpu_cores, req.memory_mb)
    assignment = next(
        (a for a in scheduler.get_worker_assignments() if a["worker_id"] == worker.worker_id),
        None,
    )
    healthy = scheduler.get_healthy_workers()
    healthy_ids = [w.worker_id for w in healthy]
    is_first = bool(healthy_ids) and healthy_ids[0] == worker.worker_id
    is_last = bool(healthy_ids) and healthy_ids[-1] == worker.worker_id
    return {
        "status": "registered",
        "worker": worker.to_dict(),
        "assignment": {
            "layers": assignment["layers"] if assignment else [],
            "is_first": is_first,
            "is_last": is_last,
            "total_layers": scheduler.total_layers,
        },
    }


@app.post("/api/workers/heartbeat")
async def worker_heartbeat(req: HeartbeatRequest):
    if req.worker_id not in scheduler.workers:
        raise HTTPException(status_code=404, detail="unknown worker, re-register")
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


@app.delete("/api/workers/{worker_id}")
async def evict_worker(worker_id: str):
    if worker_id not in scheduler.workers:
        raise HTTPException(status_code=404, detail="unknown worker")
    scheduler.remove_worker(worker_id)
    return {"status": "evicted", "remaining": list(scheduler.workers.keys())}


def _phase_for(idx: int, total: int) -> str:
    if total == 1:
        return "full"
    if idx == 0:
        return "encode"
    if idx == total - 1:
        return "decode"
    return "middle"


class WorkerCallError(Exception):
    def __init__(self, worker_id: str, cause: Exception):
        self.worker_id = worker_id
        self.cause = cause
        super().__init__(f"worker {worker_id}: {cause}")


async def _pipeline_pass(
    assignments: list[dict],
    input_ids: list[list[int]],
    stats: dict,
    request_id: str,
) -> list[float]:
    hidden_states = None
    for idx, assignment in enumerate(assignments):
        is_first = idx == 0
        is_last = idx == len(assignments) - 1

        payload: dict = {
            "is_first": is_first,
            "is_last": is_last,
            "layers": assignment["layers"],
            "request_id": request_id,
        }
        if is_first:
            payload["input_ids"] = input_ids
        else:
            payload["hidden_states"] = hidden_states

        step_start = time.time()
        try:
            resp = await http_client.post(
                f"{assignment['url']}/api/process",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            raise WorkerCallError(assignment["worker_id"], e)
        elapsed_ms = (time.time() - step_start) * 1000

        wid = assignment["worker_id"]
        if wid in stats:
            stats[wid]["calls"] += 1
            stats[wid]["total_ms"] += elapsed_ms
        scheduler.record_job(wid, elapsed_ms)

        if is_last:
            return result["logits"][0]
        hidden_states = result["hidden_states"]
    return []


def _init_stats(assignments: list[dict]) -> dict[str, dict]:
    return {
        a["worker_id"]: {
            "url": a["url"],
            "layers": a["layers"],
            "calls": 0,
            "total_ms": 0.0,
        }
        for a in assignments
    }


def _apply_repetition_penalty(logits: torch.Tensor, generated: list[int], penalty: float) -> None:
    if penalty == 1.0 or not generated:
        return
    seen = set(generated)
    for tid in seen:
        v = logits[tid].item()
        logits[tid] = v / penalty if v > 0 else v * penalty


def _banned_by_ngram(generated: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated) < ngram_size:
        return set()
    prefix = tuple(generated[-(ngram_size - 1):]) if ngram_size > 1 else ()
    banned: set[int] = set()
    for i in range(len(generated) - ngram_size + 1):
        window = tuple(generated[i : i + ngram_size - 1])
        if window == prefix:
            banned.add(generated[i + ngram_size - 1])
    return banned


def _sample_token(logits_row: list[float], generated: list[int], req: InferRequest) -> int:
    logits = torch.tensor(logits_row)

    _apply_repetition_penalty(logits, generated, req.repetition_penalty)
    for tid in _banned_by_ngram(generated, req.no_repeat_ngram_size):
        logits[tid] = -float("inf")

    if req.deterministic:
        return int(torch.argmax(logits).item())

    scaled = logits / max(req.temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)

    if 0.0 < req.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > req.top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice].item())

    return int(torch.multinomial(probs, 1).item())


@app.post("/api/infer")
async def infer(req: InferRequest):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    assignments = scheduler.get_worker_assignments()
    if not assignments:
        raise HTTPException(status_code=503, detail="No workers available")

    encoded = tokenizer(req.prompt, return_tensors="pt")["input_ids"][0].tolist()
    generated_ids: list[int] = list(encoded)
    prompt_token_count = len(generated_ids)
    eos_id = tokenizer.eos_token_id

    stats: dict[str, dict] = _init_stats(assignments)
    reshard_events: list[dict] = []

    for token_idx in range(req.max_tokens):
        attempt = 0
        while True:
            try:
                logits_row = await _pipeline_pass(
                    assignments,
                    [generated_ids],
                    stats,
                    request_id,
                )
                break
            except WorkerCallError as e:
                attempt += 1
                scheduler.remove_worker(e.worker_id)
                assignments = scheduler.get_worker_assignments()
                if not assignments:
                    raise HTTPException(
                        status_code=503,
                        detail=f"all workers failed after {e.worker_id}",
                    )
                for a in assignments:
                    stats.setdefault(a["worker_id"], {
                        "url": a["url"],
                        "layers": a["layers"],
                        "calls": 0,
                        "total_ms": 0.0,
                    })
                    stats[a["worker_id"]]["layers"] = a["layers"]
                reshard_events.append({
                    "token_index": token_idx,
                    "dropped_worker": e.worker_id,
                    "remaining": [a["worker_id"] for a in assignments],
                })
                if attempt > 5:
                    raise HTTPException(
                        status_code=503,
                        detail="too many reshards in a single token step",
                    )

        next_token = _sample_token(logits_row, generated_ids, req)
        if next_token == eos_id:
            break
        generated_ids.append(next_token)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids) - prompt_token_count
    total_ms = (time.time() - start) * 1000

    worker_trace = []
    for idx, a in enumerate(assignments):
        wid = a["worker_id"]
        s = stats[wid]
        worker_trace.append({
            "worker_id": wid,
            "url": s["url"],
            "phase": _phase_for(idx, len(assignments)),
            "layers": s["layers"],
            "calls": s["calls"],
            "latency_ms": round(s["total_ms"], 2),
            "avg_call_ms": round(s["total_ms"] / s["calls"], 2) if s["calls"] else 0.0,
        })

    job_record = {
        "request_id": request_id,
        "prompt": req.prompt,
        "result": generated_text,
        "tokens_generated": tokens_generated,
        "total_latency_ms": round(total_ms, 2),
        "worker_trace": worker_trace,
        "reshard_events": reshard_events,
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_inference(req: InferRequest) -> AsyncIterator[str]:
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    assignments = scheduler.get_worker_assignments()
    if not assignments:
        yield _sse("error", {"detail": "No workers available"})
        return

    yield _sse("start", {
        "request_id": request_id,
        "prompt": req.prompt,
        "assignments": [
            {"worker_id": a["worker_id"], "layers": a["layers"]} for a in assignments
        ],
    })

    encoded = tokenizer(req.prompt, return_tensors="pt")["input_ids"][0].tolist()
    generated_ids: list[int] = list(encoded)
    prompt_token_count = len(generated_ids)
    eos_id = tokenizer.eos_token_id

    stats = _init_stats(assignments)
    reshard_events: list[dict] = []

    for token_idx in range(req.max_tokens):
        attempt = 0
        while True:
            try:
                logits_row = await _pipeline_pass(
                    assignments, [generated_ids], stats, request_id
                )
                break
            except WorkerCallError as e:
                attempt += 1
                scheduler.remove_worker(e.worker_id)
                assignments = scheduler.get_worker_assignments()
                if not assignments:
                    yield _sse("error", {"detail": "all workers failed"})
                    return
                for a in assignments:
                    stats.setdefault(a["worker_id"], {
                        "url": a["url"],
                        "layers": a["layers"],
                        "calls": 0,
                        "total_ms": 0.0,
                    })
                    stats[a["worker_id"]]["layers"] = a["layers"]
                reshard_events.append({
                    "token_index": token_idx,
                    "dropped_worker": e.worker_id,
                })
                yield _sse("reshard", {
                    "token_index": token_idx,
                    "dropped_worker": e.worker_id,
                    "remaining": [a["worker_id"] for a in assignments],
                })
                if attempt > 5:
                    yield _sse("error", {"detail": "too many reshards"})
                    return

        next_token = _sample_token(logits_row, generated_ids, req)
        if next_token == eos_id:
            break
        generated_ids.append(next_token)

        token_text = tokenizer.decode([next_token])
        yield _sse("token", {
            "index": token_idx,
            "token_id": next_token,
            "token_text": token_text,
            "decode_worker": assignments[-1]["worker_id"],
        })

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    total_ms = (time.time() - start) * 1000
    tokens_generated = len(generated_ids) - prompt_token_count

    worker_trace = []
    for idx, a in enumerate(assignments):
        wid = a["worker_id"]
        s = stats.get(wid, {"url": a["url"], "layers": a["layers"], "calls": 0, "total_ms": 0.0})
        worker_trace.append({
            "worker_id": wid,
            "url": s["url"],
            "phase": _phase_for(idx, len(assignments)),
            "layers": s["layers"],
            "calls": s["calls"],
            "latency_ms": round(s["total_ms"], 2),
            "avg_call_ms": round(s["total_ms"] / s["calls"], 2) if s["calls"] else 0.0,
        })

    job_history.append({
        "request_id": request_id,
        "prompt": req.prompt,
        "result": full_text,
        "tokens_generated": tokens_generated,
        "total_latency_ms": round(total_ms, 2),
        "worker_trace": worker_trace,
        "reshard_events": reshard_events,
        "streamed": True,
        "timestamp": time.time(),
    })

    yield _sse("done", {
        "request_id": request_id,
        "result": full_text,
        "tokens_generated": tokens_generated,
        "total_latency_ms": round(total_ms, 2),
        "reshard_events": reshard_events,
    })


@app.post("/api/infer/stream")
async def infer_stream(req: InferRequest):
    return StreamingResponse(_stream_inference(req), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
