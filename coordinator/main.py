from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2Tokenizer

from scheduler import Scheduler, Strategy


class RegisterRequest(BaseModel):
    worker_id: str
    url: str
    cpu_cores: int = 1
    memory_mb: int = 1024


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    deterministic: bool = False
    temperature: float = 0.8
    top_p: float = 0.95


class InferResponse(BaseModel):
    request_id: str
    prompt: str
    result: str
    tokens_generated: int
    total_latency_ms: float
    worker_trace: list[dict]


class HeartbeatRequest(BaseModel):
    worker_id: str


scheduler = Scheduler(strategy=Strategy.UNIFORM, total_layers=12)
job_history: list[dict] = []
http_client: httpx.AsyncClient | None = None
tokenizer: GPT2Tokenizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, tokenizer
    http_client = httpx.AsyncClient(timeout=120.0)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    yield
    await http_client.aclose()


app = FastAPI(title="Hivemind Coordinator", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _phase_for(idx: int, total: int) -> str:
    if total == 1:
        return "full"
    if idx == 0:
        return "encode"
    if idx == total - 1:
        return "decode"
    return "middle"


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
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Worker {assignment['worker_id']} failed: {e}",
            )
        elapsed_ms = (time.time() - step_start) * 1000

        wid = assignment["worker_id"]
        stats[wid]["calls"] += 1
        stats[wid]["total_ms"] += elapsed_ms
        scheduler.record_job(wid, elapsed_ms)

        if is_last:
            return result["logits"][0]
        hidden_states = result["hidden_states"]
    return []


def _sample_token(logits_row: list[float], req: InferRequest) -> int:
    logits = torch.tensor(logits_row)
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

    stats: dict[str, dict] = {
        a["worker_id"]: {
            "url": a["url"],
            "layers": a["layers"],
            "calls": 0,
            "total_ms": 0.0,
        }
        for a in assignments
    }

    for _ in range(req.max_tokens):
        logits_row = await _pipeline_pass(
            assignments,
            [generated_ids],
            stats,
            request_id,
        )
        next_token = _sample_token(logits_row, req)
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
