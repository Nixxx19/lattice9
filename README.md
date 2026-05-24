# lattice9

deterministic token-to-token relay inference across worker nodes. split a transformer's layers across a pool of machines, relay each generated token through the full pipeline, and prove the math is bitwise-identical to a monolithic forward pass.

defaults to `TinyLlama-1.1B-Chat` (1.1b params, 22 layers, instruction-tuned, ungated); swap to any huggingface model with `MODEL_NAME=...`.

```bash
docker compose up --build
python cli/main.py infer -p "the quick brown fox" -d
```

## what it does

- splits the model's transformer layers across a pool of workers as contiguous chunks
- coordinator runs the autoregressive loop; each output token does a full pipeline pass
- `--deterministic` mode uses greedy decoding so output is reproducible
- ci runs a math-level parity test: the sharded forward pass is asserted bitwise-identical to monolithic greedy in a single process — if the math breaks, the build goes red. a separate e2e smoke test verifies the running cluster produces sensible output.
- if a worker dies mid-request, the coordinator reshards layers across survivors and retries
- sse streaming endpoint emits each token with the worker that decoded it
- dashboard renders tokens live, color-coded by worker
- prometheus `/metrics` for throughput, per-worker latency, layer counts

## system overview

```mermaid
flowchart TB
    subgraph clients
        direction LR
        cli[cli]
        dashboard[react dashboard :5173]
    end

    subgraph coordsvc["coordinator service"]
        direction LR
        coordinator["fastapi :8000<br/>tokenizer + sampler"]
        scheduler[scheduler]
        coordinator <--> scheduler
    end

    subgraph pool["worker pool"]
        direction LR
        worker1["worker 1 :8001<br/>layers 0–7"]
        worker2["worker 2 :8002<br/>layers 8–14"]
        worker3["worker 3 :8003<br/>layers 15–21"]
    end

    cli <-->|"post /api/infer"| coordinator
    dashboard <-->|"post /api/infer/stream"| coordinator

    scheduler -->|assigns layers| worker1
    worker1 -->|hidden states| worker2
    worker2 -->|hidden states| worker3
    worker3 -->|next-token logits| coordinator
```

## how one token is generated

the coordinator owns the loop. each output token costs one full pass through the chain: embed on the first worker, transit middle workers, finalize on the last worker, sample, append, repeat.

```mermaid
sequenceDiagram
    autonumber
    participant c as coordinator
    participant w1 as worker 1<br/>(layers 0–7)
    participant w2 as worker 2<br/>(layers 8–14)
    participant w3 as worker 3<br/>(layers 15–21)

    c->>w1: input_ids + is_first=true
    note right of w1: embed → run layers 0–7
    w1-->>c: hidden_states
    c->>w2: hidden_states
    note right of w2: run layers 8–14
    w2-->>c: hidden_states
    c->>w3: hidden_states + is_last=true
    note right of w3: run layers 15–21 →<br/>norm → lm_head
    w3-->>c: logits[last]
    note over c: sample next token<br/>(greedy if deterministic)<br/>append, loop
```

## how layers are sharded

the uniform strategy gives every worker a contiguous chunk of the model's transformer blocks. with tinyllama's 22 layers and 3 workers, that's `w1:0–7`, `w2:8–14`, `w3:15–21`. remainder layers (when the count doesn't divide evenly) go to the earliest workers. interleaved assignment is forbidden — block n+1 depends on block n, so non-contiguous chunks would break the forward pass. the capacity strategy weights chunk size by `cpu_cores + memory_mb/1024`; chunks are still contiguous.

## what happens when a worker dies

the coordinator catches the failed call, evicts the worker from the scheduler, recomputes contiguous chunks across the survivors, and restarts the current pipeline pass with the new topology. the stream emits a `reshard` event so the dashboard can show the moment it happened.

```mermaid
sequenceDiagram
    autonumber
    participant c as coordinator
    participant w1 as worker 1
    participant w2 as worker 2 💀
    participant w3 as worker 3

    c->>w1: input_ids (layers 0–7)
    w1-->>c: hidden_states
    c->>w2: hidden_states (layers 8–14)
    w2--xc: connection refused
    note over c: catch, evict w2,<br/>reshard to w1:0–10, w3:11–21
    note over c: emit sse "reshard"
    c->>w1: input_ids (layers 0–10)
    w1-->>c: hidden_states
    c->>w3: hidden_states (layers 11–21, is_last)
    w3-->>c: logits
    note over c: sample, continue loop
```

## verifying correctness

```bash
pytest tests/test_scheduler.py    # layer math
pytest tests/test_parity.py       # sharded vs monolithic greedy (no http)
pytest tests/test_e2e.py          # against a running cluster
```

ci runs all three on every push to `main`.

## demo: failure recovery

```bash
# terminal 1
docker compose up

# terminal 2 — start a long stream
curl -N -X POST localhost:8000/api/infer/stream \
  -H 'content-type: application/json' \
  -d '{"prompt": "once upon a time", "max_tokens": 80}'

# terminal 3 — kill a worker mid-stream
curl -X DELETE localhost:8000/api/workers/worker-2
```

the stream continues. a `reshard` event marks the topology change.

## api

| method | endpoint | notes |
|---|---|---|
| `POST` | `/api/infer` | blocking; returns full result + worker trace |
| `POST` | `/api/infer/stream` | sse; events: start, token, reshard, done, error |
| `GET` | `/api/workers` | workers + scheduler stats |
| `POST` | `/api/workers/register` | worker registration; response carries layer assignment |
| `POST` | `/api/workers/heartbeat` | worker heartbeat |
| `DELETE` | `/api/workers/{id}` | evict a worker (for demoing failure recovery) |
| `GET` | `/api/jobs` | recent job history with reshard events |
| `GET` | `/api/health` | liveness |
| `GET` | `/metrics` | prometheus exposition format |

## tech stack

| layer | tech |
|---|---|
| coordinator | fastapi, httpx, pytorch (tokenizer + sampler) |
| worker | fastapi, pytorch, transformers (any causal lm) |
| cli | click, rich |
| dashboard | react 18, typescript, vite, tailwind |
| orchestration | docker compose |
| model | tinyllama-1.1b-chat (1.1b, 22 layers) by default; swap via `MODEL_NAME` |
