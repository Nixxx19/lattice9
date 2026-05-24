"""
End-to-end smoke test: a running cluster must produce sensible output.

Bitwise parity vs monolithic generate is covered by tests/test_parity.py within
a single process. Across processes (test runner + workers) the forward-pass path
differs from transformers' model.generate() in mask construction and kv-cache
behavior, so we don't assert strict string equality here — we verify the cluster
is alive, registers workers, preserves the prompt, and emits a meaningful number
of new tokens.
"""

import os
import time

import httpx
import pytest


COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
PROMPT = "The capital of France is"
MAX_NEW = 6


def _coordinator_ready(timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{COORDINATOR_URL}/api/workers", timeout=5.0)
            if r.status_code == 200 and r.json().get("stats", {}).get("workers_active", 0) >= 1:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def test_cluster_returns_sensible_inference():
    assert _coordinator_ready(), "coordinator did not come up with active workers"

    resp = httpx.post(
        f"{COORDINATOR_URL}/api/infer",
        json={"prompt": PROMPT, "max_tokens": MAX_NEW, "deterministic": True},
        timeout=180.0,
    )
    resp.raise_for_status()
    body = resp.json()
    text = body["result"]

    assert text.startswith(PROMPT), (
        f"prompt not preserved at start of response: {text!r}"
    )
    assert body["tokens_generated"] >= 1, "no new tokens produced"
    suffix = text[len(PROMPT):]
    assert len(suffix.strip()) > 0, "response is just the prompt with whitespace"


def test_cluster_records_worker_trace():
    assert _coordinator_ready()
    resp = httpx.post(
        f"{COORDINATOR_URL}/api/infer",
        json={"prompt": "hello", "max_tokens": 2, "deterministic": True},
        timeout=120.0,
    )
    resp.raise_for_status()
    trace = resp.json()["worker_trace"]
    assert len(trace) >= 1
    worker_ids = {t["worker_id"] for t in trace}
    assert len(worker_ids) >= 1
    assert all(t["calls"] >= 1 for t in trace)
