"""
End-to-end: a running cluster (started by CI via docker compose) must produce
the same deterministic output as a locally-computed monolithic greedy decode.
"""

import os
import time

import httpx
import pytest

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAVE_HF = True
except ImportError:
    HAVE_HF = False


COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8000")
PROMPT = "The quick brown fox"
MAX_NEW = 12


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


@pytest.mark.skipif(not HAVE_HF, reason="transformers not installed")
def test_distributed_matches_monolithic_greedy():
    assert _coordinator_ready(), "coordinator did not come up with active workers"

    resp = httpx.post(
        f"{COORDINATOR_URL}/api/infer",
        json={"prompt": PROMPT, "max_tokens": MAX_NEW, "deterministic": True},
        timeout=180.0,
    )
    resp.raise_for_status()
    distributed_text = resp.json()["result"]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    monolithic_text = tokenizer.decode(out[0], skip_special_tokens=True)

    assert distributed_text == monolithic_text, (
        f"\nmonolithic:  {monolithic_text!r}\n"
        f"distributed: {distributed_text!r}"
    )
