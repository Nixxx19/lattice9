"""
Mathematical parity: greedy decode through a virtual sharded pipeline must
produce the same token sequence as monolithic greedy decode.
"""

import pytest
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not HAVE_TRANSFORMERS:
        pytest.skip("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def monolithic_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return output[0].tolist()


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(m, diagonal=1).unsqueeze(0).unsqueeze(0)


def _run_chunk(model, hidden_states, layer_indices, position_ids, attention_mask):
    for idx in layer_indices:
        layer = model.model.layers[idx]
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )[0]
    return hidden_states


def sharded_greedy(model, tokenizer, prompt: str, max_new_tokens: int, shards: list[list[int]]) -> list[int]:
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    eos = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            hidden = model.model.embed_tokens(input_ids)
            attention_mask = _causal_mask(seq_len, hidden.dtype)

            for shard in shards:
                hidden = _run_chunk(model, hidden, shard, position_ids, attention_mask)

            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden[:, -1, :])
            next_token = int(torch.argmax(logits, dim=-1).item())

        if next_token == eos:
            break
        tokens.append(next_token)

    return tokens


def _shards_for(n_layers: int, n_workers: int) -> list[list[int]]:
    base = n_layers // n_workers
    rem = n_layers % n_workers
    out, start = [], 0
    for i in range(n_workers):
        count = base + (1 if i < rem else 0)
        out.append(list(range(start, start + count)))
        start += count
    return out


@pytest.mark.parametrize("n_workers", [1, 2, 3, 4])
def test_sharded_matches_monolithic(model_and_tokenizer, n_workers):
    model, tokenizer = model_and_tokenizer
    prompt = "The capital of France is"
    max_new = 8

    n_layers = len(model.model.layers)
    shards = _shards_for(n_layers, n_workers)

    mono = monolithic_greedy(model, tokenizer, prompt, max_new)
    shard = sharded_greedy(model, tokenizer, prompt, max_new, shards)

    prompt_len = len(tokenizer(prompt)["input_ids"])
    assert mono[prompt_len:] == shard[prompt_len:]
