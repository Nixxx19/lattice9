"""
Mathematical parity: greedy decode through a virtual sharded pipeline must
produce the same token sequence as monolithic greedy decode.
"""

import pytest
import torch

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not HAVE_TRANSFORMERS:
        pytest.skip("transformers not installed")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
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


def _run_chunk(model, hidden_states: torch.Tensor, layer_indices: list[int]) -> torch.Tensor:
    for idx in layer_indices:
        hidden_states = model.transformer.h[idx](hidden_states)[0]
    return hidden_states


def sharded_greedy(model, tokenizer, prompt: str, max_new_tokens: int, shards: list[list[int]]) -> list[int]:
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    eos = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long)
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            hidden = model.transformer.wte(input_ids) + model.transformer.wpe(positions)
            hidden = model.transformer.drop(hidden)

            for shard in shards:
                hidden = _run_chunk(model, hidden, shard)

            hidden = model.transformer.ln_f(hidden)
            logits = model.lm_head(hidden[:, -1, :])
            next_token = int(torch.argmax(logits, dim=-1).item())

        if next_token == eos:
            break
        tokens.append(next_token)

    return tokens


@pytest.mark.parametrize("shards", [
    [list(range(12))],
    [list(range(6)), list(range(6, 12))],
    [list(range(4)), list(range(4, 8)), list(range(8, 12))],
    [list(range(3)), list(range(3, 6)), list(range(6, 9)), list(range(9, 12))],
])
def test_sharded_matches_monolithic(model_and_tokenizer, shards):
    model, tokenizer = model_and_tokenizer
    prompt = "The future of artificial intelligence is"
    max_new = 15

    mono = monolithic_greedy(model, tokenizer, prompt, max_new)
    shard = sharded_greedy(model, tokenizer, prompt, max_new, shards)

    prompt_len = len(tokenizer(prompt)["input_ids"])
    assert mono[prompt_len:] == shard[prompt_len:]
