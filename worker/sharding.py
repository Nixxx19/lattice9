from __future__ import annotations

import gc

import torch.nn as nn


def shard_model_inplace(
    model,
    assigned_layers: list[int],
    is_first: bool,
    is_last: bool,
) -> None:
    for i in range(len(model.model.layers)):
        if i not in assigned_layers:
            model.model.layers[i] = nn.Identity()

    if not is_first:
        model.model.embed_tokens = None

    if not is_last:
        model.model.norm = nn.Identity()
        if not is_first:
            model.lm_head = None

    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    gc.collect()


def model_memory_mb(model) -> float:
    bytes_total = 0
    seen: set[int] = set()
    for p in model.parameters():
        if id(p.data) in seen:
            continue
        seen.add(id(p.data))
        bytes_total += p.numel() * p.element_size()
    return bytes_total / (1024 * 1024)
