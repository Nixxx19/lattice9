from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn


def shard_model_inplace(
    model,
    assigned_layers: list[int],
    is_first: bool,
    is_last: bool,
) -> None:
    """Drop parameters this worker won't use. Blocks outside `assigned_layers`
    become Identity placeholders so indexed access stays valid."""

    for i in range(len(model.transformer.h)):
        if i not in assigned_layers:
            model.transformer.h[i] = nn.Identity()

    if not is_first:
        model.transformer.drop = nn.Identity()

    if not is_first and not is_last:
        model.transformer.wte = None
        model.transformer.wpe = None

    if not is_last:
        model.transformer.ln_f = nn.Identity()
        if not is_first:
            model.lm_head = None

    for p in model.parameters():
        if p.is_leaf and not p.requires_grad:
            continue
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
