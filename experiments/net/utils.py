"""Copied as is from orbit-final/rl8_code/core/utils.py (build_mlp only).

One helper keeps all network blocks uniform and lets each entity type
project from its raw dimension into the shared d_model.
"""
from __future__ import annotations

from typing import Sequence, Type

import torch.nn as nn


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: Type[nn.Module] = nn.GELU,
    norm: bool = True,
    dropout: float = 0.0,
    out_norm: bool = False,
) -> nn.Sequential:
    """Build a feed-forward MLP ``in_dim -> *hidden_dims -> out_dim``.

    Hidden layers: ``Linear -> [LayerNorm] -> activation -> [Dropout]``.
    The final layer is a bare ``Linear`` (optionally followed by ``LayerNorm`` if
    ``out_norm`` is set, e.g. for token embeddings that go into attention).

    One helper keeps all network blocks uniform (same activation, normalization
    and dropout) and lets each entity type project from its raw feature
    dimension into the shared ``d_model`` used by the transformer.
    """
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        last = i == len(dims) - 2
        layers.append(nn.Linear(a, b))
        if not last:
            if norm:
                layers.append(nn.LayerNorm(b))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
        elif out_norm:
            layers.append(nn.LayerNorm(b))
    return nn.Sequential(*layers)
