# Adapted from pokemoni-pobeda/agents/transformer/model/backbone.py.
#
# ⚠ ONE CHANGE VS THE ORIGINAL, made on a measurement, not on taste:
# nn.MultiheadAttention replaced by explicit q/k/v projections + F.scaled_dot_product_attention.
# Measured on MPS: nn.MultiheadAttention 15.6 ms vs 2.0 ms for plain SDPA of the same
# shape (B=64, S=225, D=128, 4 heads) -- 8x wrapper overhead.
# Exactly this trick is already used in the same source repository, in
# model/decision_head.py (CrossAttentionDecisionHead.forward), so the style
# does not diverge from the original.
#
# key_padding_mask (True = ignore) is built in model.py; the relative attention
# shift (model.RelBias) is built there too.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SelfAttention(nn.Module):
    """Self-attention on F.scaled_dot_product_attention, without nn.MultiheadAttention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.p = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """attn_mask is the ALREADY assembled additive shift [B,1,1,S] or [B,H,S,S]
        (padding as -inf plus the relative shift). Built once in
        Backbone.forward and reused by all layers: at [B,H,S,S] it
        weighs 224 MB at batch 512, and building it per layer would mean paying
        for it three times."""
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q, k, v = self.qkv(x).view(B, S, 3, H, Dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.p if self.training else 0.0)
        return self.out_proj(o.transpose(1, 2).reshape(B, S, D))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, attn_mask=attn_mask))
        h2 = self.norm2(x)
        x = x + self.dropout(self.ffn(h2))
        return x


class Backbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None,
                attn_bias: torch.Tensor = None) -> torch.Tensor:
        """attn_bias is the relative shift [B,H,S,S] (model.RelBias) or None.
        Padding is folded into it here, as one tensor for all layers."""
        B, S, _ = x.shape
        mask = None
        if attn_bias is not None:
            mask = attn_bias.to(x.dtype)
            if key_padding_mask is not None:
                mask = mask.masked_fill(key_padding_mask[:, None, None, :],
                                        float("-inf"))
        elif key_padding_mask is not None:
            mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            mask = mask.masked_fill(key_padding_mask[:, None, None, :],
                                    float("-inf"))
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return self.final_norm(x)
