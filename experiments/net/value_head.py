r"""Critic: the value tower, its distributions and the reward scale (R1).

EVERYTHING that sets the value scale lives here -- the head, the training target, and
the PPO reward formula. Keeping them in different files is not allowed: a critic that learns
on `tanh(Δmoney/σ)` and a trainer that issues the reward with a different σ would give an
advantage in the wrong scale, and no measurement would show it (the same class
of bug as `legal`/`legal_now`).

**What was there before R1.** `ValueHead` -- a PMA with one seed over `h.detach()` of all
trunk tokens and an MLP over three classes W/D/L. A measurement (internal review, N5)
showed what it actually relies on: permuting the opponent's field across the batch
shifts P(win) by **0.132**, permuting money -- by **0.026**. On the ladder
the opponent's field is a fingerprint of its strength, and the outcome is guessed from it; in self-play
against a copy of itself this feature means nothing. AUC by day 0.67 -> 0.85-0.89
vs 0.46-0.75 for the money difference alone.

**What R1 does.**

1. **Its own tower** (`ValueTower`). Reads `h.detach()`, but not only it: the hundred
   tokens of the OPPONENT's tiles go into it whole, through its own encoder. The
   `opp_k = 4` bottleneck stays with the policy -- it is trained by the policy gradient and loses the opponent's
   animals (R² 0.59-0.75 vs 1.0 for the raw sum, N6). The tower input
   is detached ENTIRELY, so the policy does not change by a single bit because of the critic, and this
   is a property of the construction, not of carefulness.
2. **Critic-only columns** (`features.value_extra`): an estimate of the opponent's
   stock, its purchases, the pace of its money. They do not go to the policy -- the gain for
   actions there is zero (internal architecture audit), and with a batch of value
   features the policy had already lost $16,042 (backlog, "Do not do").
3. **Two heads.** W/D/L, as it was, and HL-Gauss (arXiv:2403.03950) over the
   normalized final bank difference. One gives the outcome, the other the gradation
   between rollouts, which the terminal ±1 lacks (trap 1 from
   an internal community review).

`ValueHead` is kept NOT as a second path to the output: it is no longer in the network. It
is needed to read the `value.*` weights from BC checkpoints -- for the warm start
of the tower and for comparing the old critic with the new one on the same states
(an internal critic evaluation script).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from backbone import Backbone
from utils import build_mlp


class ValueHead(nn.Module):
    """Critic BEFORE R1. Not part of `FarmNet`: only for reading old weights and comparison.

    PMA pooling (k=1), see the internal set-transformer architecture note, section 5.
    `key_padding_mask` is the same one that `FarmNet.forward` builds.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),          # win / draw / loss
        )

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # h: [B, 1+N, D]; key_padding_mask: [B, 1+N] bool (True = padding). -> [B,3]
        B = h.shape[0]
        seed = self.seed.expand(B, -1, -1)                        # [B, 1, D]
        pooled, _ = self.attn(seed, h, h, key_padding_mask=key_padding_mask,
                              need_weights=False)                 # [B, 1, D]
        pooled = self.norm(pooled.squeeze(1))                     # [B, D]
        return self.mlp(pooled)                                   # [B, 3] logits


# --- value and reward scale -------------------------------------------
#
# σ -- from a measurement on 200 games of teacher 56263693 (the replay's `rewards`): median
# |Δbanks| $5,244, p75 $12,761, std $23,373, 92% of games fit in $30k.
# At σ = $25,000 tanh is almost linear on 70% of games (|Δ|/σ < 0.4) and saturates
# only on routs -- i.e. there is gradation where games actually
# differ, and the tail does not pull the loss onto itself.
RL_SIGMA = 25000.0
# α -- the share of gradation in the reward. The leaderboard computes skill rating by OUTCOME, not
# by money, so ±1 must remain the main term: at α = 0.3
# the maximum addition is 0.3 against a jump of 2.0 at the sign change, i.e. a win by
# $1 is never worse than a loss by $100k. Below 0.1 the gradation drowns in the noise
# of the advantage estimate, above 0.5 the agent starts to prefer "losing
# expensively" to a draw.
#
# There is deliberately no flag for α in `bc_train`: the BC stage does NOT depend on it. The HL head
# learns the distribution of `tanh(Δ/σ)` as a whole, and α only weights its expectation in
# `value_scalar` and in `terminal_reward` -- i.e. it enters only in the trainer,
# and can be tuned there without retraining the critic. Had we added a flag here, it would
# have created a second place where one constant lives, with exactly the same
# consequences as `coord_fix` in the submission package.
RL_ALPHA = 0.3
# HL-Gauss: 51 bins on [-1, 1] (step ζ = 0.04) and smoothing σ_hl = 0.75·ζ --
# the ratio σ/ζ = 0.75 from arXiv:2403.03950 (§5.1, plateau 0.5-1.0).
HL_BINS = 51
HL_SIGMA_RATIO = 0.75


def hl_edges(device=None, dtype=torch.float32) -> torch.Tensor:
    """Bin edges, [HL_BINS + 1]."""
    return torch.linspace(-1.0, 1.0, HL_BINS + 1, device=device, dtype=dtype)


def hl_centers(device=None, dtype=torch.float32) -> torch.Tensor:
    e = hl_edges(device, dtype)
    return 0.5 * (e[:-1] + e[1:])


def hl_target(z: torch.Tensor, sigma_ratio: float = HL_SIGMA_RATIO) -> torch.Tensor:
    """HL-Gauss target: mass of the normal N(z, σ) over bins. [B] -> [B, HL_BINS].

    Regression-as-classification (arXiv:2403.03950): instead of MSE on a scalar --
    cross-entropy against a histogram onto which a normal around the true
    value is smeared. The gain is where we need it: a heavy-tailed target and a
    network critic that collapses to the mean under MSE (we already had this --
    v1 emitted the constant 0.022 on any input).

    Normalizing by the mass inside [-1, 1] is mandatory: without it, for values near the
    boundary (a rout) the total mass is below one, and such games would silently
    weigh less in the loss than the others.
    """
    z = z.float().clamp(-1.0, 1.0).unsqueeze(-1)                  # [B,1]
    sigma = sigma_ratio * (2.0 / HL_BINS)
    e = hl_edges(z.device, z.dtype).unsqueeze(0)                  # [1,K+1]
    cdf = 0.5 * (1.0 + torch.erf((e - z) / (sigma * math.sqrt(2.0))))
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


def hl_expect(logits: torch.Tensor) -> torch.Tensor:
    """Scalar estimate from histogram logits: Σ p_i · center_i. [B,K] -> [B]."""
    p = torch.softmax(logits.float(), -1)
    return (p * hl_centers(logits.device, torch.float32)).sum(-1)


def terminal_reward(money_me, money_op, alpha: float = RL_ALPHA,
                    sigma: float = RL_SIGMA):
    """PPO reward at the end of the game: ±1 for the outcome plus α·tanh(Δmoney/σ).

    The only place where this formula is written. The critic learns on the same
    `tanh(Δ/σ)` (HL head), so `value_scalar` and this function measure the same thing.
    """
    d = float(money_me) - float(money_op)
    sign = 0.0 if d == 0 else (1.0 if d > 0 else -1.0)
    return sign + alpha * math.tanh(d / sigma)


def value_scalar(wdl_logits: torch.Tensor, hl_logits: torch.Tensor,
                 alpha: float = RL_ALPHA) -> torch.Tensor:
    """V(s) in the scale of `terminal_reward`: (P(W) − P(L)) + α·E[tanh(Δ/σ)].

    GAE is then computed from exactly this: both heads describe one reward, not
    two different quantities.
    """
    p = torch.softmax(wdl_logits.float(), -1)
    return (p[..., 0] - p[..., 2]) + alpha * hl_expect(hl_logits)


def gae(rewards, values, last_value, gamma: float = 1.0, lam: float = 0.95):
    """TD(λ) advantages and value targets. A place for the PPO trainer (R1 item 4).

    Not called in the BC stage: there the target is computed from the replay (the final bank
    difference, `bc_train`). Under PPO our reward is TERMINAL, so `rewards`
    are zeros everywhere except the last step, and γ = 1 by default: the horizon is fixed
    (720 turns), and discounting with a terminal reward just multiplies the target by a
    known function of time, which the critic would learn as an "hour" feature, not
    as a state estimate. Credit compression is given by λ, not γ.

    Returns (advantages, returns) with the same shapes as `values`.
    """
    T = len(rewards)
    adv = [0.0] * T
    gae_t = 0.0
    for t in range(T - 1, -1, -1):
        v_next = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        gae_t = delta + gamma * lam * gae_t
        adv[t] = gae_t
    return adv, [a + v for a, v in zip(adv, values)]


class ValueTower(nn.Module):
    r"""A separate value tower: its own encoder of the opponent's field, its own trunk, two heads.

    Input:
      * `h.detach()` -- policy trunk tokens (own tiles, K opponent latents,
        units, goods, crops, animals, global);
      * a hundred tokens of the OPPONENT's tiles WHOLE, through its own `enc_tile`. The
        policy bottleneck squeezes them into 4 latents, and that is not enough for the critic (N6);
      * one token with critic-only columns (`features.value_extra`).

    There is NO relative attention shift inside the tower, and this is not thrift: from the
    opponent's field the critic needs SUMS (how many cows it has, how much is ripe), not
    geometry -- the policy reads that on its own farm. Tile coordinates are in
    its features anyway (columns 21-22), so the tower can reconstruct them itself if needed.

    The whole input is detached HERE, not by the caller: then "the critic does not touch the
    policy" is a property of this class and cannot be lost by editing the model's
    `forward`.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 tile_dim: int, crit_dim: int, enc_hidden: int,
                 head_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.enc_tile = build_mlp(tile_dim, [enc_hidden], d_model, out_norm=True)
        self.enc_crit = build_mlp(crit_dim, [enc_hidden], d_model, out_norm=True)
        self.type_op = nn.Parameter(torch.zeros(d_model))
        self.type_crit = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.type_op, std=0.02)
        nn.init.normal_(self.type_crit, std=0.02)
        self.trunk = Backbone(d_model, n_layers, n_heads, d_ff, dropout)
        # A PMA of the same shape as the former ValueHead: then its weights transfer
        # into the tower one to one (`init_from_bc`), rather than by a "similar initialization".
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(                      # win / draw / loss
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.hl = nn.Sequential(                       # histogram of tanh(Δ/σ)
            nn.Linear(d_model, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, HL_BINS))

    @torch.no_grad()
    def init_from_bc(self, sd: dict, n_bc_layers: int) -> list:
        """Warm start from a BC checkpoint. Returns the list of transferred groups.

        Transferred is what means the same for the tower and for the policy: the tile encoder,
        the opponent tile type, the first trunk layers and -- if the checkpoint has the old
        `value.*` -- the whole PMA with the W/D/L head. The critic columns and the HL head
        have no counterpart in BC and stay random.
        """
        took = []
        pref = {k: v for k, v in sd.items()}

        def copy_group(dst: nn.Module, src_prefix: str, name: str):
            sub = {k[len(src_prefix):]: v for k, v in pref.items()
                   if k.startswith(src_prefix)}
            if not sub:
                return
            cur = dst.state_dict()
            if set(sub) != set(cur) or any(sub[k].shape != cur[k].shape for k in cur):
                return
            dst.load_state_dict(sub)
            took.append(name)

        copy_group(self.enc_tile, "enc_tile.", "enc_tile")
        if "type_tile_op" in pref and pref["type_tile_op"].shape == self.type_op.shape:
            self.type_op.copy_(pref["type_tile_op"]); took.append("type_tile_op")
        for i in range(min(len(self.trunk.layers), n_bc_layers)):
            copy_group(self.trunk.layers[i], f"backbone.layers.{i}.", f"trunk[{i}]")
        copy_group(self.trunk.final_norm, "backbone.final_norm.", "trunk.norm")
        if "value.seed" in pref and pref["value.seed"].shape == self.seed.shape:
            self.seed.copy_(pref["value.seed"]); took.append("PMA.seed")
        copy_group(self.attn, "value.attn.", "PMA.attn")
        copy_group(self.norm, "value.norm.", "PMA.norm")
        copy_group(self.mlp, "value.mlp.", "W/D/L head")
        return took

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor,
                tiles_op: torch.Tensor, vcrit: torch.Tensor):
        """h: [B,S,D]; key_padding_mask: [B,S]; tiles_op: [B,100,F]; vcrit: [B,Fv].

        -> (W/D/L logits [B,3], histogram logits [B,HL_BINS])."""
        h = h.detach()
        B = h.shape[0]
        tok_op = self.enc_tile(tiles_op.detach()) + self.type_op        # [B,100,D]
        tok_c = (self.enc_crit(vcrit.detach()) + self.type_crit).unsqueeze(1)
        x = torch.cat([h, tok_op, tok_c], dim=1)
        zeros = torch.zeros(B, tok_op.shape[1] + 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([key_padding_mask, zeros], dim=1)
        z = self.trunk(x, key_padding_mask=pad)
        seed = self.seed.to(z.dtype).expand(B, -1, -1)
        pooled, _ = self.attn(seed, z, z, key_padding_mask=pad, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        # The pooling vector of the last pass -- for the scalar head of the loop's critic
        # (`rl/critic.ScalarTowerCritic`). It has no weights of its own in the network,
        # so the state_dict and the tower outputs do not change by a single bit.
        self.last_pooled = pooled
        return self.mlp(pooled), self.hl(pooled)
