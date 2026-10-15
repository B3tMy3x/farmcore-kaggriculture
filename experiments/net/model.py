r"""Policy/value network for Kaggriculture: a BERT-style head for the whole turn.

The scheme is copied from orbit-final/rl8_code/model.py (per-type encoders -> shared
token set -> set-invariant transformer -> heads directly on tokens), the transformer
blocks and the value head are taken as is from pokemoni-pobeda.

   own tiles    opp tiles     units   goods     crops   animals  global (29)
       |           |          |       |        |        |        |
       |     bottleneck: self-attention                          |
       |     + PMA into K=4 latents                              |
     build_mlp encoders (each -> d_model) + learned type embedding
       \______________ concat into a single token set ______________/
                               |
                    Backbone (pre-norm, SwiGLU, set-invariant)
                    + relative attention shift (RelBias)
                               |
          first pass: heads on trunk tokens (input to the corrector only)
                               |
          corrector: units, goods, crops, animals, global plus their
          distributions and derived quantities (intent, E[n], shed at the moment of
          the market, spend, revenue); units -> own tiles via cross-attention; 2 layers
                               |
    result: THE SAME heads on the corrector latent
    verb + q·k to the object, target pointer scores[unit, tile], sell/buy
    per good, seeds and animals from their own tokens, hire and land from the global;
    effects and economy come from the trunk; value is a separate tower (R1):
    h.detach() + the hundred opponent tiles whole + critic columns -> 2 layers -> PMA
    -> W/D/L and an HL-Gauss histogram over the final bank difference

A unit action is "verb + pointer to an object", not an index in a flat softmax
of 40 classes; seeds and animals are decided from their own tokens, not by two heads
from the global one (there were 500 and 300 outputs). The action class set stays THE SAME:
the parametrization changed, not the space (internal note on the composite action,
completeness check by an internal expressiveness script).

The flat head and the heads from the global token were REMOVED rather than left behind a flag:
the measurement is closed (head-to-head 114/192, p = 0.011), and two paths to one output are
a silent train/play desync. The old path lives in git history and
inside already built submissions.

**Key difference from autoregression:** all turn decisions are emitted in ONE pass.
Each unit token emits its own action, each good token its own
market decision. This is BERT mode: positions predict in parallel rather than
in a chain. The price is that the explicit order of market orders is lost (a sale must come
before the purchase it finances), so the order is imposed
deterministically when assembling orders: sells first, then buys.

The target pointer (scores[unit, tile]) is the orbit from/to construction:
the unit picks the tile it goes to work on and holds onto it until it arrives.
Exactly what the prototype lacked, with 42% of its actions spent on walking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields
from typing import Dict, Optional

import torch
import torch.nn as nn

from backbone import Backbone
from utils import build_mlp
from value_head import ValueTower, HL_BINS

# --- raw feature dimensions (see features.py) ---
# 32 original + crop one-hot (5) + animal one-hot (3)
TILE_FEAT_DIM = 40
UNIT_FEAT_DIM = 33          # 27 base + 6 for the unit's previous action (28.09)
UNIT_BASE_DIM = 27
PRODUCT_FEAT_DIM = 18
# 26 base columns plus three P8 features (features.py, measured 22.09): plants dying
# tonight, feed deficit, decaying crop. The net reads all 29; the "night
# shed surplus" was removed at freeze -- it is always 0 in our games.
GLOBAL_FEAT_DIM = 32          # 29 base + 3 cash reserve (27.09)
GLOBAL_BASE_DIM = 29
# Crop and animal tokens (features.py). They did not exist at all: the five decisions
# "how many seeds to buy" and three "how many animals" were emitted by one global
# token through heads with 500 and 300 outputs, and in unit actions the item was
# baked into the class number.
CROP_FEAT_DIM = 18
ANIMAL_FEAT_DIM = 18
# Columns ONLY for the critic (R1, features.value_extra): an estimate of the opponent's
# stock, its purchases, the money pace of both sides. They do not go to the policy --
# none of these quantities is in the tokens above. The dimension lives here for the same
# reason as the others: features.py imports model.py.
VCRIT_FEAT_DIM = 32

# 11 base jobs + PICKUP for each of 9 goods and 3 animals + PLACE for
# each animal + PLANT for each of 5 crops. No directions: they
# are computable from the target (internal movement-factorization note).
N_OPS = 11 + 9 + 3 + 3 + 9 + 5      # +9: PLACE_<good>, addressed drop into the shed
# Unit target: 100 tiles of its own farm plus a "stay" column -- as many classes
# as head_to has columns (`target_logits`, [B,U,T+1]).
N_TARGET = 100 + 1
                    # (internal movement-factorization note), the mask is in masks.py
# Quantity heads -- a gate plus a conditional CORN chain (see counts.py):
# one output "act or not" and K-1 conditional "continue the count", instead of
# a softmax over exact numbers, which gave recall 0.04 and required class weights.
from counts import n_logits, split as split_counts, expect as expect_counts  # noqa: E402
from counts import ALL_CAPS as ALL_CAPS_  # noqa: E402
N_SELL_BINS = n_logits("sell")     # gate + 99 conditional + "whole stock" flag
N_BUY_BINS = n_logits("buy")
N_HIRE_BINS = n_logits("hire")
N_SEED_BINS = n_logits("seed")
N_CROPS = 5
from effects import N_EFFECTS     # 9 jobs + 3 placements + 5 plantings
from effects import ECON_DIM      # money + market stock (effects.py)
# Hand ceiling. Lives HERE, not in features.py, because features.py
# imports this module, not the other way around; features re-exports it so
# there are not two numbers that can be silently pulled apart.
MAX_UNITS = 24
N_ANIMALS = 3       # GOOSE / COW / SHEEP
N_ANIMAL_BINS = n_logits("animal")

# Prices for estimating spend inside the coordination block. Taken from the engine,
# not from labeling: these are rules, not teacher habits.
import os as _os, sys as _sys  # noqa: E402
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    CROPS as _CROPS, ANIMALS as _ANIMALS, PRODUCTS as _PRODUCTS,
)
# --- composite action: verb + pointer to an object ---
# The class order MUST match bc_train.UNIT_OPS (there is a check there):
# labeling, mask and decoder are addressed by class number.
BASE_OPS = ["PASS", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE",
            "DIG", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
OP_NAMES = (BASE_OPS
            + [f"PICKUP_{g}" for g in _PRODUCTS] + [f"PICKUP_{a}" for a in _ANIMALS]
            + [f"PLACE_{a}" for a in _ANIMALS] + [f"PLACE_{g}" for g in _PRODUCTS]
            + [f"PLANT_{c}" for c in _CROPS])
VERBS = BASE_OPS + ["PICKUP", "PLACE", "PLANT"]
N_VERBS = len(VERBS)
# Objects are the same tokens that lie in the set: goods, crops, animals.
# The order is fixed and repeats the token order (see forward).
# Names carry a group label: WHEAT is both among goods and among crops, and
# a list without labels in a diagnostics printout would read as one object.
OBJ_NAMES = ([f"good {g}" for g in _PRODUCTS] + [f"seed {c}" for c in _CROPS]
             + [f"animal {a}" for a in _ANIMALS])
N_OBJ = len(OBJ_NAMES)
_V = {v: i for i, v in enumerate(VERBS)}
# The maps are SEPARATE, and this is not pedantry: WHEAT is both among goods and among
# crops, but these are different tokens with different features. `PICKUP WHEAT` takes
# a sack of wheat (a good) from the shed, `PLANT WHEAT` spends a seed (a crop). One shared
# name map would silently glue them into one object.
_O_PROD = {g: i for i, g in enumerate(_PRODUCTS)}
_O_CROP = {c: len(_PRODUCTS) + i for i, c in enumerate(_CROPS)}
_O_ANIM = {a: len(_PRODUCTS) + len(_CROPS) + i for i, a in enumerate(_ANIMALS)}


def _factorize(name):
    """Action class -> (verb index, object index or N_OBJ = "no object")."""
    for v in ("PICKUP", "PLACE"):
        if name.startswith(v + "_"):
            it = name[len(v) + 1:]
            return _V[v], (_O_PROD[it] if it in _O_PROD else _O_ANIM[it])
    if name.startswith("PLANT_"):
        return _V["PLANT"], _O_CROP[name[len("PLANT_"):]]
    return _V[name], N_OBJ


# Item verbs go in their own slots: each has its OWN query to the objects.
# One shared query per unit would glue the preferences: then
# logit(PICKUP X) - logit(PLACE X) would not depend on X, and "carry the melon but
# put down the wheat" would become inexpressible. The flat head could do that, so
# the factorization must be able to as well.
PVERBS = ("PICKUP", "PLACE", "PLANT")
N_PVERB = len(PVERBS)
_PV = {v: i for i, v in enumerate(PVERBS)}
VERB_OF = [_factorize(n)[0] for n in OP_NAMES]
OBJ_OF = [_factorize(n)[1] for n in OP_NAMES]
# Flat cell index in (verb x object); the last index is "no object".
PAIR_OF = [(_PV[n.split("_")[0]] * N_OBJ + o) if o < N_OBJ else N_PVERB * N_OBJ
           for n, o in zip(OP_NAMES, OBJ_OF)]

SEED_COST = [float(_CROPS[c]["seed"]) for c in _CROPS]
ANIMAL_COST = [float(_ANIMALS[a]["cost"]) for a in _ANIMALS]
# `play.py` writes out BUY_PRODUCT only for these two: per the internal I/O health analysis the engine
# does not accept 55.8% of the rest, and an order slot is not spent on them.
BUYABLE = [1.0 if g in ("WHEAT", "FERTILIZER") else 0.0 for g in _PRODUCTS]
MONEY0 = 3000.0     # spend normalization: starting bank, a fixed constant
# indices in glob (features.py): price of the next hand and price of the next quadrant
G_HIRE_COST, G_LAND_PRICE = 24, 25


BOARD = 10          # farm side; offsets dx, dy lie in [-9, 9]
N_SHIFT = 2 * BOARD - 1
# Unit coordinates inside its features (features.py: units[k] = [..., x/10, y/10, ...]).
# They are deliberately NOT passed as a separate tensor: that would add a key to the feature
# cache, and it must not change at all for the relative shift.
U_FX, U_FY = 2, 3
# Unit feature columns (features.py): 5 -- stands on a shed access cell,
# 17..25 -- cargo by good (/20). Needed by the corrector to compute the shed at
# the moment of the market (`FarmNet._shed_at_market`).
U_AT_SHED, U_INV0 = 5, 17
PRODUCTS_ORDER = list(_PRODUCTS)        # order of good tokens = features.PRODUCTS
SHED_CAP_ = 100.0                       # normalization of the "in shed" column of a good token


class RelBias(nn.Module):
    r"""Additive attention shift by relative offset (dx, dy).

    Why. The backbone is set-invariant: it knows where a tile is only from two
    columns x/10 and y/10 inside the features passed through the MLP encoder.
    "Neighbouring cell", "two steps east", "this unit is closer" -- all of this
    it has to derive by arithmetic from absolute coordinates, and only AFTER
    mixing with the other forty tile features. And almost all decisions here are
    geometric: reach, avoid collisions, water the neighbour. The offset is
    the argument they depend on.

    A 19x19 table per head (dx, dy in [-9, 9]) plus a learned scalar per PAIR
    OF TOKEN TYPES: positionless tokens (goods, crops, animals, global)
    have no geometry, but a constant "unit -> good" offset is meaningful and
    costs H*n_types^2 parameters.

    Both tables are initialized with ZEROS: at the start the net is bitwise equal to the model
    without the shift. The same trick as with the output of the corrector's cross-attention,
    and it also makes a warm start from the previous checkpoint a comparison with it, not
    "with another model".

    About memory. Naively this is an index tensor [B, S, S] (224 MB in int64 at
    B=512, S=234) plus a lookup from a table of the same size. So the offsets
    are computed BLOCKWISE: tile<->tile does not depend on the batch at all (cell positions
    are constant) and is computed once over the index buffer, and per batch only
    the unit rows remain -- 24x100 and 24x24 instead of 234x234.
    """

    def __init__(self, n_heads: int, layout, board: int = BOARD):
        """layout: a list of (size, type_id, frame, xy) in token order.

        frame is the coordinate system (own farm and the opponent's differ, and
        an offset between them is meaningless); None = positionless token.
        xy is an [size, 2] array for groups with constant coordinates (tiles),
        None for the unit group: its coordinates arrive on every pass.
        """
        super().__init__()
        self.tab = nn.Parameter(torch.zeros(n_heads, N_SHIFT * N_SHIFT))
        n_types = max(g[1] for g in layout) + 1
        self.tpair = nn.Parameter(torch.zeros(n_heads, n_types, n_types))

        spans, tok_type, off = [], [], 0
        for size, tid, _frame, _xy in layout:
            spans.append((off, off + size))
            tok_type += [tid] * size
            off += size
        self.S = off
        self.spans = spans
        tt = torch.tensor(tok_type, dtype=torch.long)
        self.register_buffer("tt_i", tt.view(-1, 1).expand(self.S, self.S).contiguous(),
                             persistent=False)
        self.register_buffer("tt_j", tt.view(1, -1).expand(self.S, self.S).contiguous(),
                             persistent=False)

        self.static_pairs, self.dyn_pairs = [], []
        self.dyn = None
        for i, (_, _, fi, xyi) in enumerate(layout):
            if xyi is None and fi is not None:
                assert self.dyn is None, "exactly one dynamic group is supported"
                self.dyn = i
            for j, (_, _, fj, xyj) in enumerate(layout):
                if fi is None or fj is None or fi != fj:
                    continue
                if xyi is not None and xyj is not None:
                    idx = ((xyi[:, None, 0] - xyj[None, :, 0] + board - 1) * N_SHIFT
                           + (xyi[:, None, 1] - xyj[None, :, 1] + board - 1))
                    self.register_buffer(f"idx_{i}_{j}",
                                         torch.as_tensor(idx, dtype=torch.long),
                                         persistent=False)
                    self.static_pairs.append((i, j))
                else:
                    self.dyn_pairs.append((i, j))
        for i, (_, _, fi, xyi) in enumerate(layout):
            if xyi is not None and fi is not None:
                self.register_buffer(f"xy_{i}", torch.as_tensor(xyi, dtype=torch.long),
                                     persistent=False)
        self.board = board
        # Indices for the `onehot` path (see `_dyn_onehot`): the shift between a CELL
        # c = x·board + y of the dynamic group and each position of another group.
        # They do not depend on the batch, so their lookup from the table is constant too.
        self.onehot = False
        cells = torch.arange(board * board)
        cxy = torch.stack([cells // board, cells % board], -1)
        for i, j in self.dyn_pairs:
            xyi = cxy if i == self.dyn else getattr(self, f"xy_{i}")
            xyj = cxy if j == self.dyn else getattr(self, f"xy_{j}")
            idx = ((xyi[:, None, 0] - xyj[None, :, 0] + board - 1) * N_SHIFT
                   + (xyi[:, None, 1] - xyj[None, :, 1] + board - 1))
            self.register_buffer(f"cidx_{i}_{j}", idx.long(), persistent=False)

    def _dyn_onehot(self, tab, i, j, cell):
        """The same block as `_gather`, but via a one-hot MATRIX multiplication.

        Why. The forward `tab[:, idx[B,n,m]]` is cheap, but its backward is
        `index_put_` with accumulation over B·H·n·m indices into 361 cells: a sort
        plus `indexing_backward_kernel`, 21-27% of the WHOLE training step time on
        the GPU (profile 23.09, mb 64 and 1024), and the summation order differs there from run
        to run. Here the lookup is split into two parts: the constant
        `P = tab[:, cidx]` (cell x position, independent of the batch) and the selection of the
        row by the unit's cell with a one-hot matrix O. The backward is the GEMM
        `Oᵀ·grad`, deterministic and several times cheaper.

        ⚠ The forward is BITWISE the same: in each sum Σ_p O[p]·P[p] exactly
        one factor is 1, the rest 0, and 1·x + 0 = x exactly in any order.
        This holds only in fp32 without TF32 (TF32 would round P to 10 mantissa
        bits) -- so the path refuses if TF32 is enabled.
        """
        B, U = cell.shape
        H = tab.shape[0]
        C = self.board * self.board
        O = torch.nn.functional.one_hot(cell, C).to(tab.dtype).reshape(B * U, C)
        P = tab[:, getattr(self, f"cidx_{i}_{j}")]                  # [H, a, b]
        if i == self.dyn and j == self.dyn:
            A = (O @ P.permute(1, 0, 2).reshape(C, H * C)).view(B, U, H, C)
            A = A.permute(0, 2, 1, 3).reshape(B, H * U, C)
            return torch.bmm(A, O.view(B, U, C).transpose(1, 2)).view(B, H, U, U)
        if i == self.dyn:                                            # [H, C, m]
            m = P.shape[2]
            out = O @ P.permute(1, 0, 2).reshape(C, H * m)
            return out.view(B, U, H, m).permute(0, 2, 1, 3)
        n = P.shape[1]                                               # [H, n, C]
        out = O @ P.permute(2, 0, 1).reshape(C, H * n)
        return out.view(B, U, H, n).permute(0, 2, 3, 1)

    def _gather(self, tab, xi, yi, xj, yj):
        """tab[:, (dx+b-1)*19 + (dy+b-1)] for coordinates [B,n] and [B,m] -> [B,H,n,m]."""
        idx = ((xi[:, :, None] - xj[:, None, :] + self.board - 1) * N_SHIFT
               + (yi[:, :, None] - yj[:, None, :] + self.board - 1))
        return tab[:, idx].permute(1, 0, 2, 3)

    def forward(self, unit_xy: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """unit_xy: [B,U,2] integer coordinates of the dynamic group. -> [B,H,S,S]."""
        B = unit_xy.shape[0]
        tab = self.tab.to(dtype)
        bias = self.tpair.to(dtype)[:, self.tt_i, self.tt_j]          # [H,S,S]
        bias = bias.unsqueeze(0).expand(B, -1, -1, -1).clone()        # [B,H,S,S]
        for i, j in self.static_pairs:
            a0, a1 = self.spans[i]
            b0, b1 = self.spans[j]
            bias[:, :, a0:a1, b0:b1] += tab[:, getattr(self, f"idx_{i}_{j}")]
        if self.dyn is None:
            return bias
        ux, uy = unit_xy[..., 0], unit_xy[..., 1]
        if self.onehot:
            if torch.backends.cuda.matmul.allow_tf32 and tab.is_cuda:
                raise RuntimeError(
                    "RelBias.onehot with TF32 enabled: one-hot lookup "
                    "by multiplication is exact only in true fp32")
            cell = ux * self.board + uy
            for i, j in self.dyn_pairs:
                a0, a1 = self.spans[i]
                b0, b1 = self.spans[j]
                bias[:, :, a0:a1, b0:b1] += self._dyn_onehot(tab, i, j, cell)
            return bias
        for i, j in self.dyn_pairs:
            a0, a1 = self.spans[i]
            b0, b1 = self.spans[j]
            xi, yi = ((ux, uy) if i == self.dyn else
                      (lambda t: (t[None, :, 0].expand(B, -1),
                                  t[None, :, 1].expand(B, -1)))(getattr(self, f"xy_{i}")))
            xj, yj = ((ux, uy) if j == self.dyn else
                      (lambda t: (t[None, :, 0].expand(B, -1),
                                  t[None, :, 1].expand(B, -1)))(getattr(self, f"xy_{j}")))
            bias[:, :, a0:a1, b0:b1] += self._gather(tab, xi, yi, xj, yj)
        return bias


def tile_xy(board: int = BOARD):
    """Tile coordinates in the same order in which features.encode lays them out."""
    return torch.tensor([[x, y] for y in range(board) for x in range(board)],
                        dtype=torch.long)


@dataclass
class ModelConfig:
    d_model: int = 128
    d_k: int = 64
    n_layers: int = 3
    n_heads: int = 4
    ffn: int = 384
    dropout: float = 0.0
    enc_hidden: int = 128
    head_hidden: int = 128
    # Coordination block (corrector): two attention layers on top of the already
    # computed distributions (P6a, game b2).
    coord_layers: int = 2
    # The opponent's field through a narrow bottleneck: one self-attention layer over its tiles
    # (with its own relative shift) and a PMA into K latents that go into the trunk
    # instead of a hundred tokens (P5, game b3: no worse, x1.28 speed).
    opp_k: int = 4
    # The value tower's own layers (R1). Two -- so that it can sum the hundred
    # opponent tiles into totals and combine them with money; more makes no sense, its input
    # has already passed three layers of the policy trunk.
    value_layers: int = 2
    # Teacher styles (30.09, blend: expert = team embedding). 0 -- no condition.
    # Indices 0..styles-1 are teams, styles is "common" (trained by style-drop).
    styles: int = 0
    # Input options `runway` (cash reserve, 27.09), `styles` (conditional BC,
    # 28.09) and `prev_act` (the unit's previous action, 28.09) were removed 29.09: not taken
    # by money (-$5.9k, multi-teacher BC is worse, -$25k). The columns stay in
    # the cache (features.py is hashed into the fingerprint), the net reads only the first
    # GLOBAL_BASE_DIM / UNIT_BASE_DIM. A checkpoint with any of them enabled is
    # a loud refusal in `from_ckpt`.

    @classmethod
    def scaled(cls, width: int = 1, depth: int = 1, styles: int = 0) -> "ModelConfig":
        """The same architecture, wider/deeper by an integer factor (27.09: big oh3 x2).
        Width multiplies d_model, the pointer key, FFN, encoders, heads and the number of
        attention heads (head size is the same); depth -- trunk, corrector and
        value tower. The number of opponent latents (`opp_k`) is not width, we do not touch it."""
        b = cls()
        return cls(d_model=b.d_model * width, d_k=b.d_k * width, n_layers=b.n_layers * depth,
                   n_heads=b.n_heads * width, ffn=b.ffn * width, dropout=b.dropout,
                   enc_hidden=b.enc_hidden * width, head_hidden=b.head_hidden * width,
                   coord_layers=b.coord_layers * depth, opp_k=b.opp_k,
                   value_layers=b.value_layers * depth, styles=styles)

    @classmethod
    def from_ckpt(cls, sd) -> "ModelConfig":
        """Config from the checkpoint (`model_cfg`), otherwise the default -- all checkpoints before
        27.09 were made with the default, and they load as before, strictly."""
        mc = sd.get("model_cfg") if isinstance(sd, dict) else None
        if not mc:
            return cls()
        names = {f.name for f in fields(cls)}
        on = [k for k in _REMOVED_OPTS if mc.get(k)]
        if on:
            raise RuntimeError(f"checkpoint was trained with options {on} removed on 29.09 "
                               "(not taken by money); the code is a copy of an internal experiment branch")
        unknown = set(mc) - names - set(_REMOVED_OPTS)
        if unknown:
            raise RuntimeError(f"checkpoint model_cfg carries unknown fields {sorted(unknown)}")
        return cls(**{k: v for k, v in mc.items() if k in names})


# ModelConfig fields removed on 29.09. In `model_cfg` of earlier checkpoints they lie
# switched off (False/0/None) -- such ones load; enabled ones are refused.
_REMOVED_OPTS = ("runway", "prev_act")


# The architecture was frozen on 22.09 (backlog, "Architecture freeze"): there are no flags
# any more, everything taken is unconditional. What went in, one line per change:
#   P1  value reads `h.detach()`: a critic on the shared trunk memorized games
#       by the opponent's field (internal architecture audit); R1 gave it its own tower,
#       its own opponent field without a bottleneck and two heads (value_head.py);
#   P2b the result after the corrector is read by THE SAME heads as the first pass -- one
#       path to the output instead of `head(h) + d_head(z)` with eleven corrections;
#   P5  opponent bottleneck (`opp_k`);
#   P6a two corrector layers, P6b corrector units look at their own tiles
#       with a separate cross-attention;
#   P8  three global policy columns (dying, feed deficit, decay);
#   P9  scalar heads `head_stay`, `head_land` are Linear;
#   (a) no turn plan: play with a single market pass, and the corrector gets what
#       passes 2-3 used to give, as derived quantities -- the shed at the moment of
#       the market and revenue with slippage;
#   fixes of the corrector input `sellall` and `spent` (measurements 20.09).
# Earlier branches are the git tag `pre-freeze`, the numbers are in the project notes.


def _expect(logits, head):
    """E[n] over the CORN chain, normalized by the cap: Σ_k P(n >= k) / K.

    Differentiable and without argmax -- exactly what the hard feedback lacked:
    attention receives a quantity, not a chosen number. Since R2 (22.09)
    P(n >= k) is a product of conditionals, monotone by construction; the earlier
    sum of independent Frank-Hall sigmoids folded into the expectation also the
    "negative mass" of a non-monotone scale.
    There is no budget mask (R3) here: the corrector sees what the net WANTS, and
    executability is computed by the decoder, as before.
    """
    return (expect_counts(logits, head) / ALL_CAPS_[head]).to(logits.dtype)


def _p_all(logits, head):
    """P("take the whole stock") -- a separate decision that `_expect` did not see.

    For sells it is 43% of the teacher's orders and 73.2% of ours. `decode` pulls the flag from
    its own Bernoulli and substitutes the number with it, so E[n] describes a smaller share of
    orders than it seems.
    """
    _, _, allf = split_counts(logits, head)
    if allf is None:
        return torch.zeros_like(logits[..., 0])
    return torch.sigmoid(allf.float()).to(logits.dtype)


class FarmNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        # The module creation order MUST match what the `mega_p8` network had
        # under flags (tag `pre-freeze`): it determines whether
        # `torch.manual_seed` gives the same initial weights, and training with the same
        # seed repeats the earlier one, not a "similar" one.
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.enc_tile = build_mlp(TILE_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.n_unit = UNIT_BASE_DIM          # the previous-action tail is not read
        self.enc_unit = build_mlp(self.n_unit, [cfg.enc_hidden], d, out_norm=True)
        self.enc_product = build_mlp(PRODUCT_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.n_glob = GLOBAL_BASE_DIM        # the cash-reserve tail is not read
        self.enc_global = build_mlp(self.n_glob, [cfg.enc_hidden], d, out_norm=True)
        self.enc_crop = build_mlp(CROP_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.enc_animal = build_mlp(ANIMAL_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.type_crop = nn.Parameter(torch.zeros(d))
        self.type_animal = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.type_crop, std=0.02)
        nn.init.normal_(self.type_animal, std=0.02)
        self.id_crop = nn.Embedding(N_CROPS, d)
        self.id_animal = nn.Embedding(N_ANIMALS, d)
        nn.init.normal_(self.id_crop.weight, std=0.02)
        nn.init.normal_(self.id_animal.weight, std=0.02)
        self.register_buffer("verb_of", torch.tensor(VERB_OF), persistent=False)
        self.register_buffer("pair_of", torch.tensor(PAIR_OF), persistent=False)

        # learned type embeddings; own and opponent tiles share an encoder, but not the type
        self.type_tile_me = nn.Parameter(torch.zeros(d))
        self.type_tile_op = nn.Parameter(torch.zeros(d))
        self.type_unit = nn.Parameter(torch.zeros(d))
        self.type_product = nn.Parameter(torch.zeros(d))
        # tiles have x,y, units have a number, and goods had no identity:
        # the backbone is set-invariant, while the heads are read positionally
        self.id_product = nn.Embedding(9, d)
        nn.init.normal_(self.id_product.weight, std=0.02)
        self.type_global = nn.Parameter(torch.zeros(d))
        # teacher style vector; from zero -- start from pretrain without a condition
        self.style_emb = None
        if cfg.styles:
            self.style_emb = nn.Embedding(cfg.styles + 1, d)
            nn.init.zeros_(self.style_emb.weight)
        for p in (self.type_tile_me, self.type_tile_op, self.type_unit,
                  self.type_product, self.type_global):
            nn.init.normal_(p, std=0.02)

        self.backbone = Backbone(d, cfg.n_layers, cfg.n_heads, cfg.ffn, cfg.dropout)

        # Backbone token layout: own tiles, opponent latents, units,
        # goods, crops, animals, global. Own tiles and units share one coordinate
        # system (our farm). Opponent latents are positionless: the geometry of
        # its field stays inside the bottleneck (`op_rel`), K vectors go into the trunk.
        txy = tile_xy()
        self.rel = RelBias(cfg.n_heads, [
            (100, 0, 0, txy),          # own tiles
            (cfg.opp_k, 1, None, None),  # opponent latents
            (MAX_UNITS, 2, 0, None),   # units -- coordinates on every pass
            (9, 3, None, None),        # goods
            (N_CROPS, 4, None, None),  # crops
            (N_ANIMALS, 5, None, None),  # animals
            (1, 6, None, None),        # global
        ])
        # Opponent bottleneck (P5): 41% of trunk tokens were needed only by the critic
        # (internal architecture audit), yet everyone paid for them with the square of the set length.
        self.op_enc = Backbone(d, 1, cfg.n_heads, cfg.ffn, cfg.dropout)
        self.op_rel = RelBias(cfg.n_heads, [(100, 0, 0, txy)])
        self.op_seeds = nn.Parameter(torch.randn(cfg.opp_k, d) * 0.02)
        self.op_pma = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.op_norm = nn.LayerNorm(d)

        # --- BERT heads: each reads its own tokens ---
        # A unit action is a verb plus a pointer to an object. The identity of an
        # item stopped being a softmax row number: PICKUP_STRAWBERRY
        # with eight observations borrows from PICKUP_WHEAT with seven thousand
        # through the shared verb, and for strawberry through the shared object token,
        # which the market heads see too.
        # Each head is called TWICE: on the first pass (corrector input) and on
        # the corrector latent (result). The heads are shared (P2b).
        self.head_verb = build_mlp(d, [cfg.head_hidden], N_VERBS)
        # its OWN query per item verb: [B,U,3,d_k]. A shared query per
        # unit would glue item preferences between PICKUP and PLACE and
        # lose rank (internal expressiveness check: 29 of 40).
        self.head_qobj = build_mlp(d, [cfg.head_hidden], N_PVERB * cfg.d_k)
        self.head_kobj = build_mlp(d, [cfg.head_hidden], cfg.d_k)       # object -> key
        self.head_from = build_mlp(d, [cfg.head_hidden], cfg.d_k)          # unit -> query
        self.head_to = build_mlp(d, [cfg.head_hidden], cfg.d_k)            # tile -> key
        # scalar heads are Linear: an MLP with a hidden layer of 128 for the sake of one
        # number gave nothing (P9)
        self.head_stay = nn.Linear(d, 1)                                   # "stay" column
        # how many to take from the shed: the same CORN chain as for the market, but per unit
        self.head_qty = build_mlp(d, [cfg.head_hidden], n_logits("pickup"))
        self.head_sell = build_mlp(d, [cfg.head_hidden], N_SELL_BINS)      # per good
        self.head_buy = build_mlp(d, [cfg.head_hidden], N_BUY_BINS)        # per good
        self.head_hire = build_mlp(d, [cfg.head_hidden], N_HIRE_BINS)      # from the global
        self.head_land = nn.Linear(d, 1)                      # gate, no quantity
        # Each crop and each animal decides for itself, from its own token.
        # Before: 500 and 300 outputs from one overloaded global token,
        # while BUY_SEED is 21% of the teacher's turns.
        self.head_seed = build_mlp(d, [cfg.head_hidden], N_SEED_BINS)
        self.head_animal = build_mlp(d, [cfg.head_hidden], N_ANIMAL_BINS)

        # --- turn outcome instead of key presses (internal effect-loss note) ---
        # effect map: for each own tile, which jobs will happen on it by
        # the end of the day. Multi-label: per day 15.8 tiles get 2+ effects.
        # Invariant to who executed and in what order -- which is 59.9 bits of
        # redundancy per turn out of 92.5.
        self.head_effect = build_mlp(d, [cfg.head_hidden], N_EFFECTS)
        # table economy of the turn: delta of money, shed, seeds, hands, market stock
        self.head_econ = build_mlp(d, [cfg.head_hidden], ECON_DIM)
        # --- coordination block (corrector) ---
        # Measurement: hard feedback (argmax decisions fed back as embeddings and
        # a rerun of the backbone) brings NO money, and with active use
        # takes it away: 29% of rewritten targets cost $19k of the median of $121k.
        # The reason is structural -- in training the TRUE decisions go into the plan, in
        # play its own erroneous ones, and curing this (GLAT, SMART) means teaching the net not to
        # trust the plan: trust fell from +6.8 to +2.2 points, for `op` to -1.7.
        #
        # Here the feedback is SOFT: the distributions themselves go into attention, not
        # the chosen indices. A unit wavering 51/49 and one 99% sure no longer
        # look the same -- and that is exactly what is needed so that the wavering one yields.
        # There is no revealed input at all, so training and inference
        # coincide bitwise and there is no exposure shift.
        self.register_buffer("seed_cost", torch.tensor(SEED_COST), persistent=False)
        self.register_buffer("animal_cost", torch.tensor(ANIMAL_COST), persistent=False)
        self.register_buffer("buyable", torch.tensor(BUYABLE), persistent=False)
        # unit intent is not a tile index but a WEIGHTED tile: two units
        # going to the same place get similar vectors, and attention sees
        # the collision by dot product rather than deriving it from numbers
        self.co_intent = nn.Linear(cfg.d_k, d)
        self.co_stay = nn.Parameter(torch.zeros(d))
        self.co_act = nn.Linear(N_OPS, d)          # softmax over jobs
        self.co_unit_n = nn.Linear(1, d)           # E[how many to take from the shed]
        # E[sell], E[buy], P(sell all), shed at the moment of the market
        self.co_mkt = nn.Linear(4, d)
        pr = [f"PLACE_{g}" for g in PRODUCTS_ORDER]
        pk = [f"PICKUP_{g}" for g in PRODUCTS_ORDER]
        self.register_buffer("op_place", torch.tensor([OP_NAMES.index(n) for n in pr]),
                             persistent=False)
        self.register_buffer("op_pickup", torch.tensor([OP_NAMES.index(n) for n in pk]),
                             persistent=False)
        self.op_drop = OP_NAMES.index("DROP")
        # E over seeds, animals, hiring + P(land) + E[spend] + E[revenue]
        self.co_glob = nn.Linear(N_CROPS + N_ANIMALS + 1 + 1 + 2, d)
        # E[how many to buy] into its own object token, as for goods.
        # ⚠ The same five and three quantities also remain in `co_glob`: there they are needed
        # as terms of SPEND, which is computed globally and by arithmetic.
        # The duplication is deliberate: eight extra columns of one Linear
        # against a separate branch for the spend computation.
        self.co_seed = nn.Linear(1, d)
        self.co_anim = nn.Linear(1, d)
        self.coord = Backbone(d, cfg.coord_layers, cfg.n_heads, cfg.ffn,
                              cfg.dropout)
        # Corrector units look at their own tiles with a separate cross-attention
        # (P6b): its own softmax, output zero at the start.
        self.co_xt_norm = nn.LayerNorm(d)
        self.co_xt = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        nn.init.zeros_(self.co_xt.out_proj.weight)
        nn.init.zeros_(self.co_xt.out_proj.bias)
        # The same shift inside the block: it reconciles units, and reconciling
        # them must be by mutual placement -- two units in one cell and
        # two at opposite ends of the farm are different conflicts.
        self.rel_co = RelBias(cfg.n_heads, [
            (MAX_UNITS, 0, 0, None), (9, 1, None, None),
            (N_CROPS, 2, None, None), (N_ANIMALS, 3, None, None),
            (1, 4, None, None),
        ])

        # The value tower is created LAST -- exactly at the place where
        # `ValueHead` used to stand. Then `torch.manual_seed` gives the policy the same
        # initial weights as before R1, and arm b12 differs from b11 only in the
        # critic (see the comment on module order at the start of __init__).
        self.vtower = ValueTower(d, cfg.n_heads, cfg.ffn, cfg.value_layers,
                                 TILE_FEAT_DIM, VCRIT_FEAT_DIM, cfg.enc_hidden,
                                 cfg.head_hidden, cfg.dropout)

    def policy_parameters(self):
        """Everything except the value tower. The gradients of these two groups are not mixed.

        Why separately. The tower input is detached, so its loss does not
        flow into the policy -- but `clip_grad_norm_` over ALL parameters computes a common norm, and
        through it the critic would silently scale the policy step. Clipping goes by
        groups (`bc_train`), and then "the critic does not touch the policy" holds for
        training too, not only for the forward pass.
        """
        return [p for n, p in self.named_parameters() if not n.startswith("vtower.")]

    def value_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith("vtower.")]

    def load_bc(self, sd: dict, warm: bool = True):
        """Loading a BC checkpoint that has no value tower yet.

        Strict about everything except the critic: an extra key or a different shape in the
        policy is a loud refusal, as before. A missing `vtower.*` is not
        an error (pre-R1 checkpoints do not contain it), an extra `value.*` is not either:
        that is the former `ValueHead`, and its weights go to the tower as a warm start
        (`ValueTower.init_from_bc`). Returns the list of transferred groups or
        None if the tower came from the checkpoint as is.
        """
        miss, unexp = self.load_state_dict(sd, strict=False)
        bad = [k for k in miss if not k.startswith(("vtower.", "style_emb."))]
        extra = [k for k in unexp if not k.startswith("value.")]
        if bad or extra:
            raise RuntimeError(
                f"checkpoint of a different architecture: missing {bad[:6]}, extra {extra[:6]}")
        if not [k for k in miss if not k.startswith("style_emb.")]:
            return None
        if not warm:
            return []
        return self.vtower.init_from_bc(sd, self.cfg.n_layers)

    def _shed_at_market(self, have, units, unit_mask, op_logits, target_logits,
                        qty_logits, legal, uxy):
        """EXPECTED shed at the moment of the market from the distributions of this turn's units.

        The same arithmetic as `bc_train.shed_at_market` (the engine applies unit
        jobs BEFORE the market): DROP dumps the whole cargo, PLACE_g puts the cargo of good
        g, PICKUP_g takes E[how many] -- all only for a unit on a shed access
        cell and only if it works HERE (target is its own cell or
        "stay"), rather than walking. Previously the corrector estimated sells from the shed at the
        start of the turn and for 11.6% of the teacher's sells (good dumped this same turn)
        saw zero. Unit order and shed capacity are ignored: this is an estimate
        for attention, the decoder computes the exact shed.
        """
        ol = op_logits.float()
        if legal is not None:
            ol = ol.masked_fill(legal == 0, -1e9)
        po = torch.softmax(ol, -1)                                     # [B,U,OPS]
        pt = torch.softmax(target_logits.float(), -1)                  # [B,U,101]
        here = (uxy[..., 1] * BOARD + uxy[..., 0]).clamp(0, BOARD * BOARD - 1)
        p_here = pt.gather(-1, here.unsqueeze(-1)).squeeze(-1) + pt[..., -1]
        w = (units[..., U_AT_SHED].float() * p_here * unit_mask.float())  # [B,U]
        inv = units[..., U_INV0:U_INV0 + have.shape[1]].float() * 20.0   # [B,U,9]
        e_pick = _expect(qty_logits, "pickup").float() * ALL_CAPS_["pickup"]  # [B,U]
        d_in = (po[..., self.op_drop:self.op_drop + 1] * inv
                + po[..., self.op_place] * inv)
        d_out = po[..., self.op_pickup] * e_pick.unsqueeze(-1)
        delta = ((d_in - d_out) * w.unsqueeze(-1)).sum(1)              # [B,9]
        return torch.clamp(have.float() + delta / SHED_CAP_, 0.0, 1.0).to(have.dtype)

    def _compose(self, verb_logits, q_obj, k_obj):
        """[B,U,N_VERBS] + [B,U,3*dk] x [B,N_OBJ,dk] -> [B,U,N_OPS].

        The output stays the same 40 classes as for the flat head: labeling,
        legality mask, effect map and decoder are addressed by
        class number, and changing this addressing for the sake of factorization is pointless --
        the PARAMETRIZATION changes, not the action space.

        Degrees of freedom per unit: 14 verb logits plus 3x17 pair cells,
        i.e. 65 for 40 classes, and the class -> cell map is injective.
        So any set of 40 logits is exactly representable -- the expressiveness of the
        flat head is preserved, not approximated.
        """
        B, U, _ = q_obj.shape
        dk = self.cfg.d_k
        q = q_obj.view(B, U, N_PVERB, dk)
        pair = torch.einsum("bupd,bod->bupo", q, k_obj) / math.sqrt(dk)
        pair = pair.reshape(B, U, N_PVERB * N_OBJ)
        # the last column is "no object": objectless verbs take zero
        pair = torch.cat([pair, torch.zeros_like(pair[..., :1])], -1)
        return verb_logits[..., self.verb_of] + pair[..., self.pair_of]

    def trunk(self, tiles_me: torch.Tensor, tiles_op: torch.Tensor,
              units: torch.Tensor, unit_mask: torch.Tensor,
              products: torch.Tensor, glob: torch.Tensor,
              crops: torch.Tensor, animals: torch.Tensor, style=None):
        """Policy trunk: tokens -> (h [B,S,D], pad [B,S], uxy [B,U,2], T_op).

        Factored out of `forward` for the critic phase (`ppo.value_cache`): the tower needs
        only `h` (and `pad`), while the head, corrector and pointer are not needed for it. This is
        the same code that was in `forward`, and `forward` calls it too -- there is no second
        path to `h`.
        """
        B, T, _ = tiles_me.shape
        tok_me = self.enc_tile(tiles_me) + self.type_tile_me
        tok_op = self.enc_tile(tiles_op) + self.type_tile_op
        # opponent bottleneck: a hundred of its tiles -> K latents
        zo = self.op_enc(tok_op, attn_bias=self.op_rel(
            torch.zeros(B, 1, 2, dtype=torch.long, device=tok_op.device), tok_op.dtype))
        seeds = self.op_seeds.to(zo.dtype).unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.op_pma(seeds, zo, zo, need_weights=False)
        tok_op = self.op_norm(seeds + pooled) + self.type_tile_op
        T_op = tok_op.shape[1]
        tok_u = self.enc_unit(units[..., :self.n_unit]) + self.type_unit
        tok_p = (self.enc_product(products) + self.type_product
                 + self.id_product.weight.unsqueeze(0))
        tok_g = (self.enc_global(glob[..., :self.n_glob]) + self.type_global).unsqueeze(1)
        if self.style_emb is not None:
            if style is None:                       # no label -- the "common" style
                style = torch.full((tok_g.shape[0],), self.style_emb.num_embeddings - 1,
                                   dtype=torch.long, device=tok_g.device)
            tok_g = tok_g + self.style_emb(style).to(tok_g.dtype).unsqueeze(1)
        if crops is None or animals is None:
            raise ValueError("crop and animal tokens are required: "
                             "features.encode provides 'crops' and 'animals'")
        tok_c = (self.enc_crop(crops) + self.type_crop
                 + self.id_crop.weight.unsqueeze(0))
        tok_a = (self.enc_animal(animals) + self.type_animal
                 + self.id_animal.weight.unsqueeze(0))

        zb = lambda n: torch.zeros(B, n, dtype=torch.bool, device=tok_me.device)
        x = torch.cat([tok_me, tok_op, tok_u, tok_p, tok_c, tok_a, tok_g], dim=1)
        pad = torch.cat([zb(T), zb(T_op), ~unit_mask, zb(products.shape[1]),
                         zb(N_CROPS), zb(N_ANIMALS), zb(1)], dim=1)
        # Relative shift: unit coordinates are taken from their own features
        # (features.py puts x/10 and y/10 into columns U_FX, U_FY). Dead slots
        # give (0,0) and are closed by padding anyway.
        uxy = torch.stack([units[..., U_FX], units[..., U_FY]], -1)
        uxy = (uxy * 10.0).round().long().clamp_(0, BOARD - 1)
        h = self.backbone(x, key_padding_mask=pad,
                          attn_bias=self.rel(uxy, x.dtype))
        return h, pad, uxy, T_op

    def forward(self, tiles_me: torch.Tensor, tiles_op: torch.Tensor,
                units: torch.Tensor, unit_mask: torch.Tensor,
                products: torch.Tensor, glob: torch.Tensor,
                crops: torch.Tensor, animals: torch.Tensor,
                tile_mask: Optional[torch.Tensor] = None,
                legal: Optional[torch.Tensor] = None,
                vcrit: Optional[torch.Tensor] = None,
                want_value: bool = True,
                style: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """tiles_*: [B,T,F]; units: [B,U,F]; unit_mask: [B,U] bool (True = live unit);
        products: [B,9,F]; glob: [B,F]; tile_mask: [B,T] bool (True = legal target);
        legal: [B,U,N_OPS] (True/1 = job is allowed on the unit's CURRENT cell,
        `legal_now`) -- the corrector uses it to estimate the shed at the moment of the market, so as
        not to compute the expectation from untrained logits of illegal classes;
        vcrit: [B,VCRIT_FEAT_DIM] -- critic-only columns
        (`features.value_extra`), they do not go to the policy;
        want_value: whether to compute the value tower (it is unused at inference)."""
        if want_value and vcrit is None:
            # Zeros must not be substituted: this is exactly the silent train/play
            # desync that burned us three times (project notes).
            raise ValueError("want_value=True requires vcrit (features.value_extra)")
        B, T, _ = tiles_me.shape
        U = units.shape[1]
        h, pad, uxy, T_op = self.trunk(tiles_me, tiles_op, units, unit_mask,
                                       products, glob, crops, animals, style=style)
        zb = lambda n: torch.zeros(B, n, dtype=torch.bool, device=h.device)

        P = products.shape[1]
        h_me = h[:, :T]
        o = T + T_op                                   # start of units in the set
        h_u = h[:, o:o + U]
        h_p = h[:, o + U:o + U + P]
        h_c = h[:, o + U + P:o + U + P + N_CROPS]
        h_a = h[:, o + U + P + N_CROPS:o + U + P + N_CROPS + N_ANIMALS]
        h_g = h[:, -1]

        # --- first pass: the same heads on trunk tokens ---
        # Its output goes NOWHERE except the corrector: it is the block's input, not a
        # decision. The pair logit is computed JOINTLY: logit(verb) + q_verb *
        # k_object; the mask is elementwise over the same 40 classes (loss_fn, play.py),
        # i.e. the product of spaces is masked, not the marginals.
        k_obj = self.head_kobj(torch.cat([h_p, h_c, h_a], 1))           # [B,17,dk]
        op_logits = self._compose(self.head_verb(h_u),
                                  self.head_qobj(h_u), k_obj)
        qty_logits = self.head_qty(h_u)                                 # [B,U,100]

        q = self.head_from(h_u)                                         # [B,U,dk]
        k = self.head_to(h_me)                                          # [B,T,dk]
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.cfg.d_k)
        # tile_mask is the mask of LEGAL TARGETS (features.py): open tiles plus
        # the four shed access cells, which can be closed but work.
        if tile_mask is not None:
            scores = scores.masked_fill(~tile_mask.unsqueeze(1), float("-inf"))
        target_logits = torch.cat([scores, self.head_stay(h_u)], dim=-1)  # [B,U,T+1]

        seed_logits = self.head_seed(h_c)                        # [B,5,bins]
        animal_logits = self.head_animal(h_a)                    # [B,3,bins]
        hire_logits = self.head_hire(h_g)
        land_logits = self.head_land(h_g)
        sell_logits = self.head_sell(h_p)
        buy_logits = self.head_buy(h_p)

        # --- corrector ---
        # unit intent: where it is headed, as a tile weighted by
        # probabilities. The "stay" column goes in as a separate learned vector.
        pt = torch.softmax(target_logits.float(), -1).to(k.dtype)
        intent = self.co_intent(torch.matmul(pt[..., :T], k))
        intent = intent + pt[..., T:] * self.co_stay
        # The job distribution goes into the block WITHOUT the legality mask. Masking it
        # was measured twice (backlog, "Do not do"): a mask on the DESTINATION tile is
        # a leak ($891 in play), an honest `legal_now` gives 58/96, nothing.
        # The block reconciles units through `intent` and `h_u`.
        zu = (h_u + intent
              + self.co_act(torch.softmax(op_logits.float(), -1).to(h_u.dtype))
              + self.co_unit_n(_expect(qty_logits, "pickup").unsqueeze(-1)))
        e_sell = _expect(sell_logits, "sell")
        e_buy = _expect(buy_logits, "buy")
        # The "sell the whole stock" flag is a separate decision, and in play it sends
        # 73.2% of SELL orders. The effective sale volume accounting for it, from
        # the shed at the MOMENT OF THE MARKET (DROP/PLACE/PICKUP of this turn already counted):
        # passes 2-3 used to give this, now it is a derived quantity.
        p_all = _p_all(sell_logits, "sell")
        have = self._shed_at_market(products[:, :, 3], units, unit_mask, op_logits,
                                    target_logits, qty_logits, legal, uxy)
        e_sell_eff = p_all * have + (1 - p_all) * torch.minimum(e_sell, have)
        zp = h_p + self.co_mkt(torch.stack([e_sell, e_buy, p_all, have], -1))
        # Spend is computed EXPLICITLY, not derived from logits: it is arithmetic,
        # and approximating it with attention makes no sense. Seeds, hiring, animals, land
        # ($1000-4000) and purchases only of those goods that the decoder actually takes.
        price = products[:, :, 0] * 300.0                     # $/unit
        d_prod = (e_buy * self.buyable * price).sum(-1, keepdim=True) * 100.0
        d_seed = (_expect(seed_logits, "seed")
                  * self.seed_cost).sum(-1, keepdim=True) * 100.0
        d_anim = (_expect(animal_logits, "animal")
                  * self.animal_cost).sum(-1, keepdim=True) * 100.0
        # price of the NEXT hand (mult*fib(hires_today)); hiring gets more expensive,
        # so this is a lower estimate -- enough as a feature
        d_hire = (_expect(hire_logits.unsqueeze(1), "hire") * 10.0
                  * glob[:, G_HIRE_COST:G_HIRE_COST + 1] * 1000.0)
        p_land = torch.sigmoid(land_logits.float()).to(h_g.dtype)
        d_land = p_land * glob[:, G_LAND_PRICE:G_LAND_PRICE + 1] * 4000.0
        spent = (d_prod + d_seed + d_anim + d_hire + d_land) / MONEY0
        # slippage: each sold unit moves the price by the marginal
        # shift (column 4 of the good token, $/unit after /20); revenue of n units
        # ≈ n·(p − n·Δ/2), not below the $1 floor
        n_sell = e_sell_eff * 100.0
        slip = products[:, :, 4] * 20.0
        unit_px = torch.clamp(price - 0.5 * n_sell * slip, min=1.0)
        revenue = (n_sell * unit_px).sum(-1, keepdim=True) / MONEY0
        zg = h_g + self.co_glob(torch.cat([
            _expect(seed_logits, "seed"), _expect(animal_logits, "animal"),
            _expect(hire_logits.unsqueeze(1), "hire"),
            p_land, spent, revenue], -1))
        # Own tiles via cross-attention, NOT as tokens in the block's set.
        # Intent is not enough: `Σ p(tile)·k_tile` of a unit wavering 50/50
        # between two distant cells points to the MIDDLE, and the block cannot
        # look at what lies on the contested cell. But tiles as tokens -- v21
        # vs v18: -$30,222, 9/48, p=1.5e-05: a hundred tiles dropped the uniform
        # share of units in the set from 52% to 8%, and reconciliation drowned.
        # illegal targets do not serve as keys: the same tiles as for the pointer
        kpm = ~tile_mask if tile_mask is not None else None
        xt, _ = self.co_xt(self.co_xt_norm(zu), h_me, h_me,
                           key_padding_mask=kpm, need_weights=False)
        zu = zu + xt
        zc = self.coord(torch.cat([
            zu, zp,
            h_c + self.co_seed(_expect(seed_logits, "seed").unsqueeze(-1)),
            h_a + self.co_anim(_expect(animal_logits, "animal").unsqueeze(-1)),
            zg.unsqueeze(1)], 1),
            key_padding_mask=torch.cat([
                ~unit_mask, zb(P), zb(N_CROPS), zb(N_ANIMALS), zb(1)], 1),
            attn_bias=self.rel_co(uxy, zu.dtype))
        zu, zp, zg = zc[:, :U], zc[:, U:U + P], zc[:, -1]
        zc_c = zc[:, U + P:U + P + N_CROPS]
        zc_a = zc[:, U + P + N_CROPS:U + P + N_CROPS + N_ANIMALS]

        # --- result: the same heads on the corrector latent (P2b) ---
        op_logits = self._compose(self.head_verb(zu), self.head_qobj(zu), k_obj)
        qty_logits = self.head_qty(zu)
        sc = torch.matmul(self.head_from(zu), k.transpose(1, 2)) / math.sqrt(self.cfg.d_k)
        if tile_mask is not None:
            sc = sc.masked_fill(~tile_mask.unsqueeze(1), float("-inf"))
        target_logits = torch.cat([sc, self.head_stay(zu)], dim=-1)
        return {
            "op_logits": op_logits,                                   # [B,U,40]
            "target_logits": target_logits,                           # [B,U,101]
            "sell_logits": self.head_sell(zp),                        # [B,9,101]
            "buy_logits": self.head_buy(zp),                          # [B,9,100]
            "hire_logits": self.head_hire(zg),                        # [B,10]
            "land_logits": self.head_land(zg),                        # [B,1]
            "seed_logits": self.head_seed(zc_c),                      # [B,5,100]
            "animal_logits": self.head_animal(zc_a),                  # [B,3,100]
            "qty_logits": qty_logits,                                 # [B,U,100]
            "effect_logits": self.head_effect(h_me),                  # [B,100,17]
            "econ": self.head_econ(h_g),                              # [B,10]
            # Value is a separate tower over `h.detach()` plus a hundred OPPONENT tiles
            # whole and critic columns (R1). The input is detached inside the tower,
            # so the policy does not depend on it by a single bit.
            **dict(zip(("value_logits", "value_hl"),
                       self.vtower(h, pad, tiles_op, vcrit) if want_value
                       else (torch.zeros(B, 3, device=h.device, dtype=h.dtype),
                             torch.zeros(B, HL_BINS, device=h.device, dtype=h.dtype)))),
        }
