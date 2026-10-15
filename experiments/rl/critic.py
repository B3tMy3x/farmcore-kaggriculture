"""PPO-loop critic: three methods backed by the R1 value tower.

The loop knows nothing about the critic except three methods:

    value(turns)  -> np.ndarray [T]   estimate at collection time, no gradient
    head(out, x)  -> Tensor [B]       the same estimate in a minibatch, WITH gradient
    params()      -> list             what to add to the optimizer

Scale. `value` must live in the same scale as the reward from `RewardCfg`:
GAE subtracts one from the other. Neither the reward formula nor the estimate is
written here - both are called from `experiments/net/value_head.py`
(`terminal_reward`, `value_scalar`, `RL_ALPHA`, `RL_SIGMA`). That is the only
place they live, which is exactly why a critic with one sigma and a trainer with another
are impossible here.

WARNING: `WDLCritic` (the former `ValueHead` on the shared trunk) was removed on 23.09, not left
behind a flag. After R1 the network has neither a `value` module nor a pass without `vcrit`: the class
crashed with `ValueError: want_value=True requires vcrit` on the very first rollout, and
`params()` looked for a nonexistent `net.value`. The loop mechanics are checked by `zero`,
money by `tower`; we keep no third path to the estimate.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Protocol

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import value_head as VH  # noqa: E402


class Critic(Protocol):
    """What the loop needs from the value. The R1 implementation is plugged in here."""

    def value(self, turns: List[Any], device: str = "cpu") -> np.ndarray:
        """V(s) for every turn, in the reward scale. -> [T] float64. No gradient."""
        ...

    def head(self, out: Dict[str, torch.Tensor], x: Dict[str, torch.Tensor]):
        """V(s) of a minibatch WITH gradient. -> [B] or None if the critic does not train.

        Both inputs are given: `out` is the network pass output, `x` is its input.
        """
        ...

    def params(self):
        """Parameters trained together with the policy (may be empty)."""
        ...


class ZeroCritic:
    """Stub: V == 0.

    Not "nothing": with gamma = 1 and a terminal reward this turns GAE into
    Monte Carlo without a baseline - the advantage of each turn equals the
    (lambda-weighted) remainder of the reward to the end of the game. One can train on it;
    the variance over 720 turns per single +-1 is hopeless, which is exactly why R1 was
    a blocker. It stays as a control: its `explained_var` is identically 0
    (V == 0 and target = advantage), so any other number in a smoke run is
    the critic's work, not arithmetic.
    """

    trainable = False
    wants_value = False

    def value(self, turns, device="cpu"):
        return np.zeros(len(turns), dtype=np.float64)

    def head(self, out, x):
        return None

    def params(self):
        return []


class TowerCritic:
    r"""R1 value tower (`value_head.ValueTower`) as the loop critic.

    **Where the estimate comes from.** `V(s) = (P(W) - P(L)) + alpha*E[tanh(delta_bank/sigma)]` -
    `value_head.value_scalar` over the tower's two heads. It is not rewritten here:
    the loop has neither a softmax over W/D/L nor the HL bin centres. The rollout reward
    is computed by `value_head.terminal_reward` with the same alpha and sigma
    (`collect._turn_rewards`), so GAE subtracts one from the other in one
    scale - roughly [-1-alpha, 1+alpha].

    **Why `head` reads `out` instead of making its own pass.** The tower lives INSIDE
    `FarmNet` and reads `h.detach()` - the policy trunk tokens. It cannot have a separate pass
    over `x`: the same trunk would have to run a second time,
    paying exactly as much as the first and getting the same numbers.
    So `wants_value = True`: the trainer's pass computes the tower itself
    (`FarmNet.forward(..., want_value=True)`), and `head` takes its two outputs.
    The policy does not change by a single bit - the tower input is detached inside it
    (`ValueTower.forward`), and this is a property of the construction.

    **Critic columns.** `vcrit` (`features.value_extra`, 32 of them) is written by
    `play.agent` into the turn's `enc`. Their absence is a loud failure below, not zeros:
    with zeros the tower would compute "opponent empty, tempos zero" on every turn and
    no measurement would show it.
    """

    trainable = True
    wants_value = True     # the trainer's pass must compute the tower

    def __init__(self, net, alpha: float = None, sigma: float = None):
        if not hasattr(net, "vtower"):
            raise ValueError("the network has no value tower: checkpoint predates R1")
        self.net = net
        # sigma does NOT take part in the critic's computations: it is baked into the LABELS
        # the HL head was trained on (`bc_train.value_target`), and cannot be
        # read from the weights. The field is declarative - `ppo.Trainer` uses it to check the reward sigma
        # against the one the tower was trained with (`collect.ckpt_value_scale`).
        self.sigma = VH.RL_SIGMA if sigma is None else float(sigma)
        # alpha is taken FROM `value_head`, not from the loop config, and this matters:
        # `ppo.Trainer` checks the reward alpha against exactly this field. Pass a
        # value from `RewardCfg` here and the check would compare a number with itself.
        # The argument is kept exactly for the probe "what if they diverge".
        self.alpha = VH.RL_ALPHA if alpha is None else float(alpha)

    # --- pass ---

    def _forward(self, x):
        if "vcrit" not in x:
            raise KeyError(
                "turn has no critic columns `vcrit`: the rollout was collected by pre-"
                "R1 code (`play.agent` puts them into `enc`). Zeros must not be substituted")
        return self.net(x["tiles_me"].float(), x["tiles_op"].float(),
                        x["units"].float(), x["unit_mask"],
                        x["products"].float(), x["glob"].float(),
                        crops=x["crops"].float(), animals=x["animals"].float(),
                        tile_mask=x["tile_mask"], legal=x["legal"],
                        vcrit=x["vcrit"].float(), want_value=True)

    def scalar(self, out):
        """Two tower heads -> [B] in the `terminal_reward` scale."""
        return VH.value_scalar(out["value_logits"], out["value_hl"], self.alpha)

    # --- the three protocol methods ---

    def value(self, turns, device="cpu", batch=64):
        """V(s) at collection time. Batch 64, not 256: a turn stores the whole network input,
        and a hundred foreign tiles in the tower go WITHOUT the bottleneck (R1) - memory per batch is higher
        than in the policy pass."""
        out = np.zeros(len(turns), dtype=np.float64)
        was = self.net.training
        self.net.eval()
        with torch.no_grad():
            for s in range(0, len(turns), batch):
                sub = turns[s:s + batch]
                x = {k: torch.from_numpy(np.stack([t.enc[k] for t in sub])).to(device)
                     for k in sub[0].enc}
                x["legal"] = torch.from_numpy(
                    np.stack([t.legal0 for t in sub])).to(device)
                v = self.scalar(self._forward(x))
                out[s:s + len(sub)] = v.double().cpu().numpy()
        self.net.train(was)
        return out

    def head(self, out, x):
        # The tower is already computed by the same pass: there is no second pass over the trunk.
        return self.scalar(out)

    def params(self):
        """Tower parameters only. `FarmNet.value_parameters` is one list for
        BC training and for PPO; there is deliberately no name-based slice of our own here."""
        return list(self.net.value_parameters())


class ScalarTowerCritic(TowerCritic):
    """R1 tower plus a SCALAR linear head on its pooling (rl3, 24.09).

    Why. `TowerCritic.scalar` is (P(W) - P(L)) + alpha*E[tanh(delta/sigma)], i.e. V
    is locked to +-(1 + alpha) and describes exactly the terminal reward
    `sign + alpha*tanh(delta/sigma)`. The dense reward of the earlier project (`reward.margin_delta_scale`:
    delta margin/scale every turn plus +-1 for the outcome) gives returns of a different
    scale and different meaning (the REMAINING margin, not the final one), and the W/D/L and
    HL heads cannot express it. The head here is `Linear(D, 1)`, zeroed at start (like the
    "critic final projection zeroed" of the earlier project): V = 0 until it learns.
    The tower trunk and its pooling are unchanged (taken from `--critic-ckpt`), so
    the features of the best rl1 critic stay, only the projection changes.

    Own scale: the `ppo.Trainer._check_scale` check is not needed for it
    (`free_scale`), but a dense reward without it is forbidden there.
    """

    free_scale = True

    def __init__(self, net, alpha: float = None, sigma: float = None):
        super().__init__(net, alpha=alpha, sigma=sigma)
        d = int(net.vtower.norm.normalized_shape[0])
        dev = next(net.parameters()).device
        self.head_lin = nn.Linear(d, 1).to(dev)
        with torch.no_grad():
            self.head_lin.weight.zero_()
            self.head_lin.bias.zero_()

    def scalar(self, out):
        # Pooling of THIS same pass: `scalar` is called right after the tower call in
        # all paths (network pass, trunk cache of the critic phase, CUDA graphs - there
        # it is the same static write buffer).
        p = self.net.vtower.last_pooled
        return self.head_lin(p).squeeze(-1)

    def params(self):
        return super().params() + list(self.head_lin.parameters())

    def state_dict(self):
        return self.head_lin.state_dict()

    def load_state_dict(self, sd):
        self.head_lin.load_state_dict(sd)


KINDS = ("zero", "tower", "tower_scalar")


def make(kind: str, net=None, **kw) -> Critic:
    """`zero` - stub (GAE without a baseline), `tower` - R1 tower."""
    if kind == "zero":
        return ZeroCritic()
    if kind == "tower":
        if net is None:
            raise ValueError("TowerCritic needs a network")
        return TowerCritic(net, **kw)
    if kind == "tower_scalar":
        if net is None:
            raise ValueError("ScalarTowerCritic needs a network")
        return ScalarTowerCritic(net, **kw)
    if kind == "wdl":
        raise ValueError(
            "critic `wdl` was removed 23.09: the former `ValueHead` no longer exists in the network "
            "(R1), and a pass without `vcrit` is impossible. Use `tower`")
    raise ValueError(f"unknown critic {kind!r}; available {KINDS}")

