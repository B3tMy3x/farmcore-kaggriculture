"""Rollout collection: games are played by THE SAME code as evaluation.

The repo rule "training and play must compute one quantity" is stricter under
PPO: the `log π` used for the ratio comes from the same `policy.run_turn` that
emitted the turn into the env. So there is no decoder here — the turn is built
by `play.agent` in record mode (`play._NET["record"]`), exactly as in the
internal log-prob parity check. All this module does: installs weights,
assigns seats, cuts records into streams and computes the reward.

The env is swappable (R6). Python `kaggle_environments` gives ~3 660
decisions/s on 15 cores vs 15 150 states/s for the net at batch 256 — the
bottleneck is the env. The Rust port (`engines/kaggriculture_rs`) is not in the
loop, so running a game lives in `EnvRunner`: swapping the env means writing a
second class with one method, not editing collection.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import play  # noqa: E402
import networth as NW  # noqa: E402
import policy as POL  # noqa: E402
import value_head as VH  # noqa: E402
from model import FarmNet, ModelConfig  # noqa: E402

from buffer import Episode, Turn  # noqa: E402


# ---------------------------------------------------------------- net

def load_net(path: str, device: str = "cpu") -> FarmNet:
    """`bc_train.py` checkpoint -> net. Strict load, as in `play._load`.

    Takes `ck["model"]` — the EMA (P0). N10: the optimizer in the checkpoint
    belongs to the RAW weights, so PPO does not pick it up but creates a fresh
    one; here we simply don't read it.

    `load_bc`, not `load_state_dict` — exactly as in `play._load`: every policy
    key is checked, and a missing value tower (checkpoints before R1) is
    warm-started. Whether the tower arrived TRAINED under `--critic tower` is
    checked by `train.Run`: it is not visible here.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)  # own files; run state holds generators and league
    w = sd.get("model", sd) if isinstance(sd, dict) else sd
    net = FarmNet(ModelConfig.from_ckpt(sd))
    net.load_bc(w)
    net.eval()
    return net.to(torch.device(device))


def has_tower(path: str) -> bool:
    """Whether the checkpoint has value-tower weights (`vtower.*`)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)  # own files; run state holds generators and league
    w = sd.get("model", sd) if isinstance(sd, dict) else sd
    return any(k.startswith("vtower.") for k in w)


def load_tower(net: FarmNet, path: str) -> int:
    """Load ONLY the tower from a separate checkpoint into the net. -> key count.

    Policy weights from that file are not touched: otherwise the policy would
    silently change while the run kept its old name.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)  # own files; run state holds generators and league
    w = sd.get("model", sd) if isinstance(sd, dict) else sd
    sub = {k[len("vtower."):]: v for k, v in w.items() if k.startswith("vtower.")}
    if not sub:
        raise SystemExit(f"checkpoint {path} has no value tower (`vtower.*`)")
    net.vtower.load_state_dict(sub)      # strict: shapes and key set
    return len(sub)


def ckpt_value_scale(path: str):
    """σ and γ the critic in this checkpoint was trained with. -> (sigma, gamma) or (None, None).

    `bc_train.py` stores its args in `ck["cfg"]`, `--value-sigma` among them.
    This is the only way to know which σ the HL head was trained on: σ is not
    visible in the weights, and `value_head.RL_SIGMA` is only a DEFAULT the run
    may have overridden by flag. `ppo.Trainer` checks the reward scale against
    exactly this number.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)  # own files; run state holds generators and league
    c = sd.get("cfg") if isinstance(sd, dict) else None
    if not isinstance(c, dict):
        return None, None
    return c.get("value_sigma"), c.get("value_gamma")


# How to load a CHECKPOINT OPPONENT. Default is `load_net`, i.e. own weights in
# this process. The inference server (`infer_client`) replaces it with
# `RemoteNet` carrying a model id, so the opponent is computed on the GPU along
# with us instead of batch 1 on a core. Only the OBJECT queried for logits is
# replaced — decoder, labelling and seat assignment are unchanged.
NET_FOR_OPPONENT = None
# Opponent checkpoint path -> file holding only `prior` (`infer_client.slim_prior`).
# Set by the inference-server worker; without it `prior` is read from the
# checkpoint itself, as before.
PRIOR_PATH_FOR: Dict[str, str] = {}


def load_prior(path: str, device: str = "cpu"):
    """`ck["prior"]` — log-frequencies of ops and targets, with which
    `policy.family` restores the teacher distribution (N2). Absent in
    checkpoints before logit adjustment; not needed at `beta = 0`, and at
    `beta > 0` `policy.family` fails loudly. Loaded separately from weights
    because `load_net` returns only the net, and this is not part of it."""
    sd = torch.load(path, map_location="cpu", weights_only=False)  # own files; run state holds generators and league
    pr = sd.get("prior") if isinstance(sd, dict) else None
    if not pr:
        return None
    return {k: v.to(torch.device(device)) for k, v in pr.items()}


def ppo_decode(cfg) -> Dict[str, Any]:
    """How the trained policy plays in rollout and how `log π` is recomputed
    — ONE place (`rollout.sample_units`/`sample_qty`, −1 = `policy.PPO`).
    Rollout and trainer must both take it from here: otherwise play and
    training would compute different distributions."""
    rc = cfg.rollout
    d = dict(POL.PPO)
    su = float(getattr(rc, "sample_units", -1.0))
    sq = float(getattr(rc, "sample_qty", -1.0))
    if su >= 0:
        d["sample_units"] = su
    if sq >= 0:
        d["sample_qty"] = sq
    return d


def turn_seed(a: int, step: int, c: int = 0) -> int:
    """Per-turn generator seed (`rollout.crn`)."""
    return (int(a) * 1_000_003 + int(step) * 7_919 + int(c) * 104_729) % (2 ** 63 - 1)


def decode_mode(base: Dict[str, Any]) -> Dict[str, Any]:
    """Decoder mode + distribution family (N2/N4) — merged in ONE place.

    `policy.BC`/`policy.PPO` set only temperatures; β and the gate lower
    bound live in `policy.FAMILY` and must reach `_NET` together with them,
    otherwise rollout and recompute play different distributions
    (`policy.family`). Same merge as `play.DECODE_DEFAULTS`.
    """
    return dict(base, **POL.FAMILY)


def _step_of(o) -> int:
    """Turn number from the observation. `step` is not always present — then day and hour."""
    s = o.get("step")
    return int(s) if s is not None else int(o["day"]) * 24 + int(o["hour"])


def _shops_of(o) -> tuple:
    """Town's unlocked shops as a tuple (order = unlock order, one shop every
    3 days). Needed by group collection: shops are the main exogenous
    randomness, and with one env seed their sequence DIVERGES across a group's
    games — `_end_of_day` draws the shop from the same `rng` that was
    previously consumed over empty tiles of both farms. Empty if the
    observation carries no town."""
    try:
        return tuple(o["town"].get("unlocked_shops", []) or ())
    except Exception:           # noqa: BLE001 — truncated observation
        return ()


class SlimRecord(list):
    """List of turn records that drops excess on input.

    `play.agent` puts `copy.deepcopy(obs)`, `action` and `cfg` into the record
    UNCONDITIONALLY, while collection needs exactly four numbers from `obs`:
    whose turn, which step, and both sides' money (`play_episode`). With 720
    turns per side, live observation copies keep the whole game in memory —
    a hundred tiles × two players × turn — for nothing when `keep_obs = False`.

    Why by replacing the list rather than a flag in `play.py`. `_NET["record"]`
    is an object the loop hands to the game; the loop decides what its
    `append` does. So `play.py` stays one code for both evaluation and rollout,
    with no "if PPO" branch.

    ✅ The `deepcopy` itself was removed 23.09: `play.agent` copies only under
    `_NET["keep_obs"]` (default `True` — "as before"), and `install` clears the
    flag exactly when the record is this class. Only it has that right: its
    `append` below rebuilds the four numbers as NEW objects and keeps no
    reference to the live observation, so the env later mutating farms in
    place does not affect the record. The copy cost 693 µs per decision on
    port dicts and 1 731 µs on framework `Struct`s, vs a 6.4 ms Python env.
    """

    __slots__ = ("net", "every")

    def __init__(self, net=None, every: int = 1):
        super().__init__()
        # `net` is the game module's `_NET`: the list can tell it whether to
        # compute `log π` on the NEXT turn. Sparse-check logic lives here, not
        # in `play.py`, for the same reason record truncation lives here:
        # `play.py` stays one code, and the loop decides how often the check
        # is needed.
        self.net = net
        self.every = max(1, int(every))

    def append(self, r):
        o = r["obs"]
        r["obs"] = {"player": int(o["player"]), "step": _step_of(o),
                    "farms": [{"money": float(o["farms"][0]["money"])},
                              {"money": float(o["farms"][1]["money"])}],
                    "shops": _shops_of(o),
                    # Φ margin is computed HERE, on the live observation, while
                    # it is still whole: the `reward.phi_delta_scale` reward
                    # needs the shed and market inventory, absent from the slim record.
                    "phi": NW.phi_obs(o),
                    "phi_cost": NW.phi_cost_obs(o),
                    "mkt": {k: int(v) for k, v in o["market"]["inventory"].items()},
                    "opp_money": float(o["farms"][1 - int(o["player"])]["money"])}
        r.pop("action", None)
        r.pop("cfg", None)
        super().append(r)
        if self.net is not None and self.every > 1:
            # Turn 0 is always computed (`install` sets the flag up), then
            # every `every`-th. The check keeps its meaning: exactly the same
            # two quantities are compared — `log π` FROM PLAY and the trainer's
            # recompute — just on a sample of turns.
            self.net["logp_now"] = (len(self) % self.every == 0)


def install(mod, net, device: str, decode: Dict[str, Any], record: bool,
            prior=None, slim: bool = True, who: str = "", logp_every: int = 1):
    """Install weights and decoder mode into the game module.

    `play._NET` is the module's global dict: `play.main` fills it the same way
    and `--opp-ckpt` replaces it the same way (with a separate module copy, so
    sides don't share weights and accumulated flow).
    """
    mod._NET.clear()
    mod._NET["m"] = net
    mod._NET["dev"] = torch.device(device)
    # `prior` is set HERE because we bypass `play._load`: weights are passed
    # in as a ready net, and the teacher log-frequencies (`ck["prior"]`) sit
    # next to them in the checkpoint and are loaded by `load_prior`. Without
    # them at β > 0 `policy.family` fails, which beats playing raw logits.
    # ⚠ LOUD FAILURE, not silent play with raw logits. `policy.family` fails
    # at β > 0 with empty `prior` — but INSIDE the turn, and the competition
    # framework swallows agent exceptions and marks it ERROR. From outside it
    # looks like an opponent idling 719 turns at $3 000, and that is exactly
    # how the league ran on 23.09: actor checkpoints were saved without
    # `prior`. Worst of all, such games went into critic labels as huge wins.
    #
    # The check is here because this is the only place where decoder settings
    # and log-frequencies meet: `play._load` has this check, but league
    # opponents bypass it — through `load_net` and here.
    if float(decode.get("beta", 0.0)) > 0.0 and not prior:
        raise SystemExit(
            f"{who or 'net'} is loaded without `prior` at β = "
            f"{decode.get('beta')}: `policy.family` adds β·prior to the op "
            "and target logits; without it this is a DIFFERENT distribution. Checkpoints before 23.09 "
            "and frozen packages carry no `prior` — such an opponent needs "
            "`--beta 0` on its side, and an actor checkpoint must "
            "carry `prior` (train.Run.save)")
    mod._NET["prior"] = prior
    mod._NET.update(decode)
    mod._NET["check_orders"] = False     # engine cross-check is a separate probe
    if record:
        mod._NET["record"] = (SlimRecord(mod._NET, logp_every) if slim else [])
        # Turn 0 is always checked; afterwards `SlimRecord` decides.
        mod._NET["logp_now"] = True
        # ⚠ `keep_obs = False` ALLOWS `play.agent` to skip
        # `copy.deepcopy(obs)` before recording. Only `SlimRecord` has that
        # right: its `append` immediately rebuilds four numbers from the
        # observation as new objects and keeps no reference to the live one.
        # If the record is a plain list (`slim=False`, `rollout.keep_obs`), the
        # copy must stay — there the observation outlives the turn, and the env
        # mutates farm structures in place.
        mod._NET["keep_obs"] = not slim
    return mod


def fresh_play_module():
    """A second copy of `play.py` for the opponent.

    `_NET` and `_FLOW` in `play.py` are global, and a plain import would return
    the same object: sides would share weights and accumulated flow. This is
    exactly how `play.py --opp-ckpt` does it too.
    """
    spec = importlib.util.spec_from_file_location(
        "play_opp_rl", os.path.join(ROOT, "experiments/net/play.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- env

class KaggleEnvRunner:
    """A game on the competition's Python env.

    A replacement (Rust port, R6) must honour the same contract: agents are
    `agent(obs, config)` functions; output is final rewards and a "removed from
    game" flag.
    """

    name = "kaggle_environments"

    def __init__(self, episode_steps: int = 720):
        self.steps = episode_steps

    def run(self, agents, seed: int):
        from kaggle_environments import make
        env = make("kaggriculture",
                   configuration={"episodeSteps": self.steps, "seed": seed},
                   debug=False)
        env.run(list(agents))
        last = env.steps[-1]
        rew = [last[i].reward or 0.0 for i in (0, 1)]
        # The LAST step's status is useless: the interpreter unconditionally
        # sets DONE for everyone (env_ref.py:958-963), and an agent that crashed
        # on turn five is indistinguishable from a weak one. Check all steps
        # (as `play.main` does).
        dead = []
        for i in (0, 1):
            bad = next((s[i]["status"] for s in env.steps[:-1]
                        if s[i]["status"] not in ("ACTIVE", "INACTIVE")), None)
            dead.append(bad is not None)
        return rew, dead


def make_runner(rc):
    """Env per `rollout.env`. -> object with a `run` method.

    One point: both serial collection and the fan-out worker get the env here,
    otherwise "parallel" would play in a different env than "serial".
    """
    kind = getattr(rc, "env", "kaggle")
    if kind == "kaggle":
        return KaggleEnvRunner()
    if kind == "rust":
        # Imported here, not at the top: the port is an optional dependency
        # (wheel built with `maturin`), and the loop on the framework must
        # start without it.
        from collect_lockstep import RustEnvRunner
        return RustEnvRunner(native_encode=bool(getattr(rc, "native_encode", False)))
    raise SystemExit(f"unknown rollout.env: {kind!r} (kaggle | rust)")


# ---------------------------------------------------------------- reward

def shaping_potential(margins, rcfg) -> List[float]:
    """Φ(s) on turn states: `cash_shaping · margin / σ`. -> list of length n.

    A separate function because with shaping the correct baseline is
    `V(s) − Φ(s)`, and when we get there Φ will also have to be subtracted in
    GAE. The one point where it is computed is here.
    """
    return [rcfg.cash_shaping * float(m) / max(1.0, rcfg.cash_sigma)
            for m in margins]


def _turn_rewards(n: int, money: float, opp_money: float, margins,
                  rcfg, phis=None) -> List[float]:
    """Trainer reward per turn for one side. -> list of length n.

    Terminal form:
    `r_T = outcome_scale·sign(Δ) + α·tanh(Δmoney/σ)`. Computed NOT here but in
    `value_head.terminal_reward` — same code and same α and σ the critic is
    trained on; the loop has no copy of the formula. `outcome_scale` (1.0 for
    us) sits on top as a multiplier ONLY on the sign: at ≠ 1 the reward scale
    no longer matches `value_scalar`, and `ppo.Trainer` catches that by refusing.
    Sign — a bet on wins
    (the leaderboard is skill rating, not money), tanh — gradation between
    rollouts: bare ±1 has none, and two rollouts won by $1 and by $40k give
    the critic the same signal (a known PPO pitfall reported by the community).

    The dense term is potential-based shaping on cash margin (archive,
    `cash_delta_reward`), `F_t = γ·Φ(s_{t+1}) − Φ(s_t)`, `Φ = shaping_potential`.

    ⚠ Terminal. Ng's theorem (Ng, Harada, Russell, ICML 1999) preserves the
    optimum only with `Φ(terminal) = 0`; otherwise the sum telescopes not to a
    constant but to `−Φ(s₀) + Φ(s_T)`, adding a term LINEAR in the bank
    difference on top of `α·tanh(Δ/σ)` — the optimum shifts silently. The old
    code used the FINAL banks as the last "margin", i.e. both shifted the
    optimum and counted the terminal money difference twice (second time in
    `terminal_reward`). Now `Φ(s') = 0` on the last turn, the per-game sum
    equals `−Φ(s₀)` and does not depend on the policy (checked by a probe).

    Default 0 — first the loop is measured on a clean terminal signal.

    ⚠ `outcome` (true ±1) is computed separately and not mixed back into the
    reward: the archive explicitly warns here about confusing diagnostics with
    the GAE target, and `Episode.outcome` has its own field.
    """
    r = [0.0] * n
    mds = float(getattr(rcfg, "margin_delta_scale", 0.0) or 0.0)
    pds = float(getattr(rcfg, "phi_delta_scale", 0.0) or 0.0)
    if pds > 0:
        # Local reward on farm value (`RewardCfg.phi_delta_scale`): increment
        # of own Φ per turn; on the last turn Φ(s') = own money.
        if mds > 0 or rcfg.cash_shaping:
            raise ValueError("phi_delta_scale together with margin_delta_scale or "
                             "cash_shaping: two dense rewards stacked on each other")
        if phis is None or len(phis) < n:
            raise ValueError("Φ reward needs Φ for every turn (SlimRecord/obs)")
        for t in range(n - 1):
            r[t] = (float(phis[t + 1]) - float(phis[t])) / pds
        # terminal: Φ(s') = OWN money. Sum = (money − Φ(s₀)) / pds: the target
        # is own money at season end; the opponent matters only via market prices.
        r[n - 1] = (float(money) - float(phis[n - 1])) / pds
        d = money - opp_money
        r[-1] += float(rcfg.outcome_scale) * (1.0 if d > 0 else (-1.0 if d < 0 else 0.0))
        return r
    if mds > 0:
        # Archive dense reward (`RewardCfg.margin_delta_scale`): margin
        # increment each turn, ±1 for the outcome at the end. `margins` — n
        # per-turn margins plus final, so sum = (final − first-turn margin)/mds.
        if rcfg.cash_shaping:
            raise ValueError("margin_delta_scale and cash_shaping together: two "
                             "dense rewards stacked on each other")
        if margins is None or len(margins) < n + 1:
            raise ValueError("dense reward needs margins for all turns plus the final one")
        for t in range(n):
            r[t] = (float(margins[t + 1]) - float(margins[t])) / mds
        d = money - opp_money
        r[-1] += float(rcfg.outcome_scale) * (1.0 if d > 0 else (-1.0 if d < 0 else 0.0))
        return r
    if rcfg.cash_shaping and margins is not None and len(margins) >= n:
        # Φ on turn STATES s_0..s_{n−1}; `margins[n]` (final banks) no longer
        # enters shaping — it is terminal, and Φ(terminal) = 0.
        phi = shaping_potential(margins[:n], rcfg)
        for t in range(n - 1):
            r[t] += phi[t + 1] - phi[t]      # γ = 1: no discount needed here
        r[n - 1] += 0.0 - phi[n - 1]         # Φ(terminal) = 0
    d = money - opp_money
    sign = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    r[-1] += VH.terminal_reward(money, opp_money, rcfg.cash_alpha,
                                max(1.0, rcfg.cash_sigma))
    if rcfg.outcome_scale != 1.0:
        r[-1] += (rcfg.outcome_scale - 1.0) * sign
    return r


# ---------------------------------------------------------------- collection

def play_episode(net, opponent, seed: int, index: int, cfg, runner=None,
                 opp_net_cache: Optional[dict] = None,
                 our_decode: Optional[Dict[str, Any]] = None,
                 record: bool = True, sample_seed: Optional[int] = None,
                 group: str = "", branch: Optional[Tuple[int, int]] = None
                 ) -> List[Episode]:
    """One game -> experience streams of the trained policy (1 or 2).

    `our_decode` — how WE play. Default `policy.PPO` (rollout: sampling,
    honest `log π`). `policy.BC` is used by the acceptance panel
    (`collect_eval`): the checkpoint opponent plays argmax, and only argmax is
    comparable with it — see `buffer.Episode.opp_decode`. `record=False`
    disables turn recording: the acceptance panel does not go into training,
    and recording costs 18% of game time.

    `sample_seed` — sampling generator seed when it DIFFERS from the env seed
    (group collection, `rollout.group_size`); `None` — as before. `group` —
    advantage-group label; the stream's seat is appended to it.

    `index` — global game number: its parity decides which seat we play.
    Without alternation the seat alone gives +$1.3k and 58% wins (evaluation
    protocol), and policy improvement would be indistinguishable from it.
    """
    runner = runner or make_runner(cfg.rollout)
    rc = cfg.rollout
    decode = decode_mode(our_decode if our_decode is not None else ppo_decode(cfg))
    # Teacher log-frequencies — from the same file the policy started from.
    # Read once per rollout and stored in the opponent cache: it is the only
    # long-lived dict that reaches here.
    cache0 = opp_net_cache if opp_net_cache is not None else {}
    pkey = "prior:" + (cfg.ckpt or "")
    if pkey not in cache0:
        cache0[pkey] = load_prior(cfg.ckpt, rc.device) if cfg.ckpt else None
    install(play, net, rc.device, decode, record=record, prior=cache0[pkey],
            slim=not rc.keep_obs, who=f"trained policy ({cfg.ckpt})",
            logp_every=getattr(rc, "logp_every", 1))
    play._reset_flow()

    mirror = opponent.kind == "self"
    cache = cache0
    opp_mod = None
    if mirror:
        opp_agent = play.agent
    elif opponent.kind == "package":
        # A package is loaded ONCE per rollout: its `exec` reads weights from
        # disk each time and temporarily removes our modules from `sys.modules`.
        key = "pkg:" + opponent.path
        if key not in cache:
            cache[key] = play._package_agent(opponent.path)
        opp_agent = cache[key]
    else:
        opp_mod = fresh_play_module()
        key = opponent.path
        if key not in cache:
            cache[key] = (NET_FOR_OPPONENT(key, rc.device)
                          if NET_FOR_OPPONENT is not None
                          else load_net(key, rc.device))
        # The opponent plays the argmax decoder (`policy.BC`) — that is how it
        # would play on the ladder and how `play.py --opp-ckpt` runs it. The
        # rollout temperature is for our exploration, not its.
        pk = "prior:" + key
        if pk not in cache:
            cache[pk] = load_prior(PRIOR_PATH_FOR.get(key, key), rc.device)
        install(opp_mod, cache[key], rc.device, decode_mode(POL.BC),
                record=False, prior=cache[pk],
                who=f"league opponent {opponent.key} ({key})")
        opp_mod._reset_flow()
        opp_agent = opp_mod.agent

    me = 0 if index % 2 == 0 else 1
    agents = [play.agent, opp_agent] if me == 0 else [opp_agent, play.agent]
    # The generator is fixed for EACH game: env seed and policy seed coincide,
    # and the per-seed difference measures only the model (play.main).
    # ⚠ Group collection separates them: the K games of a group share ONE env
    # seed but each has its own sampling seed — otherwise they would be bitwise identical.
    ss = seed if sample_seed is None else int(sample_seed)
    torch.manual_seed(ss)
    n_ex = int(getattr(rc, "explore_turns", 0) or 0)
    ex_turns = None
    if n_ex > 0 and record:
        # sampled turns — per game (from the sampling seed), via Python's
        # generator, not torch: per-turn torch seeding (`crn`) leaves them alone.
        # `explore_shared` (25.09): turns from the ENV seed, shared by the group.
        # Then a group's games run bitwise identical up to the first shared
        # turn, each picks its own there, then argmax again — so the in-group
        # outcome difference is the effect of decisions on those turns, without
        # the noise of games diverging on other turns (branching with common numbers).
        ex_seed = seed if getattr(rc, "explore_shared", False) else ss
        ex_turns = set(random.Random(ex_seed * 7_919 + 13).sample(range(719), min(n_ex, 719)))
        agents = _explore_agents(agents, me, mirror, ex_turns)
    if getattr(rc, "crn", False) or branch is not None:
        agents = _crn_agents(agents, me, mirror, ss, seed, branch)
    rew, dead = runner.run(agents, seed)
    recs = play._NET.pop("record", [])
    play._NET["record"] = (SlimRecord() if not rc.keep_obs else []) if record else None
    if opp_mod is not None:
        opp_mod._NET.pop("record", None)
    kind = {"self": "mirror", "package": "package"}.get(opponent.kind, "bc")
    our = ("ppo" if our_decode is None or decode["sample_units"] > 0
           else "bc")
    # ⚠ The game is void if EITHER side was removed, not only us. A crashed
    # opponent passes from then on, we finish at $140k vs $3k — and such a game
    # went into the buffer, critic targets, league difficulty (`league.record`)
    # and win rate as a "huge win". On 23.09 this happened via an opponent
    # without `prior`; that cause was fixed, the path was not (review 23.09, finding 1).
    void = bool(dead[0]) or bool(dead[1])
    if void and not bool(dead[me]):
        print(f"⚠ game {seed}: opponent {opponent.key} removed by the env — game "
              "excluded from training and stats", flush=True)

    if not record:
        # Acceptance panel: only money and outcome are needed; no turns by construction.
        ep = Episode(seed=seed, seat=me, opponent=opponent.key,
                     money=float(rew[me]), opp_money=float(rew[1 - me]),
                     dead=void, decode=our, opp_decode=kind,
                     index=int(index), sample_seed=ss)
        d = ep.money - ep.opp_money
        ep.outcome = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        return [ep]

    # Both sides' records are in one list — split by player and step.
    by_seat: Dict[int, List[dict]] = {}
    for r in recs:
        o = r["obs"]
        by_seat.setdefault(int(o["player"]), []).append(r)
    seats = [me]
    if mirror and rc.both_seats:
        # Mirror self-play: experience of BOTH sides is on-policy, the sample
        # doubles for free. Not so against a frozen opponent — there the other
        # side is not our policy.
        seats = sorted(by_seat)

    out: List[Episode] = []
    for seat in seats:
        # ⚠ `dict.get(k, default)` ALWAYS evaluates default, so the fallback
        # `day*24 + hour` was computed here even when `step` existed. With a
        # full `obs` in the record this went unnoticed; on a slim record
        # (`SlimRecord`) — KeyError. Compute the fallback only when needed.
        rr = sorted(by_seat.get(seat, []), key=lambda r: _step_of(r["obs"]))
        if rc.max_turns:
            rr = rr[:rc.max_turns]
        if not rr:
            continue
        margins = [float(r["obs"]["farms"][seat]["money"]
                         - r["obs"]["farms"][1 - seat]["money"]) for r in rr]
        margins.append(float(rew[seat] - rew[1 - seat]))
        phis = [(float(r["obs"]["phi"]) if "phi" in r["obs"] else NW.phi_obs(r["obs"]))
                for r in rr]
        ep = Episode(seed=seed, seat=seat, opponent=opponent.key,
                     money=float(rew[seat]), opp_money=float(rew[1 - seat]),
                     dead=void, decode=our, opp_decode=kind,
                     index=int(index), sample_seed=ss,
                     group=(f"{group}:{seat}" if group else ""),
                     shops=(rr[-1]["obs"].get("shops") or _shops_of(rr[-1]["obs"])))
        d = ep.money - ep.opp_money
        ep.outcome = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        rws = _turn_rewards(len(rr), ep.money, ep.opp_money, margins, cfg.reward, phis)
        for i, r in enumerate(rr):
            on = ex_turns is None or _step_of(r["obs"]) in ex_turns
            ep.turns.append(Turn(
                enc=r["enc"], legal0=r["legal0"], lat=r["lat"],
                # on a deterministic turn the rollout `log π` was computed by a
                # different decoder — wrong as a drift check, don't store it
                logp=(r["logp"] if on else None),
                reward=rws[i], terminated=(i == len(rr) - 1), truncated=False,
                obs=(copy.deepcopy(r["obs"]) if rc.keep_obs else r["obs"]),
                explore=on))
        out.append(ep)
    return out


_DET = {"sample": False, "sample_units": 0.0, "sample_qty": 0.0}


def _explore_agents(agents, me, mirror, ex_turns):
    """Sparse exploration (`rollout.explore_turns`): outside turns `ex_turns`
    our turn is deterministic — the `play._NET` decoder mode is swapped for
    the turn and restored. In mirror — both sides (both are us)."""
    def wrap(fn):
        def f(obs, config=None):
            if _step_of(obs) in ex_turns:
                return fn(obs, config)
            saved = {k: play._NET.get(k) for k in _DET}
            play._NET.update(_DET)
            try:
                return fn(obs, config)
            finally:
                play._NET.update(saved)
        return f
    out = list(agents)
    out[me] = wrap(agents[me])
    if mirror:
        out[1 - me] = wrap(agents[1 - me])
    return out


def _crn_agents(agents, me, mirror, ss, env_seed, branch):
    """Agent wrappers for common random numbers (`rollout.crn`).

    Our turn t: `torch.manual_seed(turn_seed(ss, t))` right before the
    decision — turn randomness depends on (sampling seed, turn), not on how
    many numbers were drawn before. `branch = (t, alt)` — branching: on turn t
    the seed is `turn_seed(ss, t, alt)`, other turns unchanged (vine probe).
    Opponent: its own stream `turn_seed(env_seed, t, 7)` — identical across a
    group's games — and our generator state is saved and restored around its
    turn. In mirror the second side is also us, with its own offset.
    """
    bt, alt = branch if branch is not None else (-1, 0)

    def ours(fn, side):
        def f(obs, config=None):
            st = _step_of(obs)
            torch.manual_seed(turn_seed(ss, st, side * 13 + (alt if st == bt else 0)))
            return fn(obs, config)
        return f

    def theirs(fn):
        def f(obs, config=None):
            keep = torch.get_rng_state()
            torch.manual_seed(turn_seed(env_seed, _step_of(obs), 7))
            try:
                return fn(obs, config)
            finally:
                torch.set_rng_state(keep)
        return f

    out = list(agents)
    out[me] = ours(agents[me], 0)
    out[1 - me] = ours(agents[1 - me], 1) if mirror else theirs(agents[1 - me])
    return out


def score(critic, eps: List[Episode], device: str = "cpu") -> None:
    """Set `V(s)` as of collection time. One point for both serial and
    parallel collection: weights don't change between rollout games, so
    "compute along the way" and "compute after" give the same number, and the
    worker needs no tower at all."""
    if critic is None:
        return
    # All rollout turns in ONE call, on GPU at batch 1024 (review 23.09,
    # finding 7): previously the critic was called per game at batch 64 —
    # ~1 000 passes with numpy rebuilds per update, all sitting inside
    # "collection". Memory is fine: the trainer on the same GPU runs the tower
    # at the same batch 1024 (`ppo.minibatch`). V depends on batch size at
    # ~1e-6 (different kernel) — bitwise equality was never promised here.
    turns = [t for ep in eps for t in ep.turns]
    if not turns:
        return
    bs = 1024 if str(device).startswith("cuda") else 64
    kw = {"batch": bs} if "batch" in critic.value.__code__.co_varnames else {}
    for t, vv in zip(turns, critic.value(turns, device, **kw)):
        t.value = float(vv)


def collect_serial(net, plan, cfg, runner=None, progress=True,
                   offset: int = 0, tag: str = "",
                   eval_mode: bool = False) -> List[Episode]:
    """Single-process rollout: list of opponents -> list of experience streams.

    `offset` — global number of the chunk's first game. Both seed and parity
    (which seat we play) depend on it, so a chunk must know its place in the
    plan: otherwise all chunks would start on the same seat and the seat alone
    would give +$1.3k and 58% wins (exactly how `play.py --index0` works).

    ⚠ One line per game and `flush=True`: the repo rule about buffered output
    was written after a head-to-head that ran 45 minutes in total silence.
    """
    rc = cfg.rollout
    eps: List[Episode] = []
    cache: Dict[str, Any] = {}
    t0 = time.time()
    K = int(getattr(rc, "group_size", 1) or 1)
    for i, opp in enumerate(plan):
        gi = offset + i
        # `eval_mode` — acceptance panel (`collect_eval`): own seed block,
        # argmax on our side, no turn recording
        if eval_mode:
            got = play_episode(net, opp, rc.eval_seed0 + gi, gi, cfg, runner,
                               cache, our_decode=POL.BC, record=False)
        elif K > 1:
            env_seed, samp = group_seeds(rc.seed0, gi, K)
            got = play_episode(net, opp, env_seed, gi, cfg, runner, cache,
                               sample_seed=samp,
                               group=f"{env_seed}:{opp.key}")
        else:
            got = play_episode(net, opp, rc.seed0 + gi, gi, cfg, runner, cache)
        eps += got
        if progress or tag:
            el = time.time() - t0
            eta = el / (i + 1) * (len(plan) - i - 1)
            if got:
                m = got[0]
                line = (f"game {gi + 1} vs {opp.key}: "
                        f"${m.money:,.0f} / ${m.opp_money:,.0f}, turns "
                        f"{sum(len(e) for e in got)}")
            else:
                line = f"game {gi + 1} vs {opp.key}: empty"
            # In a chunk the ETA is computed by the parent: a chunk would
            # estimate from its own share and be off the more uneven the chunks.
            print(f"{tag}{line}" if tag else
                  f"  {line}, {el / 60:.1f} min, ~{eta / 60:.0f} min left",
                  flush=True)
    return eps


def group_seeds(seed0: int, gi: int, K: int) -> Tuple[int, int]:
    """Group collection: (env seed, sampling seed) of the game with global number gi.

    A block of 2K consecutive games (K with us on seat 0, K on seat 1 — seat by
    number parity) plays ONE env layout `seed0 + block start`, while sampling
    is seeded by the game's own number — each its own. Blocks share no env
    seeds with each other or with the next update (`seed0` shifts by
    `episodes`).
    """
    return int(seed0 + gi - gi % (2 * K)), int(seed0 + gi)


def check_group_plan(plan, K: int) -> None:
    """Plan for group collection: blocks of 2K with one opponent. Otherwise a
    group would mix opponents, and the baseline would mix different tasks."""
    if K <= 1:
        return
    if len(plan) % (2 * K):
        raise SystemExit(f"rollout.group_size = {K}: plan has {len(plan)} games, "
                         f"needs a multiple of {2 * K} (K games per seat)")
    for b in range(0, len(plan), 2 * K):
        keys = {m.key for m in plan[b:b + 2 * K]}
        if len(keys) != 1:
            raise SystemExit(f"group plan: block {b // (2 * K)} has opponents "
                             f"{sorted(keys)} — plan not from `League.plan(group=K)`")


def collect(net, plan, cfg, critic=None, runner=None, progress=True) -> List[Episode]:
    """Rollout: list of opponents -> list of experience streams.

    `rollout.jobs > 1` fans games out to processes (`collect_parallel`). Games
    are independent by construction — env seed and policy seed are set per game
    (`torch.manual_seed`), the `FlowTracker` flow is reset, the opponent is
    reinstalled — so fan-out must give a BITWISE identical rollout. This is not
    an argument but a probe that compares `log π`, reward
    and banks chunk against chunk.
    """
    jobs = int(getattr(cfg.rollout, "jobs", 1) or 1)
    check_group_plan(plan, int(getattr(cfg.rollout, "group_size", 1) or 1))
    # Where the net itself lives is where the critic estimate is computed:
    # `score` runs on it, not on `rollout.device`, which refers to workers.
    here = str(next(net.parameters()).device)
    if getattr(cfg.rollout, "serve", False):
        # Inference server: J weightless processes plus one GPU with a CUDA graph.
        # It assigns devices itself (workers on cores, net on GPU), so the
        # `here` vs `rollout.device` check below does not apply to it.
        from infer_client import collect_served
        eps = collect_served(net, plan, cfg, workers=jobs, progress=progress,
                             server_device=cfg.rollout.serve_device,
                             graph=cfg.rollout.serve_graph,
                             serve_opponents=cfg.rollout.serve_opponents,
                             min_batch=cfg.rollout.serve_min_batch,
                             max_wait=cfg.rollout.serve_max_wait)
    elif jobs > 1:
        eps = collect_parallel(net, plan, cfg, progress=progress)
    else:
        if here.split(":")[0] != cfg.rollout.device.split(":")[0]:
            raise ValueError(
                f"net is on {here}, but rollout asks for {cfg.rollout.device}: with "
                "one process the game is played by THIS net, i.e. batch "
                "1 on the GPU (10.78 ms vs 2.64 on a core, R6). Devices can "
                "only be split with `rollout.jobs > 1`")
        eps = collect_serial(net, plan, cfg, runner, progress)
    score(critic, eps, here)
    return eps


# ---------------------------------------------------------------- fan-out

def _blocks(n: int, jobs: int) -> List[Tuple[int, int]]:
    """Contiguous chunks of roughly equal length. -> [(offset, length)].

    ⚠ Unlike `play.py._fan_out`, chunks HERE need not have even length. There
    parity is needed because a chunk is its own evaluation and must be
    seat-balanced; here the seat is chosen by the GLOBAL game number sent to
    the worker (`offset`), and balance holds over the whole rollout regardless
    of how it is sliced. Requiring parity would halve fan-out on an 8-game
    rollout, and we don't collect more than eight games per update anyway.
    """
    if n <= 0:
        return []
    J = max(1, min(int(jobs), n))
    if J == 1:
        return [(0, n)]
    base, rest = divmod(n, J)
    out, off = [], 0
    for i in range(J):
        cnt = base + (1 if i < rest else 0)
        out.append((off, cnt))
        off += cnt
    return out


def collect_parallel(net, plan, cfg, progress=True) -> List[Episode]:
    """Same rollout via processes. -> list of streams in plan order.

    Design copied from `play.py --jobs`, along with its pitfalls:

    - **core quota, not `nproc`**: `play.cpu_quota` reads the cgroup, because
      a container may show 28 cores with 15 actually available, and fanning
      out by `nproc` launches twice as many processes as cores;
    - **one torch thread per chunk**: otherwise J processes each claim all
      cores and fight. ⚠ This is also a PARITY requirement: thread count
      changes the reduction order in CPU matmuls, i.e. the last bits of the
      logits. The parent is also pinned to one thread during fan-out —
      otherwise "parallel" and "serial" would give different numbers for a
      reason unrelated to fan-out;
    - **output is read as a STREAM**, not via `communicate()` at the end:
      otherwise progress exists but is visible only after completion, i.e.
      it doesn't exist (45 minutes of silence in a head-to-head — exactly this).

    Weights go to the worker as a file: `state_dict` is written to a temp dir
    next to the run, not taken from `cfg.ckpt` — the policy is already
    training, and the starting checkpoint is unrelated to it.

    MEASURED 23.09 (15 cores, load 2.1-2.8), 8 full self-play games,
    11 504 decisions: 345.5 s with one process vs 55.3 s with eight (×6.25),
    bitwise parity. Table and analysis — `config.RolloutCfg.jobs`.
    """
    import pickle
    import subprocess
    import tempfile
    import threading

    rc = cfg.rollout
    n = len(plan)
    cap = play.cpu_quota()
    J = max(1, min(int(rc.jobs), n, cap))
    if int(rc.jobs) > cap:
        print(f"chunks {rc.jobs} exceed the core quota ({cap}), capping at {cap}",
              file=sys.stderr, flush=True)
    blocks = _blocks(n, J)
    if len(blocks) == 1:
        return collect_serial(net, plan, cfg, None, progress)

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    tmp = tempfile.mkdtemp(prefix="rl_collect_")
    try:
        wpath = os.path.join(tmp, "weights.pt")
        torch.save(net.state_dict(), wpath)
        worker = os.path.join(HERE, "collect_worker.py")
        procs, outs = [], []
        t0 = time.time()
        for bi, (off, cnt) in enumerate(blocks):
            task = os.path.join(tmp, f"task{bi}.pkl")
            res = os.path.join(tmp, f"res{bi}.pkl")
            with open(task, "wb") as f:
                pickle.dump({"cfg": cfg, "plan": list(plan[off:off + cnt]),
                             "offset": off, "weights": wpath, "out": res}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            procs.append(subprocess.Popen(
                [sys.executable, "-u", worker, task], stdout=subprocess.PIPE,
                stderr=None, text=True, env=env, bufsize=1))
            outs.append(res)

        done = [0]
        tails = [[] for _ in procs]
        lock = threading.Lock()

        def pump(i, pr):
            for ln in pr.stdout:
                if ln.startswith("__PROG__"):
                    with lock:
                        done[0] += 1
                        k = done[0]
                    el = time.time() - t0
                    if progress:
                        print(f"  {k}/{n} games, {el / 60:.1f} min, "
                              f"{k / max(1e-9, el) * 60:.1f} games/min, "
                              f"~{(n - k) / max(1e-9, k / max(1e-9, el)) / 60:.0f} min left"
                              f" | {ln[len('__PROG__'):].strip()}", flush=True)
                else:
                    tails[i].append(ln)
                    if len(tails[i]) > 40:
                        tails[i].pop(0)
            pr.stdout.close()

        th = [threading.Thread(target=pump, args=(i, pr), daemon=True)
              for i, pr in enumerate(procs)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        eps: List[Episode] = []
        for i, pr in enumerate(procs):
            if pr.wait() != 0 or not os.path.exists(outs[i]):
                raise SystemExit(
                    f"chunk {i} failed (rc={pr.returncode}); output tail:\n"
                    + "".join(tails[i][-20:]))
            with open(outs[i], "rb") as f:
                eps += pickle.load(f)
        if progress:
            el = time.time() - t0
            print(f"  rollout: {len(blocks)} processes at a quota of {cap} cores, "
                  f"{el:.0f}s, {el / max(1, n):.1f}s per game", flush=True)
        return eps
    finally:
        torch.set_num_threads(prev_threads)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def rollout_stats(eps: List[Episode]) -> Dict[str, float]:
    """Rollout summary: win rate, money, share of dead games, per opponent.

    ⚠ `win_rate` over the whole rollout is a MIXED quantity and cannot be a
    threshold (N1/R4). Two reasons, both measurable:

    1. **Decoder asymmetry.** We sample (`policy.PPO`), the checkpoint opponent
       plays argmax (`policy.BC`) — as it would on the ladder. The cost of
       this asymmetry was measured: −$1 784 median, 16/48 on the same
       weights. So equal strength gives us a win rate of about 0.33, and 0.60
       means "better than the opponent by about two thresholds", not "clearly better".
    2. **Mirror self-play.** With `both_seats` both streams of one game are
       ours, their outcomes are exactly +1 and −1, and their win rate is
       identically 0.5. Mixing them into the total pulls any win rate toward
       0.5, the more so the more mirror games in the rollout.

    Hence slices are computed alongside: `win_rate/vs_bc` (argmax checkpoints
    only), `win_rate/vs_package` (frozen packages — the only link to the
    ladder), `win_rate/mirror` (must be 0.5, a wiring check).
    The honest admission threshold into history is the `collect_eval` panel,
    where we play argmax too (`train.Run.step`, `rollout.eval_episodes`).
    """
    live = [e for e in eps if not e.dead]
    if not live:
        return {"episodes": len(eps), "dead": len(eps)}
    win = sum(e.outcome > 0 for e in live)
    tie = sum(e.outcome == 0 for e in live)
    st = {"episodes": len(eps), "dead": len(eps) - len(live),
          "turns": sum(len(e) for e in live),
          "win_rate": win / len(live), "tie_rate": tie / len(live),
          "money_median": float(np.median([e.money for e in live])),
          "margin_median": float(np.median([e.money - e.opp_money for e in live]))}
    if any(len(e) for e in live):
        # The acceptance panel has no turns by construction (`record=False`),
        # and "mean reward 0.0" there would mean not "rewards are zero" but
        # "nothing to average" — better not to print such a metric at all.
        st["reward_mean"] = float(np.mean([sum(t.reward for t in e.turns)
                                           for e in live]))
    for kind in ("bc", "package", "mirror"):
        sub = [e for e in live if e.opp_decode == kind]
        if sub:
            st[f"win_rate/vs_{kind}"] = sum(e.outcome > 0 for e in sub) / len(sub)
            st[f"margin_median/vs_{kind}"] = float(
                np.median([e.money - e.opp_money for e in sub]))
            st[f"games/vs_{kind}"] = len(sub)
    for key in sorted({e.opponent for e in live}):
        sub = [e for e in live if e.opponent == key]
        st[f"loss_rate/{key}"] = sum(e.outcome < 0 for e in sub) / len(sub)
        st[f"games/{key}"] = len(sub)
    return st


def collect_eval(net, plan, cfg, runner=None, progress=True) -> List[Episode]:
    """Acceptance panel: same opponents, but argmax on BOTH sides.

    Why separate from the rollout. The rollout must sample — otherwise there is
    neither exploration nor a well-defined `log π`. The checkpoint opponent
    must play argmax — otherwise it is not what it would be on the ladder. The
    two can't be compared (N1/R4: −$1 784, 16/48 on the same weights), yet the
    admission threshold into history (`league.promote`) rests on exactly that
    comparison. So acceptance is separate games where we play argmax too:
    exactly what this checkpoint will be on the ladder, against exactly the
    same opponent.

    No turn recording here (`record=False`): these games are off-policy and do
    not go into the buffer, and recording costs 18% of game time.

    Seeds come from a SEPARATE block (`eval_seed0`) so the panel shares none
    with the rollout: otherwise "improvement" on acceptance would partly be
    the very thing the policy just trained on.
    """
    rc = cfg.rollout
    if getattr(rc, "serve", False):
        # ⚠ Via the inference server, like main collection. Before 24.09 the
        # panel ran in one process directly on the TRAINER's net: with
        # `train_device=cuda` input was built on CPU (`rollout.device`), the net
        # was on GPU, and EVERY acceptance game crashed on turn one — $3 000 vs
        # $140-165k, and admission into history never passed (rl1, update 21).
        # It's also faster: 8 games in parallel instead of sequential batch 1.
        from infer_client import collect_served
        return collect_served(net, plan, cfg,
                              workers=min(int(rc.jobs or 1), len(plan)),
                              progress=False, server_device=rc.serve_device,
                              graph=rc.serve_graph,
                              serve_opponents=rc.serve_opponents,
                              eval_mode=True)
    here = str(next(net.parameters()).device)
    if here.split(":")[0] != rc.device.split(":")[0]:
        raise ValueError(
            f"acceptance panel: net is on {here}, but input is built on {rc.device} — "
            "every game would crash on turn one. Enable `rollout.serve`")
    eps: List[Episode] = []
    cache: Dict[str, Any] = {}
    t0 = time.time()
    for i, opp in enumerate(plan):
        eps += play_episode(net, opp, rc.eval_seed0 + i, i, cfg, runner, cache,
                            our_decode=POL.BC, record=False)
        if progress:
            el = time.time() - t0
            # ⚠ `if m` instead of `if m is not None` would lie here: `Episode`
            # defines `__len__`, and an acceptance game has zero turns by
            # construction (`record=False`), i.e. it is FALSY as an object.
            m = eps[-1] if eps else None
            print(f"  acceptance {i + 1}/{len(plan)} vs {opp.key}: "
                  + (f"${m.money:,.0f} / ${m.opp_money:,.0f}"
                     if m is not None else "empty")
                  + f", {el / 60:.1f} min", flush=True)
    return eps
