"""Playing the trained FarmNet: decoding the BERT heads into environment actions.

The turn itself (masks before sampling, sampling, `log π`) is assembled by `policy.py`; here
are the network input, decoder modes, games and measurement. The order of orders is imposed
deterministically (sells first, then buys) -- the engine executes them by
list index, and a sale that finances a buy must come first. This is the
price of BERT mode, see the net architecture note.

Run: ./env/bin/python experiments/net/play.py -n 6 --opp starter
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import features  # noqa: E402
from model import FarmNet, ModelConfig, N_OPS  # noqa: E402
from bc_train import UNIT_OPS, FlowTracker  # noqa: E402
from features import MAX_UNITS  # noqa: E402
from masks import legal_ops  # noqa: E402
import envcfg  # noqa: E402
import policy as POL  # noqa: E402

_NET = {}
# Decoder settings -- ONE place for the agent, the command line and the package: market
# gates are drawn by sampling, units by argmax, quantity by the chain median, the
# CARE habit is on (`policy.BC`). Previously the number of market passes lived in three places
# (agent, CLI, the package's decode.json) and silently diverged since 68fd64a: the ladder played
# with one pass, the measurements with three. The PPO rollout mode is `policy.PPO` (`--ppo`).
#
# Quantity in BC mode is the median, not a sample: on 25 corpus games
# mega_p8's median is closer to the teacher on every head (MAE sell 1.77 vs 2.60,
# buy 5.36/7.70, seed 0.52/0.73, hire 0.18/0.26, pickup 0.06/0.10), and it is what
# the net played with before R2. The price of the median is a volume shift: buy averages 5.0 against
# 8.0 for the teacher, sampling gives 8.1 (R2, 22.09).
#
# The distribution family (`policy.FAMILY`: β·prior on op and target, gate lower
# bound) is part of the decoder settings, not a separate key: the submission package, the measurement and the
# rollout must play in one family. The mode (`--ppo`) does NOT override it:
# temperatures and family are different things.
DECODE_DEFAULTS = dict(POL.BC, **POL.FAMILY)
# Feed top-up for a hungry herd (policy.run_turn, 28.09): own net only.
DECODE_DEFAULTS["feed_guard"] = False
# Last turn of the season: sell the whole shed, buy nothing (policy.run_turn).
DECODE_DEFAULTS["end_sell"] = False
# Season end: loaded units walk to the shed in time and drop the cargo (29.09).
DECODE_DEFAULTS["end_haul"] = False
# Bias added to the BUILD_COOP/BUILD_PASTURE logits (29.09, a manual review of Champion·Guild:
# on new land the clone plants, the teacher builds for animals; herd -2..7 on d6-d15).
DECODE_DEFAULTS["build_bias"] = 0.0
# Execution order of the net's market blocks (None -- order from the checkpoint or
# policy.MARKET_ORDER). 28.09: "animals before goods" -- otherwise wheat for feed
# ate the budget for an animal (cap 0 in 26% of the teacher's animal purchases).
DECODE_DEFAULTS["mkt_order"] = None
# Market gates by accumulated frequency instead of the 0.5 threshold (policy.Sampler.gate).
DECODE_DEFAULTS["gate_rate"] = False
# Heads where the gate accumulator acts ("" -- all). 29.09: on all heads
# clone 56254996 -$13.8k (small buy P's accumulated), sells are a separate case.
DECODE_DEFAULTS["gate_rate_heads"] = ""
# Unit targets by greedy matching without repeats (policy.Sampler.targets).
DECODE_DEFAULTS["dedup_targets"] = False
# Feed rule only on real escape risk (policy.run_turn(feed_guard_risk)).
DECODE_DEFAULTS["feed_guard_risk"] = False
# The hour from which the feed rule in risk mode feeds all unfed animals.
DECODE_DEFAULTS["feed_guard_hour"] = POL.FEED_GUARD_HOUR
# Land purchase by accumulated frequency (policy.Sampler.land).
DECODE_DEFAULTS["land_rate"] = False
# Expert router (30.09, teacher blend): "" -- one expert (the --ckpt net).
# Rules separated by ';': "d0-d1:group=i,group=j" -- on days d0..d1 the group's heads
# are taken from expert i (0 -- the --ckpt net, 1.. -- --experts). Groups: all, units,
# market, sell, buy, stock (seed+animal), hire, land. A later rule overrides.
# "conf" -- on every turn the expert with the most confident unit policy.
DECODE_DEFAULTS["route"] = ""
# Last turn of the day: if hands will not fit into the shed at the night dump -- sell the excess
# from the shed in this same turn (policy.night_room, 30.09).
DECODE_DEFAULTS["night_room"] = False
DECODE_DEFAULTS["night_room_hour"] = 23
# "stock" -- sell the good with the largest remainder (first version); "value" -- by price:
# save only what is worth more than the loss from the knocked-down price (policy.night_room_value).
DECODE_DEFAULTS["night_room_mode"] = "stock"

_GROUPS = {"units": ("op_logits", "target_logits", "qty_logits"),
           "sell": ("sell_logits",), "buy": ("buy_logits",),
           "stock": ("seed_logits", "animal_logits"),
           "hire": ("hire_logits",), "land": ("land_logits",)}
_GROUPS["market"] = sum((_GROUPS[g] for g in ("sell", "buy", "stock", "hire", "land")), ())
_GROUPS["all"] = _GROUPS["units"] + _GROUPS["market"]


def _style_idx(sd0, name):
    """Style-team of a style checkpoint: team name or number; "" -- the "common" one."""
    teams = sd0.get("style_teams") or []
    if not teams:
        if name:
            raise SystemExit("style is given, but the checkpoint has no styles")
        return None
    if name == "":
        return torch.tensor([len(teams)])
    i = int(name) if name.isdigit() else teams.index(name)
    return torch.tensor([i])


def _load_expert(path, device="cpu"):
    """Additional expert: a net and its prior; decoder settings come from the --ckpt net.
    `path@style` -- a style checkpoint with the given team (number or name).
    A dict instead of a path -- an expert that arrived inside the package's weights.pt."""
    if isinstance(path, dict):
        sd0, sty, path = path, path.get("style", ""), "<package>"
    else:
        path, _, sty = path.partition("@")
        sd0 = torch.load(path, map_location="cpu")
    m = FarmNet(ModelConfig.from_ckpt(sd0))
    m.load_bc(sd0.get("model", sd0))
    m.eval()
    pr = sd0.get("prior")
    return {"m": m.to(torch.device(device)), "path": path, "style": _style_idx(sd0, sty),
            "prior": ({k: v.to(torch.device(device)) for k, v in pr.items()} if pr else None)}


class _Router(torch.nn.Module):
    """Expert router over the global trunk token of the main net (router_train.py)."""
    def __init__(self, d, k):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(),
                                       torch.nn.Linear(256, 128), torch.nn.GELU(),
                                       torch.nn.Linear(128, k))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def _load_router(src):
    """A router_train.py file or a dict from the package's weights.pt -> (Router, experts)."""
    ck = torch.load(src, map_location="cpu") if isinstance(src, str) else src
    r = _Router(int(ck["dim"]), len(ck["experts"]))
    r.load_state_dict(ck["state"])
    r.eval()
    return r, list(ck["experts"])


def _route(fams, spec, day, unit_mask):
    """Assembling the turn from the experts' outputs (each already through `policy.family`)."""
    if len(fams) == 1 or not spec:
        return fams[0], 0
    if spec == "conf":
        live = unit_mask.bool()
        sc = []
        for o in fams:
            p = torch.softmax(o["op_logits"], -1).amax(-1)[0]
            sc.append(float(p[live].mean()) if live.any() else 0.0)
        i = int(np.argmax(sc))
        return fams[i], i
    pick = {k: 0 for k in _GROUPS["all"]}
    for rule in spec.split(";"):
        win, _, asg = rule.partition(":")
        d0, _, d1 = win.partition("-")
        if not (int(d0) <= day <= int(d1 or d0)):
            continue
        for a in asg.split(","):
            g, _, i = a.partition("=")
            for k in _GROUPS[g.strip()]:
                pick[k] = int(i)
    o = POL.Logits({**fams[0], **{k: fams[i][k] for k, i in pick.items()}})
    o.spec = fams[0].spec
    return o, max(pick.values())


_FLOW = {}


def _reset_flow():
    """Start of a game: the pressure tracker starts from zero."""
    _FLOW.clear()


def _load(path=None, device="cpu"):
    """Net from a `bc_train.py` checkpoint. There is one architecture (frozen 22.09),
    loading is STRICT: a checkpoint of a different shape is a loud refusal, not silently
    zero tensors. Checkpoints from before the freeze (`n1_base`, `base_p01` and all
    ablation arms) do not load with the new code: for them use the `pre-freeze` tag or
    the frozen packages in `raw-content/opponents/`."""
    if not _NET:
        if not path:
            raise SystemExit("--ckpt is required: there is no default checkpoint")
        sd0 = torch.load(path, map_location="cpu")
        w = sd0.get("model", sd0) if isinstance(sd0, dict) else sd0
        try:
            m = FarmNet(ModelConfig.from_ckpt(sd0))
            # `load_bc`, not `load_state_dict`: BC checkpoints from before R1 have no
            # value tower. Strictness does not suffer -- every key of the policy
            # is checked, and the tower is not computed in play at all
            # (want_value=False) and is initialized from the BC weights.
            m.load_bc(w)
        except RuntimeError as e:
            raise SystemExit(f"checkpoint {path} is of a different architecture (before the freeze? "
                             f"tag pre-freeze): {str(e)[:400]}")
        m.eval()
        _NET["m"] = m.to(torch.device(device))
        _NET["dev"] = torch.device(device)
        # Logarithm of the prior frequencies of op and targets: the net was TRAINED with it (logit
        # adjustment in `bc_train.loss_fn`), so without it it plays not the
        # distribution it was trained on (N2). It is added by `policy.family`.
        pr = sd0.get("prior") if isinstance(sd0, dict) else None
        _NET["prior"] = ({k: v.to(torch.device(device)) for k, v in pr.items()}
                         if pr else None)
        # With what strength τ the correction entered the loss (`bc_train --tau`). The played
        # family matches the trained one only at β = τ; `main` checks this.
        ck_cfg = sd0.get("cfg") if isinstance(sd0, dict) else None
        _NET["prior_tau"] = (float(ck_cfg["tau"]) if isinstance(ck_cfg, dict)
                             and ck_cfg.get("tau") is not None else None)
        # Market execution order -- of the teacher the net was trained on (27.09).
        mo = ck_cfg.get("mkt_order") if isinstance(ck_cfg, dict) else None
        _NET["mkt_order"] = tuple(mo.split(",")) if mo else None
        # Decoder settings that travel WITH the weights (28.09: feed rule).
        # The package and local play read them from here -- there is no second path.
        # style checkpoint: the "common" style by default, `--style` sets the team
        _NET["style"] = (_style_idx(sd0, os.environ.get("FARMCORE_STYLE", sd0.get("style", "")))
                         if isinstance(sd0, dict) else None)
        # experts that arrived in the package (build.py --experts): the router is in decode
        if isinstance(sd0, dict) and sd0.get("experts"):
            _NET["experts"] = [_load_expert(ex, device) for ex in sd0["experts"]]
        if isinstance(sd0, dict) and sd0.get("router"):
            _NET["router_src"] = sd0["router"]
        dec = sd0.get("decode") if isinstance(sd0, dict) else None
        if dec:
            bad = set(dec) - set(DECODE_DEFAULTS)
            if bad:
                raise SystemExit(f"{path}: unknown decoder settings {sorted(bad)}")
            _NET.update(dec)
    return _NET["m"], _NET["dev"]


# Modules of the submission package that share names with ours (build.MODULES + extract_features).
_PKG_MODULES = ("play", "model", "features", "bc_train", "masks", "counts", "effects",
                "envcfg", "backbone", "utils", "value_head", "extract_features",
                "policy")


def _package_agent(main_py):
    """Agent from an assembled package (`main.py` next to `weights.pt`) -- with its OWN code.

    Why. Frozen opponents (`raw-content/opponents/n1_base`, `base_p01`)
    are pre-freeze nets, their checkpoints do not load with the new code. The package carries its own
    code, but its `import model` in our process would return OUR `model` from
    `sys.modules`: the package would fail on loading weights, and the environment would silently give
    it $3,000 (verified: that is what happened). So our modules are removed from `sys.modules`
    for the time of exec and then returned; the package's functions hold
    their modules through their own globals after that as well.
    """
    main_py = os.path.abspath(main_py)
    d = os.path.dirname(main_py)
    saved = {k: sys.modules.pop(k) for k in _PKG_MODULES if k in sys.modules}
    n_thr = torch.get_num_threads()          # the package's main.py sets 2 threads
    sys.path.insert(0, d)
    try:
        g = {}
        exec(compile(open(main_py).read(), main_py, "exec"), g)
        for k in _PKG_MODULES:
            sys.modules.pop(k, None)
    finally:
        sys.path.remove(d)
        sys.modules.update(saved)
        torch.set_num_threads(n_thr)
    # The entry is the LAST callable of the namespace, as in `kaggle_environments`
    # (`agent.get_last_callable`), not `g["agent"]`. For ladder scripts
    # these are different functions: v47 defines `agent` nine times and wraps it in
    # `_y_agent_shopherd`, so by name the league would have played the INNER layer, not
    # v47 (23.09, an internal script-match check). `play.py --opp` for a package
    # without weights hands the path to the environment and takes the last one -- both roads are now one.
    return [v for v in g.values() if callable(v)][-1]


def _ostats():
    """Counters of what the decoder CUT after the choice. The net asks for a quantity, but
    less can be written out: not enough money, shed room, goods in hands or
    order slots. Under PPO this is a gradient bias -- the policy gets
    a reward for an action it did not take. Since R3 (22.09) the caps stand BEFORE
    sampling, and `clip` must be empty: non-empty is a mask error. `env` is
    substitutions by the environment (DROP -> PLACE), and
    habit closures [the net would have chosen exactly this]."""
    return _NET.setdefault("order_stats",
                           {"turns": 0, "over": 0, "want": 0, "cut": 0,
                            "kinds": {}, "clip": {}, "env": {}})


def agent(obs, config=None):
    # The environment configuration is read ONCE per process. Previously the second argument
    # was ignored, and five engine quantities lived as constants in four files
    # (A7). A mismatch on those of them baked into the weights is a loud refusal.
    envcfg.configure(config)
    m, dev = _load()
    p = obs["player"]
    farm = obs["farms"][p]
    priv = obs.get("private") or {}
    shed = priv.get("shed") or {}
    seeds = dict(priv.get("seeds") or {})
    invs = priv.get("inventories") or [{}]
    pos = [tuple(q) for q in ([farm["farmer"]] + farm["hands"])]

    # --- external pressure and accumulated sales: the same FlowTracker as in training ---
    st = int(obs.get("step", obs["day"] * 24 + obs["hour"]))
    f = _FLOW.get(p)
    if f is None or st <= f["step"]:
        f = _FLOW[p] = {"tr": FlowTracker(), "obs": None, "act": None, "step": -1, "acc": {},
                        "land_acc": {}}
    if f["obs"] is not None and st == f["step"] + 1:
        f["tr"].observe(f["obs"], obs, f["act"] or {}, p, st, f["obs"].get("day", 0))
    # The previous turn is our own, if it was exactly at the previous step (as in
    # training). The net does NOT read its columns (the `prev_act` option was removed 29.09), but
    # `features.encode` still fills them, and in play they must be the same
    # quantity as in the cache.
    prev = f["act"] if f["obs"] is not None and st == f["step"] + 1 else None
    e = features.encode(obs, p, extra=dict(f["tr"].extra(), prev_act=prev))
    tt = lambda a: torch.from_numpy(np.asarray(a)).unsqueeze(0).to(dev)

    # Legality on the unit's CURRENT cell, as of the start of the turn -- the same
    # `legal_now` that goes into the net in training: the corrector uses it to estimate the
    # shed at the moment of the market. Below, the action loop edits `seeds` and `shed`, and
    # after it the mask would already be different.
    n_live = int(min(len(pos), MAX_UNITS))
    legal0 = []
    for k in range(MAX_UNITS):
        if k < n_live:
            ok = legal_ops(obs, p, k, pos[k], invs[k] if k < len(invs) else {},
                           dict(seeds), dict(shed))
            legal0.append(np.array([nm in ok for nm in UNIT_OPS], dtype=bool))
        else:
            legal0.append(np.zeros(N_OPS, dtype=bool))
    legal0 = np.stack(legal0)

    # Columns ONLY for the critic (R1). Computed HERE, at the same moment
    # as the rest of the input, and from here go into the rollout record: rebuilding them
    # later from the stored `obs` would mean creating a second place where the
    # `FlowTracker` accumulator lives -- exactly the class of bug that burned us
    # three times (project rule: "training and play must compute ONE quantity").
    vcrit = features.value_extra(obs, p, f["tr"].critic())

    # ONE net pass for the whole turn: both units and market. Refining the turn by
    # itself (measured: -$19,201 and -$20,694) and passes 2-3 over the
    # market with a turn plan (audit 22.09, M5: the net barely read them) were removed;
    # what passes 2-3 used to give, the corrector now gets as derived quantities.
    # want_value=False -- the value head is not used at inference.
    with torch.no_grad():
        out = m(tt(e["tiles_me"]).float(), tt(e["tiles_op"]).float(),
                tt(e["units"]).float(), tt(e["unit_mask"]),
                tt(e["products"]).float(), tt(e["glob"]).float(),
                crops=tt(e["crops"]).float(), animals=tt(e["animals"]).float(),
                tile_mask=tt(e["tile_mask"]),
                legal=torch.from_numpy(legal0).unsqueeze(0).to(dev),
                vcrit=tt(vcrit).float(),
                want_value=False, style=_NET.get("style"))
    # The turn is assembled by `policy.run_turn`: masks before sampling (R3), the CORN chain for
    # quantities (R2), `log π` over the same masks (R5). The same function recomputes
    # the turn from a recorded latent -- there is no second copy of the decoder logic.
    #
    # Target holding (`--commit`), the `--fix shed`/`--fix night` fixes and
    # the `gate`/`effect` decoders were removed: all measured and none taken
    # (hard holding -$14,544, 12/96; `--fix shed` 36/96; `effect` $31k
    # vs $155k on the same seed). Numbers in the project notes, code in git history.
    cfg = {k: _NET.get(k, v) for k, v in DECODE_DEFAULTS.items()}
    # Distribution family -- ONCE, right after the net pass: from then on both
    # sampling and `log π` see the same logits (N2, N4).
    mac = _NET.get("macro")
    if callable(mac):
        mac = mac(obs)
    out = POL.family(out, _NET.get("prior"), cfg["beta"], cfg["floor"], macro=mac)
    if _NET.get("experts") and str(cfg["route"]).startswith("learned"):
        # learned[:δ] -- the router chooses the expert; departing from the main net (0) only when
        # the probability exceeds δ. ONLY the chosen net is computed.
        if "router" not in _NET:
            _NET["router"] = _load_router(_NET.get("router_src"))
        rt, _names = _NET["router"]
        delta = float(str(cfg["route"]).split(":")[1]) if ":" in str(cfg["route"]) else 0.0
        with torch.no_grad():
            h, _, _, _ = m.trunk(tt(e["tiles_me"]).float(), tt(e["tiles_op"]).float(),
                                 tt(e["units"]).float(), tt(e["unit_mask"]),
                                 tt(e["products"]).float(), tt(e["glob"]).float(),
                                 tt(e["crops"]).float(), tt(e["animals"]).float(),
                                 style=_NET.get("style"))
            pr = torch.softmax(rt(h[:, -1].float()), -1)[0]
        who = int(pr.argmax())
        if who != 0 and float(pr[who] - pr[0]) <= delta:
            who = 0
        if who:
            ex = _NET["experts"][who - 1]
            with torch.no_grad():
                o = ex["m"](tt(e["tiles_me"]).float(), tt(e["tiles_op"]).float(),
                            tt(e["units"]).float(), tt(e["unit_mask"]),
                            tt(e["products"]).float(), tt(e["glob"]).float(),
                            crops=tt(e["crops"]).float(), animals=tt(e["animals"]).float(),
                            tile_mask=tt(e["tile_mask"]),
                            legal=torch.from_numpy(legal0).unsqueeze(0).to(dev),
                            vcrit=tt(vcrit).float(), want_value=False, style=ex["style"])
            out = POL.family(o, ex["prior"], cfg["beta"], cfg["floor"], macro=mac)
        rs = _NET.setdefault("route_stats", {})
        rs[who] = rs.get(who, 0) + 1
    elif _NET.get("experts") and cfg["route"]:
        fams = [out]
        with torch.no_grad():
            for ex in _NET["experts"]:
                o = ex["m"](tt(e["tiles_me"]).float(), tt(e["tiles_op"]).float(),
                            tt(e["units"]).float(), tt(e["unit_mask"]),
                            tt(e["products"]).float(), tt(e["glob"]).float(),
                            crops=tt(e["crops"]).float(), animals=tt(e["animals"]).float(),
                            tile_mask=tt(e["tile_mask"]),
                            legal=torch.from_numpy(legal0).unsqueeze(0).to(dev),
                            vcrit=tt(vcrit).float(), want_value=False, style=ex["style"])
                fams.append(POL.family(o, ex["prior"], cfg["beta"], cfg["floor"], macro=mac))
        out, who = _route(fams, cfg["route"], int(obs.get("day", st // 24)),
                          tt(e["unit_mask"])[0])
        rs = _NET.setdefault("route_stats", {})
        rs[who] = rs.get(who, 0) + 1
    if cfg["build_bias"]:
        ol = out["op_logits"].clone()
        for nm in ("BUILD_COOP", "BUILD_PASTURE"):
            ol[..., POL.UNIT_OPS.index(nm)] += float(cfg["build_bias"])
        out["op_logits"] = ol
    pick = POL.Sampler(out, cfg["sample"], cfg["sample_units"], cfg["sample_qty"],
                       acc=f["acc"] if cfg["gate_rate"] else None,
                       acc_heads=cfg["gate_rate_heads"],
                       dedup=cfg["dedup_targets"],
                       land_acc=f["land_acc"] if cfg["land_rate"] else None)
    action, lat, info = POL.run_turn(obs, pick, order=_NET.get("mkt_order"),
                                     feed_guard=bool(cfg["feed_guard"]),
                                     end_sell=bool(cfg["end_sell"]),
                                     end_haul=bool(cfg["end_haul"]),
                                     feed_guard_risk=bool(cfg["feed_guard_risk"]),
                                     feed_guard_hour=int(cfg["feed_guard_hour"]),
                                     night_room_on=bool(cfg["night_room"]),
                                     night_room_hour=int(cfg["night_room_hour"]),
                                     night_room_mode=str(cfg["night_room_mode"]))

    os_ = _ostats()
    os_["turns"] += 1
    os_["want"] += len(action["market"])
    for kk, (dec, lost, want) in info["clip"].items():
        c = os_["clip"].setdefault(kk, [0, 0, 0])
        c[0] += dec; c[1] += lost; c[2] += want
    for kk, v in info["env"].items():
        os_["env"][kk] = os_["env"].get(kk, 0) + v
    if _NET.get("check_orders"):
        # Check against the engine: our whole turn by engine code on a copy of the state,
        # without the opponent. Everything underexecuted is a cut that the R3 mask
        # did not anticipate.
        for kk, (dec, lost, want) in POL.engine_check(obs, action).items():
            c = os_["clip"].setdefault(f"engine did not execute {kk}", [0, 0, 0])
            c[0] += dec; c[1] += lost; c[2] += want
    if _NET.get("record") is not None:
        # PPO rollout and `log π` check (an internal parity script):
        # net input, turn latent and its `log π` computed right here.
        #
        # ⚠ `log π` in the rollout is a CHECK, not training: the trainer recomputes
        # it itself and in a batch (`ppo.recompute_old_logp`, N3(2)), and the ratio
        # uses exactly the recomputation. And here it is computed at batch 1 and costs 1.60 ms
        # per turn -- more than `features.encode`. So the loop may ask
        # not to compute it every turn: `_NET["logp_now"]`. The default `True` is
        # "as before", every turn; whoever turns it off must know what they do to
        # `ppo/logp_drift` (`collect.SlimRecord`).
        lp = (POL.log_prob(out, POL.stack([lat], dev), cfg["sample"],
                           cfg["sample_units"], cfg["sample_qty"])
              if _NET.get("logp_now", True) else None)
        # obs as a copy: the environment edits shared farm structures in place.
        #
        # ⚠ The copy is NOT needed by everyone who records. the log π parity check and everything that later
        # looks at the observation -- yes; the PPO rollout with `rollout.keep_obs = False`
        # -- no: `collect.SlimRecord` throws away from the observation everything
        # except four numbers on entry, and copies them anew. The copy is nevertheless
        # made, HERE, before `append`, and costs 0.69 ms per decision with a
        # Python field of 6.4 ms (measurement 23.09: 693 µs on ordinary
        # port dicts, 1,731 µs on the framework's `Struct`).
        #
        # Default `True` is "as before": whoever does not know about the flag gets a copy.
        # It is turned off only by one who promises the observation will not outlive --
        # `collect.install` with `slim=True`.
        keep = _NET.get("keep_obs", True)
        _NET["record"].append({"obs": copy.deepcopy(obs) if keep else obs,
                               "enc": {**e, "vcrit": vcrit}, "legal0": legal0,
                               "lat": lat,
                               "logp": (float(lp[0]) if lp is not None else None),
                               "action": action, "cfg": cfg})
    f["obs"], f["act"], f["step"] = obs, action, st
    return action


def _report_clip(ost):
    """How many network decisions the decoder wrote out SMALLER than it asked, what the
    environment substituted and how much the habit closed."""
    if not ost or not ost.get("turns"):
        return
    cl = ost.get("clip") or {}
    if not cl:
        print(f"no cuts after choice (turns {ost['turns']:,})")
    else:
        print(f"cut after choice (decisions / units lost / % of "
              f"requested), turns {ost['turns']:,}:")
        for k, (dec, lost, want) in sorted(cl.items(), key=lambda kv: -kv[1][1]):
            print(f"  {k:36s} {dec:6,}  {lost:7,}  {100*lost/max(1,want):5.1f}%")
    for k, v in sorted((ost.get("env") or {}).items()):
        print(f"  environment substitution: {k} -- {v:,}")


def _report(arr, opp, head=""):
    """Summary over games. Dead games are counted SEPARATELY and excluded from money.

    arr: [N,4] -- our money, opponent money, "removed from game" flag, time reserve.
    An agent removed by the environment for exceeding actTimeout gets reward 0. If such
    games are summed with live ones into one median, the measurement stops distinguishing "plays badly"
    from "did not finish", and a more expensive decoder loses for no good reason.
    """
    dead = arr[:, 2] > 0
    ok = arr[~dead]
    if head:
        print(head)
    print(f"\nagainst {opp}, games {len(arr)}")
    if dead.any():
        print(f"  ⚠ REMOVED FROM GAME: {int(dead.sum())} of {len(arr)} -- "
              f"not included in money")
    if len(ok) == 0:
        print("  no live games")
        return
    print(f"  our money: median ${np.median(ok[:,0]):,.0f}  "
          f"min ${ok[:,0].min():,.0f}  max ${ok[:,0].max():,.0f}")
    print(f"  opponent:    median ${np.median(ok[:,1]):,.0f}")
    print(f"  wins: {(ok[:,0] > ok[:,1]).sum()}/{len(ok)}")
    print(f"  time reserve: min {arr[:,3].min():.0f}s of 60 "
          f"(median {np.median(arr[:,3]):.0f}s)")


def cpu_quota():
    """How many cores are REALLY allotted to us.

    `nproc` and `os.cpu_count()` show the node's cores, not the container quota: on the
    box that is 28 against the actual 15 (`cpu.max = 1500000/100000`). A split
    computed from nproc puts twice as many processes as cores.
    """
    for path, div in (("/sys/fs/cgroup/cpu.max", None),
                      ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "period")):
        try:
            txt = open(path).read().split()
            if div is None:
                q, per = txt[0], txt[1]
            else:
                q = txt[0]
                per = open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().strip()
            if q != "max" and int(q) > 0:
                return max(1, int(int(q) / int(per)))
        except (OSError, ValueError, IndexError):
            pass
    return os.cpu_count() or 1


def _fan_out(a):
    """Distribute games across processes and collect the result.

    Each chunk gets its own block of seeds and its own `--index0`, so that the alternation of
    sides stays the same as with one process: otherwise all chunks would start
    from one side and the measurement would be skewed.
    """
    import subprocess
    cap = cpu_quota()
    J = min(a.jobs, a.n)
    if J > cap:
        print(f"chunks {a.jobs} exceed the core quota ({cap}), cutting to {cap}",
              file=sys.stderr)
        J = cap
    # blocks of even size: within a chunk the sides alternate, and each chunk
    # is balanced by itself
    base = max(1, (a.n // J + 1) // 2 * 2)
    blocks, off = [], 0
    while off < a.n:
        cnt = min(base, a.n - off)
        blocks.append((off, cnt))
        off += cnt
    skip = {"-j", "--jobs", "-n", "--seed0", "--index0", "--json"}
    keep, it = [], iter(sys.argv[1:])
    for tok in it:
        if tok in skip:
            next(it, None)
            continue
        if tok.split("=")[0] in skip:
            continue
        keep.append(tok)
    procs = []
    t0 = time.time()
    for off, cnt in blocks:
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "-n", str(cnt),
               "--seed0", str(a.seed0 + off), "--index0", str(off), "--json"] + keep
        # one thread per chunk: otherwise 12 processes each allot 28 cores
        # to themselves and fight over them, while the GPU does all the computing anyway
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, env=env,
                                      bufsize=1))
    # Worker output is read as a STREAM, not by `communicate()` at the end: otherwise
    # progress exists but becomes visible only after completion -- i.e.
    # there is none. The measurement that caused this rewrite: a head-to-head of 192 games
    # ran 45 minutes, and all that time the only sign of life was the growing
    # CPU time in `ps`.
    done = [0]
    lines = [[] for _ in procs]
    lock = threading.Lock()

    def pump(i, pr):
        for ln in pr.stdout:
            if ln.startswith("__PROG__"):
                with lock:
                    done[0] += 1
                    k = done[0]
                if k == 1 or k % max(1, a.n // 20) == 0 or k == a.n:
                    el = time.time() - t0
                    rate = k / max(1e-9, el)
                    eta = (a.n - k) / max(1e-9, rate)
                    print(f"  {k}/{a.n} games, {el/60:.1f} min, "
                          f"{rate*60:.1f} games/min, ~{eta/60:.0f} min left",
                          file=sys.stderr, flush=True)
            else:
                lines[i].append(ln)
        pr.stdout.close()

    th = [threading.Thread(target=pump, args=(i, pr), daemon=True)
          for i, pr in enumerate(procs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    res = []
    for i, pr in enumerate(procs):
        pr.wait()
        line = next((l for l in lines[i] if l.startswith("__RES__")), None)
        if line is None:
            raise SystemExit(f"chunk crashed (rc={pr.returncode}), no output")
        d = json.loads(line[len("__RES__"):])
        res += [tuple(g) for g in d["games"]]
        od = d.get("orders")
        if od:
            acc = _ostats()
            for kk in ("turns", "over", "want", "cut"):
                acc[kk] += od[kk]
            for kk, vv in od["kinds"].items():
                acc["kinds"][kk] = acc["kinds"].get(kk, 0) + vv
            for kk, vv in (od.get("clip") or {}).items():
                c = acc["clip"].setdefault(kk, [0, 0, 0])
                c[0] += vv[0]; c[1] += vv[1]; c[2] += vv[2]
            for kk, vv in (od.get("env") or {}).items():
                acc["env"][kk] = acc["env"].get(kk, 0) + vv
    el = time.time() - t0
    head = ""
    ost = _NET.get("order_stats")
    if ost and ost["turns"]:
        print(f"orders: on average {ost['want']/ost['turns']:.1f} per turn, "
              f"overflow the limit of 10 in {100*ost['over']/ost['turns']:.1f}% of turns, "
              f"cut {ost['cut']:,} orders "
              f"({100*ost['cut']/max(1,ost['want']):.1f}% of written)")
        if ost["kinds"]:
            print("  what is cut: " + "  ".join(
                f"{k} {v:,}" for k, v in sorted(ost["kinds"].items(),
                                                key=lambda kv: -kv[1])))
        _report_clip(ost)
    head += (f"\nscore: {len(blocks)} processes at a quota of {cap} cores, "
             f"{el:.0f}s, {el/len(res):.1f}s per game")
    if a.dump:
        json.dump({"opp": a.opp, "seed0": a.seed0,
                   "games": [list(g) for g in res]}, open(a.dump, "w"))
        print(f"games merged into {a.dump}")
    _report(np.array(res), a.opp_ckpt or a.opp, head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--opp", default="starter",
                    help="'starter' barely plays: it takes $3.5k and does not touch the market, "
                         "prices hold. A measurement against it is fine "
                         "as a 'did not break' check, but not as a comparison "
                         "of versions -- the gain there is inflated threefold (v18 over v11: +35%% "
                         "against starter, +11%% in self-play). A path to main.py "
                         "of a submission package (with weights.pt next to it) -- an opponent with its own "
                         "code: this is how pre-freeze nets are played")
    ap.add_argument("--opp-ckpt", default="",
                    help="the opponent is the same net with this checkpoint (self-play). "
                         "Loaded as a SEPARATE copy of the module so that the weights and "
                         "the flow tracker of the two sides do not mix")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default, and this is not thrift: Kaggle has no GPU, "
                         "and arithmetic on the card differs in the last digits -- "
                         "the same checkpoint and seeds gave $137,902 on cuda against "
                         "$149,676 on cpu. With parallel games cpu is also "
                         "three times faster: the card becomes a queue")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--sample-units", type=float,
                    default=DECODE_DEFAULTS["sample_units"],
                    help="sampling temperature of the target and op of units; 0 = argmax "
                         "(internal R4 note). The opponent from --opp-ckpt plays argmax")
    ap.add_argument("--no-sample", action="store_true",
                    help="gates by the 0.5 threshold (default since 25.09; the flag is kept "
                         "for old commands)")
    ap.add_argument("--sample-gates", action="store_true",
                    help="market gates by sampling, as the ladder played until 25.09")
    ap.add_argument("--sample-qty", type=float,
                    default=DECODE_DEFAULTS["sample_qty"],
                    help="sampling temperature of quantities from the CORN chain; 0 = "
                         "median (BC, as before R2)")
    ap.add_argument("--ppo", action="store_true",
                    help="the whole PPO rollout mode (policy.PPO): units τ=0.5, "
                         "quantities by sampling, no habits. Overrides the three "
                         "flags above")
    ap.add_argument("--opp-gates", default="same", choices=("same", "sample", "threshold"),
                    help="market gates of the opponent from --opp-ckpt: like ours (default), "
                         "by sampling or by threshold 0.5 -- independently of --no-sample")
    ap.add_argument("--opp-ppo", action="store_true",
                    help="the opponent from --opp-ckpt is also in PPO rollout mode "
                         "(policy.PPO) -- a head-to-head of stochastic policies")
    ap.add_argument("--beta", type=float, default=DECODE_DEFAULTS["beta"],
                    help="how much log prior to add to the op and target logits "
                         "(N2). β = 1 at τ = 1 is the teacher's distribution, β = 0 is "
                         "raw logits, as played before 23.09")
    ap.add_argument("--gate-floor", type=float, default=DECODE_DEFAULTS["floor"],
                    help="lower bound on the probability of saturated gates (seed, "
                         "animal, hire, land): p' = (1−ε)p + ε/2 (N4)")
    ap.add_argument("--opp-beta", type=float, default=None,
                    help="β of the opponent from --opp-ckpt; by default the decoder "
                         "default. The family differs between the sides ONLY in the N2/N4 "
                         "measurement, in play it is one")
    ap.add_argument("--opp-gate-floor", type=float, default=None,
                    help="the same for the opponent's gate lower bound")
    ap.add_argument("--check-orders", type=int, default=1,
                    help="check every one of our turns against the engine on a copy of the state "
                         "(policy.engine_check): the underexecuted goes into the "
                         "cut summary")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="games in parallel. The measurement is bound by LATENCY, not by "
                         "computation: the net is called ~2160 times per game at batch 1, "
                         "and the A100 idles on this (18%% utilization, 0.6 GB of 80). "
                         "Processes share the card without loss, the count goes J times faster")
    ap.add_argument("--index0", type=int, default=0,
                    help="global number of the first game: its parity determines "
                         "which side we play. Set by the parent with -j, "
                         "otherwise every chunk would start from the same side")
    ap.add_argument("--json", action="store_true", help="emit the result as one JSON line")
    ap.add_argument("--land-rate", action="store_true",
                    help="land purchase by accumulated frequency instead of the 0.5 threshold")
    ap.add_argument("--dedup-targets", action="store_true",
                    help="unit targets without repeats (greedy matching)")
    ap.add_argument("--gate-rate", action="store_true",
                    help="market gates by accumulated frequency instead of the 0.5 threshold")
    ap.add_argument("--gate-rate-heads", default="",
                    help="gate accumulator heads separated by commas (sell,buy,...); empty -- all")
    ap.add_argument("--mkt-order", default="",
                    help="market block order of own net, e.g. sell,hire,animal,seed,buy,land")
    ap.add_argument("--night-room", action="store_true",
                    help="last turn of the day: sell from the shed what would keep hands from dumping at night")
    ap.add_argument("--night-room-mode", default=None, choices=["stock", "value"],
                    help="what to sell for the night dump: stock (largest remainder) or value (by price)")
    ap.add_argument("--night-room-hour", type=int, default=None,
                    help="from which hour --night-room acts (default 23 -- only the last turn of the day)")
    ap.add_argument("--experts", default="",
                    help="extra experts of own side separated by commas (weights); indices 1.. in --route")
    ap.add_argument("--router", default="", help="router file (router_train.py) for --route learned[:δ]")
    ap.add_argument("--route", default="",
                    help="expert router: 'd0-d1:group=i;...' or 'conf' (see DECODE_DEFAULTS)")
    ap.add_argument("--build-bias", type=float, default=0.0,
                    help="bias added to BUILD_COOP/BUILD_PASTURE logits of own net")
    ap.add_argument("--ablate", default="",
                    help="switch off corrector inputs of own net, separated by commas (co_intent,co_mkt,...)")
    ap.add_argument("--end-haul", action="store_true",
                    help="at season end lead loaded units to the shed and drop")
    ap.add_argument("--end-sell", action="store_true",
                    help="last turn of the season: sell the whole shed, buy nothing")
    ap.add_argument("--feed-guard", action="store_true",
                    help="buy wheat for a hungry herd (own net only)")
    ap.add_argument("--feed-guard-risk", action="store_true",
                    help="feed rule only on escape risk (until --feed-guard-hour)")
    ap.add_argument("--feed-guard-hour", type=int, default=None,
                    help="hour from which the feed rule in risk mode feeds all")
    ap.add_argument("--dump", default="",
                    help="dump the per-game result to a file. Comparing configurations "
                         "by medians at a 3x spread and n=48 is almost useless; "
                         "the seeds are the same across runs, so the comparison should be made "
                         "PAIRED by seed, and for that the games themselves are needed")
    a = ap.parse_args()
    if a.jobs > 1:
        return _fan_out(a)
    _load(a.ckpt, a.device)
    # Gates: the default is from `DECODE_DEFAULTS` (threshold since 25.09), flags only
    # override. Previously `not a.no_sample` stood here, i.e. the command
    # line's own default overrode the shared one.
    if a.sample_gates and a.no_sample:
        raise SystemExit("--sample-gates and --no-sample together")
    _NET["sample"] = (True if a.sample_gates else
                      False if a.no_sample else bool(DECODE_DEFAULTS["sample"]))
    _NET["sample_units"] = a.sample_units
    _NET["sample_qty"] = a.sample_qty
    if a.ppo:
        _NET.update(POL.PPO)
    # The family is set AFTER the mode: `--ppo` changes temperatures, not it.
    _NET["beta"] = a.beta
    _NET["floor"] = a.gate_floor
    if a.beta and not _NET.get("prior"):
        # The check is HERE, not on the first turn: from inside `agent` the exception
        # is caught by `kaggle_environments`, which removes the agent from the game and gives it
        # $3,000 -- i.e. the measurement looks like "plays badly".
        raise SystemExit(f"{a.ckpt}: the checkpoint has no prior, and β = {a.beta}. "
                         "It can be played only with --beta 0 (this is a DIFFERENT "
                         "distribution than the defaults) or by recomputing "
                         "prior over its corpus (bc_train.logit_prior)")
    if (a.beta and _NET.get("prior_tau") is not None
            and abs(a.beta - _NET["prior_tau"]) > 1e-9):
        # The loss trained softmax(f + τ·log prior); play takes f + β·log prior. At
        # β ≠ τ the net plays not the distribution it was trained on (N2), and
        # there is no loud refusal anywhere else (internal policy-parity note, 24.09).
        print(f"⚠ {a.ckpt}: trained with --tau {_NET['prior_tau']}, plays β = {a.beta} -- "
              "game family does not match the trained one", file=sys.stderr, flush=True)
    _NET["check_orders"] = bool(a.check_orders)
    if a.feed_guard:                 # the flag only turns on; from the weights -- see _load
        _NET["feed_guard"] = True
    if a.end_sell:
        _NET["end_sell"] = True
    if a.end_haul:
        _NET["end_haul"] = True
    if a.ablate:
        # Probe 29.09: switch off corrector inputs (co_intent, co_mkt, ...) -- contribution to money.
        class _Z(torch.nn.Module):
            def __init__(s_, mod, tup=False):
                super().__init__(); s_.mod, s_.tup = mod, tup
            def forward(s_, *x, **k):
                r = s_.mod(*x, **k)
                return (r[0] * 0, r[1]) if s_.tup else r * 0
        _m = _NET["m"]
        for mn in a.ablate.split(","):
            setattr(_m, mn, _Z(getattr(_m, mn), tup=(mn == "co_xt")))
    if a.build_bias:
        _NET["build_bias"] = a.build_bias
    if a.night_room:
        _NET["night_room"] = True
    if a.night_room_hour is not None:
        _NET["night_room_hour"] = a.night_room_hour
    if a.night_room_mode:
        _NET["night_room_mode"] = a.night_room_mode
    if a.experts:
        _NET["experts"] = [_load_expert(x, a.device) for x in a.experts.split(",")]
        _NET["route"] = a.route
        if a.router:
            _NET["router_src"] = a.router
    if a.feed_guard_risk:
        _NET["feed_guard_risk"] = True
    if a.feed_guard_hour is not None:
        _NET["feed_guard_hour"] = a.feed_guard_hour
    if a.mkt_order:
        _NET["mkt_order"] = tuple(a.mkt_order.split(","))
    if a.gate_rate:
        _NET["gate_rate"] = True
    if a.gate_rate_heads:
        _NET["gate_rate_heads"] = a.gate_rate_heads
    if a.dedup_targets:
        _NET["dedup_targets"] = True
    if a.land_rate:
        _NET["land_rate"] = True
    from kaggle_environments import make
    opp = a.opp
    if a.opp.endswith(".py") and os.path.exists(
            os.path.join(os.path.dirname(a.opp), "weights.pt")):
        opp = _package_agent(a.opp)       # submission package -- with its own code
    if a.opp_ckpt:
        # A separate copy of the module: `_NET` and `_FLOW` in play.py are global, and
        # an ordinary import would return the same object -- the sides would share weights and
        # accumulated flow. Exactly how the packaged submission does it too.
        import importlib.util
        spec = importlib.util.spec_from_file_location("play_opp",
                                                      os.path.abspath(__file__))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._load(a.opp_ckpt, a.device)
        mod._NET.update({"sample": _NET["sample"] if a.opp_gates == "same"
                         else a.opp_gates == "sample"})
        if a.opp_ppo:
            # the opponent is also in rollout mode: a comparison of stochastic
            # policies that PPO actually trains (internal RL diagnosis note, item 4)
            mod._NET.update(POL.PPO)
        if a.opp_beta is not None:
            mod._NET["beta"] = a.opp_beta
        if a.opp_gate_floor is not None:
            mod._NET["floor"] = a.opp_gate_floor
        if mod._NET.get("beta", DECODE_DEFAULTS["beta"]) and not mod._NET.get("prior"):
            raise SystemExit(f"{a.opp_ckpt}: the opponent has no prior, and β > 0 -- "
                             "see --opp-beta 0")
        opp = mod.agent
        _NET["opp_mod"] = mod
    res = []
    for i in range(a.n):
        g = a.index0 + i                  # global game number
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": a.seed0 + i},
                   debug=False)
        _reset_flow()
        # The generator is fixed for EACH game. Without this market gates
        # are sampled differently, games diverge from the first turn, and one and the
        # same checkpoint on the same seeds gives a spread in which almost
        # all comparisons drown. Measurement: two runs of v18 gave medians $150,136 and
        # $138,172, paired 16 wins of 48 at p=0.029 -- a "significant degradation"
        # of the model relative to itself. Median |difference| by seed $18,498.
        # The environment seed and the policy seed now coincide, and the paired difference measures
        # ONLY the model.
        torch.manual_seed(a.seed0 + i)
        t0 = time.time()
        if _NET.get("opp_mod") is not None:
            _NET["opp_mod"]._reset_flow()
        env.run([agent, opp] if g % 2 == 0 else [opp, agent])
        last = env.steps[-1]
        me = 0 if g % 2 == 0 else 1
        r_me, r_op = last[me].reward or 0.0, last[1 - me].reward or 0.0
        # Status and time reserve are NOT decoration. An agent removed from the game gives
        # reward 0 and looks like an agent with a bad strategy: this is how a refining
        # decoder once already "lost" a measurement, while actually eating up the
        # 60-second reserve on actTimeout overruns under load.
        # The status of the LAST step is useless: the interpreter unconditionally sets
        # DONE for everyone and reward = the frozen bank (env_ref.py:958-963). An agent
        # that crashed on the fifth turn gives DONE, its money and a reserve of 60 of 60 --
        # indistinguishable from a badly playing one. Verified by experiment: statuses per
        # game ERROR 714, ACTIVE 5, DONE 1, on output DONE and $3,000.
        # So we look at ALL steps: the live one is the one that was ACTIVE to the end.
        bad = next((s[me]["status"] for s in env.steps[:-1]
                    if s[me]["status"] not in ("ACTIVE", "INACTIVE")), None)
        over = last[me]["observation"].get("remainingOverageTime", 60.0)
        res.append((r_me, r_op, bad is not None, float(over)))
        st_g = bad or "DONE"
        if a.json:
            # A line per game, unbuffered. Without it the worker is silent until the end
            # of the run, and in an hour-long head-to-head there is not a single sign of whether
            # work is going or hung (project rule on output buffering).
            print(f"__PROG__ {g} {r_me:.0f} {r_op:.0f} {time.time()-t0:.1f}",
                  flush=True)
        if not a.json:
            bad = f"  STATUS {st_g}" if st_g != "DONE" else ""
            print(f"  game {g+1}: we ${r_me:,.0f}  opponent ${r_op:,.0f}  "
                  f"({time.time()-t0:.0f}s, reserve {over:.0f}s){bad}", flush=True)
    if a.json:
        print("__RES__" + json.dumps({"games": res,
                                      "orders": _NET.get("order_stats")}))
        return
    head = ""
    _report_clip(_NET.get("order_stats"))
    _report(np.array(res), a.opp_ckpt or a.opp, head)


if __name__ == "__main__":
    main()
