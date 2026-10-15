"""The whole PPO loop: collection -> buffer -> update -> checkpoint -> league.

Launch (smoke, local):
  python -u experiments/rl/train.py --ckpt <bc.pt> --name smoke \
      --updates 1 --set rollout.episodes=2 rollout.max_turns=24

Full-scale launch: all the speed-up is one set, `--preset fast` (inference
server, environment port, graphs; `config.PRESETS`), the algorithm goes via `--set`:
  python3 -u experiments/rl/train.py --ckpt <bc.pt> --name rl1 \
      --critic tower --preset fast --set rollout.episodes=84 \
      rollout.eval_episodes=4 ppo.value_epochs=4
`--preset overlap` - the same plus collection in parallel with the update (data lags
by one update, see `RunCfg.overlap`).

WARNING: `rollout.max_turns` cuts only the RECORDING, not the game: it does not save time
(measured 23.09), and the truncated stream gets the terminal reward for the full
game. It is for checking the mechanics, not for training or timing.

WARNING: real training has never been launched from here. The critic is wired in
(`--critic tower`, the R1 tower); what remains is deciding H1/H2/H4 on the
distribution family.

The critic is chosen by a flag: `zero` - stub (GAE degenerates into Monte Carlo without a
baseline, good only for checking the mechanics), `tower` - the R1 value
tower. The tower must arrive TRAINED: either in the same checkpoint as the policy
(as produced by `bc_train.py --value-only`), or in a separate `--critic-ckpt`.

Checkpoints. Two files per update, as in the earlier project:
  `actor-NNNNNN.pt` - weights only, immutable; the league plays with it;
  `state-NNNNNN.pt` - policy, optimizer, generators, league, counters.
The write is atomic (tmp + fsync + replace): a run killed mid-write would otherwise
leave a truncated checkpoint that loads silently.
WARNING: we write to `experiments/rl/runs`, NOT to `experiments/net/runs`: on 23.09 that folder
in the main copy became a symlink to itself, and all checkpoints under that path
were lost.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import copy
import math
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import policy as POL  # noqa: E402

import collect as COL  # noqa: E402
import critic as CRIT  # noqa: E402
import diagnostics as DIAG  # noqa: E402
import league as LG  # noqa: E402
import ppo as PPO  # noqa: E402
from buffer import Batch  # noqa: E402
from config import PRESETS, RunCfg, override  # noqa: E402


# ---------------------------------------------------------------- writing

def atomic_save(obj, path: str) -> None:
    """tmp + fsync + replace: a truncated checkpoint loads silently, which is worse than a failure."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class JsonlLog:
    """Metrics log, one line per update.

    jsonl rather than TensorBoard: the repo already has `metrics.jsonl` from BC training, and
    runs are compared by code, not by eye. TensorBoard is attached
    on top - the file is read line by line.
    """

    def __init__(self, path: str):
        self.path = path

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            f.flush()


# ---------------------------------------------------------------- run

class Run:
    def __init__(self, cfg: RunCfg):
        self.cfg = cfg
        self.dir = os.path.join(ROOT, cfg.out, cfg.name) \
            if not os.path.isabs(cfg.out) else os.path.join(cfg.out, cfg.name)
        os.makedirs(self.dir, exist_ok=True)
        # WARNING: the TRAINER's device, not the rollout's: batch 1 and batch 64 want different
        # hardware (R6). Empty = as before, one for everything.
        dev = cfg.train_device or cfg.rollout.device
        if cfg.train_device and cfg.train_device != cfg.rollout.device \
                and int(cfg.rollout.jobs or 1) <= 1:
            raise SystemExit(
                f"train_device = {cfg.train_device} with rollout.device = "
                f"{cfg.rollout.device} and one collection process: the rollout would go "
                "through the trainer's network, i.e. batch 1 on the GPU. Set "
                "`--set rollout.jobs=<N>`")
        if not cfg.ckpt:
            raise SystemExit("--ckpt is required: the BC checkpoint is both start and anchor (R7)")
        adv_mode = getattr(cfg.ppo, "advantage", "gae")
        if adv_mode not in ("gae", "rloo", "grpo_t"):
            raise SystemExit(f"ppo.advantage = {adv_mode!r}: gae | rloo | grpo_t")
        if adv_mode != "gae" and int(cfg.rollout.group_size) < 2:
            raise SystemExit(f"ppo.advantage = {adv_mode} requires "
                             "`rollout.group_size >= 2`: without a group there is "
                             "nothing to take a baseline from")

        # H10: start from `ck["model"]` (this is the EMA) and a FRESH optimizer -
        # `ck["opt"]` belongs to the raw weights, not the EMA, and picking it up
        # would drive the moments onto different weights.
        self.net = COL.load_net(cfg.ckpt, dev)
        # WARNING: the teacher's log-frequencies are part of the POLICY, not of the start file.
        # `policy.family` adds beta*prior to the op and target logits, and at beta > 0
        # (the default) a network without them plays NOT the distribution it
        # was trained on. Training does not change them, so they are read once here
        # and travel with every actor checkpoint: a league opponent is a file
        # `actor-NNNNNN.pt` and nothing more, and everything needed must be inside it.
        #
        # 23.09: they were NOT there. The opponent was loaded without prior,
        # `policy.family` refused to play, the competition harness swallowed the exception and gave
        # the agent ERROR - in the log that is $3 000 of $3 000 over 719 turns. Not only
        # `win_rate` was spoiled: such games were labelled as giant wins, and
        # the critic trained on knowingly false labels.
        self.prior = COL.load_prior(cfg.ckpt, "cpu")
        if self.prior is None:
            print(f"WARNING: {cfg.ckpt} has no `prior`: actor checkpoints will go to the league "
                  "without it, and at beta > 0 the opponent will refuse to play",
                  file=sys.stderr, flush=True)
        # The anchor is THE SAME checkpoint, frozen and lying alongside. If the leash
        # pulled toward the "raw" network without temperature, it would pull toward a flatter
        # distribution than the starting one (H9).
        self.anchor = COL.load_net(cfg.ckpt, dev)
        if cfg.critic in ("tower", "tower_scalar"):
            # The tower must arrive TRAINED. Otherwise `load_bc` would bring it up as a
            # warm start (the HL head being random), the estimate would be
            # noise in the reward scale, and GAE would silently subtract this noise.
            if cfg.critic_ckpt:
                n = COL.load_tower(self.net, cfg.critic_ckpt)
                print(f"value tower from {cfg.critic_ckpt}: {n} tensors",
                      flush=True)
            elif not COL.has_tower(cfg.ckpt):
                raise SystemExit(
                    f"in {cfg.ckpt} there is no value tower (`vtower.*`): this is a "
                    "pre-R1 checkpoint. Warm it up with `bc_train.py --value-only` "
                    "or pass --critic-ckpt")
        # alpha is deliberately NOT passed to the critic from the config: it takes it from
        # `value_head`, and `Trainer` checks the reward alpha against it. If we passed it
        # here, the check would compare a number with itself. Sigma, however, is taken from
        # the CHECKPOINT's `cfg`: the HL head was trained on it, and the default could have been
        # overridden by the `bc_train.py --value-sigma` flag.
        kw = {}
        if cfg.critic in ("tower", "tower_scalar"):
            sg, gm = COL.ckpt_value_scale(cfg.critic_ckpt or cfg.ckpt)
            if sg is None and cfg.critic_ckpt:
                # `--critic-ckpt` is a run STATE file (a tower from the loop): its
                # `cfg` holds run settings, not BC ones, and there is no sigma there. The loop's
                # tower trains on the sigma of its starting BC checkpoint - and it is here too.
                sg, gm = COL.ckpt_value_scale(cfg.ckpt)
            if sg is not None:
                kw["sigma"] = sg
            print(f"critic: tower, trained with sigma = {sg}, gamma = {gm}; "
                  f"reward alpha = {cfg.reward.cash_alpha}, sigma = {cfg.reward.cash_sigma}",
                  flush=True)
        self.critic = CRIT.make(cfg.critic, net=self.net, **kw)
        self.trainer = PPO.Trainer(self.net, self.anchor, self.critic, cfg.ppo,
                                   COL.decode_mode(COL.ppo_decode(cfg)), device=dev,
                                   ppg=cfg.ppg, reward=cfg.reward,
                                   prior=COL.load_prior(cfg.ckpt, dev))
        self.league = LG.League(cfg.league, ROOT, anchor_path=cfg.ckpt)
        self.log = JsonlLog(os.path.join(self.dir, "metrics.jsonl"))
        # TensorBoard is a MIRROR of `metrics.jsonl`, not a second source: the same
        # fields of the same line, the step is the update number. View locally by
        # syncing `*/tb` from the training machine and running TensorBoard on a laptop.
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(self.dir, "tb"))
        except Exception as e:     # noqa: BLE001 - the log is not critical for the loop
            print(f"WARNING: TensorBoard unavailable ({e}): only jsonl is written",
                  file=sys.stderr, flush=True)
        self.env_steps = 0
        self.rng = random.Random(cfg.seed)
        self.update = 0
        self.stopping = False
        # Parallel collection (`RunCfg.overlap`): the rollout of the next update
        # is in flight. Dict: thread, plan, start seed, result/exception.
        self._pending: Optional[Dict[str, Any]] = None
        # (actor path, update number) not yet through acceptance
        # (`rollout.eval_async`)
        self._last_actor = None
        if cfg.overlap:
            bad = []
            if not cfg.rollout.serve:
                bad.append("`rollout.serve = true` (per-process serving holds the "
                           "CPU, which the update needs too)")
            if cfg.ppo.behavior_rho <= 0:
                bad.append("`ppo.behavior_rho > 0` (without an importance weight the rollout "
                           "of old weights would enter the update as on-policy)")
            if int(cfg.rollout.logp_every) != 1:
                bad.append("`rollout.logp_every = 1` (the behaviour `log pi` "
                           "is needed on every turn - it is the weight's denominator)")
            if bad:
                raise SystemExit("`overlap = true` requires: " + "; ".join(bad))
        atomic_json(cfg.to_dict(), os.path.join(self.dir, "config.json"))
        if self.league.missing:
            print("WARNING: frozen packages not found (no raw-content/): "
                  + ", ".join(self.league.missing), file=sys.stderr, flush=True)

    # ---------------- state ----------------

    def save(self) -> str:
        actor = self.save_actor()
        self.save_state()
        return actor

    def save_actor(self) -> str:
        actor = os.path.join(self.dir, f"actor-{self.update:06d}.pt")
        # `prior` goes together with the weights, always. The actor file must be
        # self-contained: its only consumer is the league, and it reads
        # only it.
        atomic_save({"model": self.net.state_dict(), "prior": self.prior}, actor)
        return actor

    def save_state(self) -> None:
        """WARNING: the state is written LAST in an update: after acceptance, league
        admission and counter shifts. Earlier it was written before them, and `--resume`
        after a sudden death lost the accepted snapshot, replayed the acceptance seed
        block and undercounted `env_steps` (review 23.09, finding 4)."""
        atomic_save({
            "model": self.net.state_dict(),
            "prior": self.prior,
            "opt": self.trainer.opt.state_dict(),
            "update": self.update,
            "env_steps": self.env_steps,
            # actor not yet through acceptance (`rollout.eval_async`)
            "last_actor": self._last_actor,
            # critic head outside the network (`critic.ScalarTowerCritic`)
            "critic_head": (self.critic.state_dict()
                            if hasattr(self.critic, "head_lin") else None),
            "kl_coef": self.trainer.kl_coef,
            "league": self.league.state(),
            "cfg": self.cfg.to_dict(),
            # All three generators: the Python one drives the league, torch drives turn
            # sampling, numpy drives permutations. Without any of them a resume
            # gives a different run, and the difference of two runs stops measuring
            # an edit.
            "rng": {"python": self.rng.getstate(),
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state()},
            # WARNING: the seed counters go under a separate key, not "in cfg". The `cfg` in
            # the checkpoint is there for history and is NOT read on resume
            # (otherwise a resume would silently cancel the command-line `--set`),
            # while `seed0` moves with every update. Without it a resumed
            # run would replay the same seed block from scratch - i.e. would train
            # on games it has already seen, and the "growth" would partly be
            # repetition.
            # With parallel collection the in-flight rollout is not yet trained on: on
            # resume its seed block is played again (with the new weights).
            "seeds": {"seed0": (self._pending["seed0"] if self._pending
                                else self.cfg.rollout.seed0),
                      "eval_seed0": self.cfg.rollout.eval_seed0},
        }, os.path.join(self.dir, f"state-{self.update:06d}.pt"))

    def load(self, path: str) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(st["model"])
        # A resume takes `prior` from the state if it is there. It does not silently overwrite its
        # own value (from `cfg.ckpt`): they must
        # match - the policy does not change them - and a mismatch means
        # the run is being resumed from a DIFFERENT starting checkpoint.
        pr = st.get("prior")
        if pr is not None and self.prior is not None:
            bad = [k for k in set(pr) | set(self.prior)
                   if k not in pr or k not in self.prior
                   or not torch.equal(pr[k].cpu(), self.prior[k].cpu())]
            if bad:
                raise SystemExit(
                    f"`prior` in {path} does not match the one in {self.cfg.ckpt} "
                    f"(differing: {sorted(bad)}). Resuming from a different starting "
                    "checkpoint would change the distribution under the same weights")
        elif pr is not None:
            self.prior = pr
        opt = self.trainer.opt
        if hasattr(opt, "step_value"):
            opt.load_state_dict(st["opt"], legacy_order=self.trainer.params)
        else:
            opt.load_state_dict(st["opt"])
        # WARNING: `load_state_dict` also returns the group HYPERPARAMETERS (lr, betas,
        # weight_decay) - those the state was written with. A resume with a
        # different `--set ppo.lr` (critic warm-up -> PPO) would silently run on the old
        # lr. We take the moments from the state and the numbers from the command line.
        pc = self.cfg.ppo
        # The critic lr is its own, if set (`ppo.value_lr`), and a resume
        # must take it, not the shared one: otherwise the flag would silently have no effect
        # exactly when a run moves to new settings.
        vlr = float(getattr(pc, "value_lr", -1.0))
        vlr = pc.lr if vlr < 0 else vlr
        opt = self.trainer.opt
        pairs = ([(g, pc.lr) for g in opt.pol.param_groups]
                 + [(g, vlr) for g in (opt.val.param_groups if opt.val else [])]
                 if hasattr(opt, "pol") else [(g, pc.lr) for g in opt.param_groups])
        for g, lr in pairs:
            if (g["lr"], tuple(g["betas"]), g["weight_decay"]) != \
                    (lr, tuple(pc.betas), pc.weight_decay):
                print(f"resume: lr {g['lr']} -> {lr}, betas "
                      f"{tuple(g['betas'])} -> {tuple(pc.betas)}, weight_decay "
                      f"{g['weight_decay']} -> {pc.weight_decay} (from config)",
                      flush=True)
            g["lr"], g["betas"], g["weight_decay"] = lr, tuple(pc.betas), pc.weight_decay
        self.update = int(st["update"])
        self.env_steps = int(st.get("env_steps", 0))
        if hasattr(self.critic, "head_lin"):
            if st.get("critic_head") is not None:
                self.critic.load_state_dict(st["critic_head"])
            else:
                print("WARNING: the state has no scalar-critic head - it "
                      "starts from zero", file=sys.stderr, flush=True)
        la = st.get("last_actor")
        self._last_actor = tuple(la) if la and os.path.exists(la[0]) else None
        # The leash coefficient from the state only if it is ADAPTIVE
        # (`kl_target > 0`): then it is run state. With a constant one it is
        # a setting, and `--set ppo.kl_coef` must win on resume
        # (the same class of error as with lr above; review 23.09, finding 5).
        if self.cfg.ppo.kl_target > 0:
            self.trainer.kl_coef = float(st.get("kl_coef", self.cfg.ppo.kl_coef))
        else:
            old_k = float(st.get("kl_coef", self.cfg.ppo.kl_coef))
            if old_k != self.cfg.ppo.kl_coef:
                print(f"resume: kl_coef {old_k} -> {self.cfg.ppo.kl_coef} "
                      "(constant, from config)", flush=True)
            self.trainer.kl_coef = self.cfg.ppo.kl_coef
        self.league.load_state(st.get("league") or {})
        r = st.get("rng") or {}
        if r:
            self.rng.setstate(r["python"])
            torch.set_rng_state(r["torch"])
            np.random.set_state(r["numpy"])
        s = st.get("seeds") or {}
        if s:
            self.cfg.rollout.seed0 = int(s["seed0"])
            self.cfg.rollout.eval_seed0 = int(s["eval_seed0"])
        else:
            print("WARNING: the checkpoint has no seed counters: a resume will replay "
                  f"the block starting at {self.cfg.rollout.seed0}", file=sys.stderr, flush=True)

    # ---------------- iteration ----------------

    # ---------------- parallel collection ----------------

    def _snapshot(self):
        """Snapshot of the network on CPU for the collection or acceptance thread.

        WARNING: before the copy, `vtower.last_pooled` is removed (the pooling vector of
        the last pass for `critic.ScalarTowerCritic`): it is not an autograd graph
        leaf, and `copy.deepcopy` fails on it (rl3, 24.09, failed on the second
        collection). The attribute lives only inside a pass - the next tower call
        sets it anew, and CUDA graphs hold their own tensors."""
        vt = getattr(self.net, "vtower", None)
        if vt is not None and getattr(vt, "last_pooled", None) is not None:
            vt.last_pooled = None
        return copy.deepcopy(self.net).to("cpu").eval()

    def _launch(self) -> None:
        """Start collecting the next rollout with the CURRENT weights in a thread.

        The thread only waits for the collection processes (`infer_client.collect_served`: server
        and workers are separate processes), so the GIL hardly interferes with the update.
        The thread gets a CPU SNAPSHOT of the network: the live GPU network is
        training at that time, and `collect_served` (shape probe, `torch.save` of weights to the server)
        would read it in the middle of an optimizer step. Settings too are a copy: the seed
        is shifted here as well, in the main thread.
        """
        cfg = self.cfg
        plan = self.league.plan(cfg.rollout.episodes, self.rng,
                                    group=cfg.rollout.group_size)
        snap = self._snapshot()
        rc = copy.deepcopy(cfg)
        job: Dict[str, Any] = {"seed0": cfg.rollout.seed0, "update": self.update,
                               "t0": time.time()}
        cfg.rollout.seed0 += cfg.rollout.episodes

        def body():
            try:
                job["eps"] = COL.collect(snap, plan, rc, critic=None)
            except BaseException as e:          # noqa: BLE001 - handed over in join
                job["err"] = e
        job["thread"] = threading.Thread(target=body, name="rollout", daemon=True)
        job["thread"].start()
        self._pending = job

    def _join(self):
        job, self._pending = self._pending, None
        job["thread"].join()
        if "err" in job:
            raise RuntimeError("collection in the thread failed") from job["err"]
        return job

    def _launch_eval(self) -> Dict[str, Any]:
        """Acceptance panel for a snapshot of the current weights - in a thread (`rollout.eval_async`).

        The same trick as in `_launch`: a CPU snapshot of the network and a copy of the settings, the
        acceptance seed is shifted here as well, in the main thread."""
        cfg = self.cfg
        actor, upd = self._last_actor
        eplan = self.league.plan(cfg.rollout.eval_episodes, self.rng)
        snap = self._snapshot()
        rc = copy.deepcopy(cfg)
        if not cfg.rollout.eval_seed_fixed:
            cfg.rollout.eval_seed0 += cfg.rollout.eval_episodes
        job: Dict[str, Any] = {"actor": actor, "update": upd}

        def body():
            try:
                job["eps"] = COL.collect_eval(snap, eplan, rc)
            except BaseException as e:          # noqa: BLE001 - handed over in join
                job["err"] = e
        job["thread"] = threading.Thread(target=body, name="eval", daemon=True)
        job["thread"].start()
        return job

    @staticmethod
    def _join_eval(job):
        job["thread"].join()
        if "err" in job:
            raise RuntimeError("acceptance in the thread failed") from job["err"]
        return job

    def step(self) -> Dict[str, Any]:
        cfg = self.cfg
        t0 = time.time()
        lag = 0
        # `ppo.prepass_value`: V for GAE comes from the trainer's pre-pass; there is no separate
        # critic pass over the rollout (`collect.score`) - it ran the same
        # trunk over the same turns with the same weights (R6, second round)
        pv = bool(getattr(cfg.ppo, "prepass_value", False))
        if cfg.overlap:
            if self._pending is None:
                self._launch()
            job = self._join()
            eps = job["eps"]
            lag = self.update - job["update"]
            # the next rollout - with the weights from BEFORE this update, and right away
            self._launch()
            # V(s) - by the current critic, in the main thread: it goes into GAE too
            if not pv:
                COL.score(self.critic, eps, str(next(self.net.parameters()).device))
        else:
            plan = self.league.plan(cfg.rollout.episodes, self.rng,
                                    group=cfg.rollout.group_size)
            eps = COL.collect(self.net, plan, cfg,
                              critic=None if pv else self.critic)
            # The shift AFTER collection: it used to come before, and the first update played
            # not on `seed0` but on `seed0 + episodes` - a run named "from
            # 9000" started at 9008. Harmless, but the measurement protocol
            # stands on the seed block being known from the run's name.
            cfg.rollout.seed0 += cfg.rollout.episodes    # seeds do not repeat
        # With `overlap` this is the WAIT for collection, not its duration: zero means
        # the rollout finished within the time of the previous update.
        t_roll = time.time() - t0
        stats = COL.rollout_stats(eps)
        # the summary reads `t.value`: with `prepass_value` they do not exist yet - after the update
        diag = None if pv else DIAG.summarize(eps)

        # Acceptance in parallel with the update (`rollout.eval_async`): a snapshot of the
        # CURRENT weights (= the previous update's actor) plays its panel while
        # the GPU trains this update. League admission - by the results, for that actor.
        ev_job = None
        if cfg.rollout.eval_async and cfg.rollout.eval_episodes > 0 \
                and self._last_actor:
            ev_job = self._launch_eval()

        t1 = time.time()
        batch = Batch(eps, cfg.ppo, horizon=cfg.rollout.gae_horizon,
                      defer_values=pv)
        m = self.trainer.update(batch)
        t_upd = time.time() - t1
        if diag is None:
            diag = DIAG.summarize(eps)

        # Difficulty EMA - over the games actually played, before promotion:
        # otherwise the new snapshot would immediately get the old one's weight.
        for key in {e.opponent for e in eps if not e.dead}:
            sub = [e for e in eps if e.opponent == key and not e.dead]
            self.league.record(key, sum(e.outcome < 0 for e in sub) / len(sub))

        self.update += 1
        # over the BUFFER, not `ppo/samples`: that one counts the policy-epoch turns and is
        # zero during critic warm-up (`ppo.epochs=0`)
        self.env_steps += len(batch)
        saving = self.update % cfg.save_every == 0
        # actor - BEFORE acceptance (its path goes to `league.promote`), state -
        # after everything (`save_state`)
        actor = self.save_actor() if saving else ""

        # Acceptance panel: argmax on both sides, its own seed block, does not go into the buffer.
        # Only on it is the win rate comparable with a checkpoint opponent (H1/R4).
        ev: Dict[str, float] = {}
        t2 = time.time()
        promote_actor, promote_update = actor, self.update
        if cfg.rollout.eval_async and cfg.rollout.eval_episodes > 0:
            # acceptance results are for the PREVIOUS update's actor; the current actor
            # will be accepted (or not) in the next one
            promote_actor, promote_update = "", 0
            if ev_job is not None:
                job = self._join_eval(ev_job)
                ev = {f"eval/{k}": v for k, v in
                      COL.rollout_stats(job["eps"]).items()}
                ev["eval/of_update"] = job["update"]
                promote_actor, promote_update = job["actor"], job["update"]
            self._last_actor = (actor, self.update) if actor else None
        elif cfg.rollout.eval_episodes > 0:
            eplan = self.league.plan(cfg.rollout.eval_episodes, self.rng)
            eeps = COL.collect_eval(self.net, eplan, cfg)
            if not cfg.rollout.eval_seed_fixed:
                cfg.rollout.eval_seed0 += cfg.rollout.eval_episodes
            ev = {f"eval/{k}": v for k, v in COL.rollout_stats(eeps).items()}
        t_eval = time.time() - t2

        promoted = False
        wr = ev.get("eval/win_rate")
        honest = wr is not None
        if wr is None:
            wr = stats.get("win_rate")
        if promote_actor and wr is not None:
            promoted = self.league.promote(promote_actor, promote_update, wr,
                                           honest=honest)
        self.league.dump(os.path.join(self.dir, "league.json"))
        if saving:
            self.save_state()

        row = {"update": self.update, "time_rollout_s": t_roll,
               "rollout_lag": lag,
               "time_update_s": t_upd, "time_eval_s": t_eval,
               "promoted": promoted, "promote_honest": honest,
               "pool_history": len(self.league.history)}
        row.update(ev)
        row.update({f"ppo/{k}": v for k, v in m.items()})
        row.update({f"rollout/{k}": v for k, v in stats.items()})
        row.update(diag)
        row["env_steps"] = self.env_steps
        self.log.write(row)
        if self.tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) \
                        and math.isfinite(v):
                    self.tb.add_scalar(k, v, self.update)
            self.tb.flush()
        print(json.dumps({k: row[k] for k in (
            "update", "ppo/policy_loss", "ppo/value_loss", "ppo/entropy",
            "ppo/kl_anchor", "ppo/approx_kl", "ppo/clip_frac", "ppo/logp_drift",
            "ppo/early_stop", "ppo/explained_var", "ppo/explained_var_mc",
            "ppo/explained_var_mc_n", "ppo/max_abs_log_ratio",
            "ppo/grad_norm", "ppo/grad_clipped_frac", "ppo/grad_norm_value",
            "rollout/win_rate", "rollout/win_rate/vs_bc", "eval/win_rate",
            "rollout/money_median", "time_rollout_s", "time_update_s",
            "time_eval_s", "ppo/line_alpha", "ppo/line_tries",
            "ppo/heldout_gain", "ppo/heldout_gain_per_sqrt_kl", "ppo/half_cos", "ppo/pg_norm", "ppo/reg_norm", "ppo/cos_prev_step", "ppo/cos_to_anchor")
            if k in row}, ensure_ascii=False, default=float), flush=True)
        return row

    def run(self) -> None:
        def stop(_sig, _frm):
            # Stop at an update BOUNDARY, not instantly: an interrupted
            # update leaves weights that are in no checkpoint.
            self.stopping = True
            print("stop requested: finishing the update", flush=True)
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, stop)
        n = self.cfg.updates
        while not self.stopping and (n <= 0 or self.update < n):
            self.step()
        if self._pending is not None:
            # We wait for the in-flight rollout (collection processes would otherwise stay
            # orphans) but do not train on it: its block's seed goes into the
            # state, and a resume will play the block again.
            print("waiting for the in-flight rollout (it will not go into training)", flush=True)
            job = self._pending
            job["thread"].join()
            self.save()
            self._pending = None
            return
        self.save()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="BC checkpoint: both the policy start and the anchor (R7)")
    ap.add_argument("--name", default="rl0")
    ap.add_argument("--out", default="experiments/rl/runs")
    ap.add_argument("--updates", type=int, default=0, help="0 = until signal")
    ap.add_argument("--critic", default="zero", choices=CRIT.KINDS,
                    help="zero - stub (GAE degenerates into Monte Carlo without a "
                         "baseline, the loop is only checked this way); "
                         "tower - R1 value tower")
    ap.add_argument("--critic-ckpt", default="",
                    help="separate tower checkpoint (`vtower.*`), if it is "
                         "warmed up by a file other than the policy's")
    ap.add_argument("--resume", default="", help="state-*.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="", choices=["", *PRESETS],
                    help="full-scale plumbing set (`config.PRESETS`), before --set")
    ap.add_argument("--set", nargs="*", default=[],
                    help="targeted config edits: ppo.clip_range=0.1")

    a = ap.parse_args(argv)
    cfg = RunCfg(name=a.name, out=a.out, ckpt=a.ckpt, updates=a.updates,
                 seed=a.seed)
    cfg.critic = a.critic
    cfg.critic_ckpt = a.critic_ckpt
    if a.preset:
        override(cfg, PRESETS[a.preset])
    if a.set:
        override(cfg, dict(kv.split("=", 1) for kv in a.set))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run = Run(cfg)
    if a.resume:
        run.load(a.resume)
    run.run()


if __name__ == "__main__":
    main()
