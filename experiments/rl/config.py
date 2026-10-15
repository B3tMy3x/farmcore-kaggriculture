"""PPO loop settings: source defaults plus our rules on top.

Defaults come from `conf/ppo/default.yaml` of an earlier archive for the same
game (gamma 1.0, gae_lambda 0.95, clip 0.2, vf 0.5, ent 0.01, epochs 4,
minibatch 64). Wherever we deviate from the source, the comment says which
measurement motivated it. This matters: these are exactly the fields on which
the two sources disagree, and the choice has to be explainable months later.

Anything that can be switched off is switched off by a FLAG, not a code edit:
we only know how to compare versions pairwise by seed (`tools/bench/pair.py`),
and for that two arms must differ in exactly one field.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import value_head as VH  # noqa: E402


@dataclass
class PPOCfg:
    """Policy update."""

    # --- from the archive's conf/ppo/default.yaml, change only by measurement ---
    gamma: float = 1.0
    # gamma = 1 is not "by convention": a game is exactly 720 turns, the reward
    # is terminal (money in the bank at season end); discounting here would
    # distort the competition objective rather than fight an infinite horizon.
    gae_lambda: float = 0.95
    # Credit window (25.09): K > 0 — a turn's return = plain sum of rewards over
    # K turns without bootstrap, the critic target is the same, GAE is unused.
    # Baseline: "critic" (V(s) trained on the same target) or "turn_mean" (mean
    # return by turn index — for `snr` probes without retraining the critic).
    window_k: int = 0
    window_baseline: str = "critic"
    # λ of the CRITIC target (`buffer.Batch`). < 0 — same as `gae_lambda` (as
    # before). 1.0 — Monte Carlo: with γ = 1 the target is the game outcome, no
    # critic estimates inside it. Measured on rl1 23.09: with shared λ = 0.95
    # the critic loss dropped 3x, mean V sat at 0.65 with win rate ~0.5, and
    # `explained_var_mc` was near zero (the critic learned to agree with itself).
    value_lambda: float = -1.0
    clip_range: float = 0.2
    value_coef: float = 0.5
    # ⚠ MEASURED 23.09 (probe on value_tower/e11.pt, smoke on 2
    # games): policy gradient norm BEFORE clipping is 28-32 vs threshold 0.5,
    # i.e. clipping binds on **100% of steps** (`ppo/grad_clipped_frac` = 1.0).
    # This is not outlier insurance but constant renormalisation: per-minibatch
    # differences in gradient magnitude are erased and only direction gets into
    # the step. With AdamW the scale mostly cancels anyway (the step is
    # invariant to a CONSTANT gradient multiplier), so "lr does nothing" is
    # wrong; the opposite, that the threshold is harmless, is wrong too: the
    # multiplier here is not constant, it is 0.5/‖g‖ and varies per minibatch.
    # Do NOT change the number without a paired measurement: two arms must
    # differ in one field (`--set ppo.max_grad_norm=...`). The metric showing
    # this is `grad_clipped_frac`, always logged.
    max_grad_norm: float = 0.5
    epochs: int = 4
    # Critic-ONLY epochs, on top of `epochs`, outside the `target_kl` gate
    # (`ppo.Trainer._value_phase`). 0 = as before: the critic trains in the same
    # epochs as the policy and stops together with it — exactly when its target
    # is most stale. PPG (arXiv:2009.04416) measures an optimum of 1 policy epoch
    # vs 6 critic epochs, so one number cannot set both. Default 0 is kept so
    # enabling it is a visible decision measured pairwise.
    value_epochs: int = 0
    # Importance weight for collection lag (`RunCfg.overlap`): w = π_θk/π_θk−1,
    # clipped from above by this number (ρ̄ from V-trace), applied to the
    # advantage (`ppo.Trainer.update`). 0 = off. Mandatory with `overlap = true`
    # (`train.Run` refuses to start without it) and requires
    # `rollout.logp_every = 1`. 1.0 — like c̄ in V-trace; not measured in
    # money. The per-update shift is small (smoke: w ∈ [0.71, 1.0x], mean
    # 0.996), so the upper clip cuts almost nothing — see `ppo/is_clip_frac`
    # and `ppo/is_max_abs_log`.
    behavior_rho: float = 0.0
    minibatch: int = 64
    normalize_adv: bool = True
    # Group advantage instead of GAE (24.09). Requires `rollout.group_size > 1`:
    # a group is K games with one env seed against one opponent from one seat,
    # with different sampling seeds of our policy (`collect.collect_serial`).
    #   "gae"    — as before: GAE over the critic's V;
    #   "rloo"   — A_i = R_i − mean_{j≠i} R_j over the full game return, one A
    #              for all turns of the game (RLOO, arXiv:2402.14740);
    #   "grpo_t" — G_t (return from turn t to the end, γ = 1) minus mean G_t
    #              over the other games of the group at the SAME turn t (GR2PO,
    #              arXiv:2609.19850). With dense reward it gives temporal credit.
    # The critic is not part of the advantage; the critic target (`ret`) is
    # computed as before, so it can still train for metrics. Buffer
    # normalisation (`normalize_adv`) comes after, as before.
    advantage: str = "gae"
    # Divide the group advantage by the group std (GRPO; for `grpo_t` — std
    # over the group at each turn t). False — only subtract the baseline.
    group_std: bool = False
    # Breakdown of `clip_frac` and `max |log r|` by factor groups (H11).
    # Computed from the already-computed network output, no second pass; the
    # flag is for the case where 6 extra `log_prob` per minibatch become visible.
    group_diag: bool = True

    # --- our rules on top of the source ---
    # Entropy bonus is OFF, although both sources use 0.01. Reason: in a public
    # notebook for this same game, entropy moved mass off the neutral action
    # from 91.6% to 46.5% and wrecked the economy. The regulariser role is
    # played by the KL leash to the BC anchor (R7), and it REPLACES entropy
    # rather than adding to it. Flag kept so it can be measured.
    ent_coef: float = 0.0
    # KL leash to the frozen BC checkpoint (R7, H9): computed over the latent,
    # analytically per factor, with the same masks and temperatures as `log π`.
    kl_coef: float = 0.1
    # Soft leash: if KL creeps above target the coefficient grows, below it
    # shrinks (as in TRPO-like implementations). 0 = constant coefficient.
    kl_target: float = 0.0
    kl_adapt: float = 1.5

    # Dual-clip (Ye et al., JueWu): lower bound c·A for A < 0. Vanilla
    # min(rA, clip(r)A) is unbounded below — for large r the rA term goes to
    # −inf and min picks exactly it. Our ratio is a sum over ~29 factors of a
    # turn (H11), so large |log r| are more likely than usual and the guard is
    # not theoretical. c = 3.0 — value from a previous project of ours.
    dual_clip: float = 3.0

    # Value-function clipping. A previous project of ours clips; the archive and
    # Andrychowicz et al. do not ("no value-function clipping"). We take "no";
    # flag in case the critic diverges.
    value_clip: float = 0.0
    # Huber instead of MSE — rejected on the same grounds ("MSE, no Huber").

    # Early stop on KL inside an update. A previous project of ours checks on
    # EVERY minibatch (measured: checking once per epoch took win rate 0.84 →
    # 0.49 in two updates), gate 1.5·target. 0 = off.
    target_kl: float = 0.03
    kl_stop_mult: float = 1.5

    # Gradient accumulation: N minibatches per optimizer step, weighted by
    # sample count in the group (the archive's `PPO_ACCUMULATION.md`).
    grad_accum: int = 1
    # With sparse exploration (`rollout.explore_turns`) train only on sampled
    # turns: deterministic ones carry A = 0 and drown the 1-2 signal turns in a
    # minibatch (`buffer.Batch._finish`). False — as before.
    explore_only: bool = False
    # Noise arm: shuffle A between sampled turns within a group
    # (`buffer.Batch._finish`). Control only.
    adv_shuffle: bool = False
    # Step mode (25.09). "ppo" — as before: minibatch epochs, AdamW, KL gate.
    # "kl_line" — one step per update: policy gradient over the WHOLE buffer
    # (at r = 1 the PPO surrogate and REINFORCE coincide), step along its
    # direction, length found by KL search on held-out groups up to `line_kl`.
    # Why: AdamW with 1-2 steps per update is sign descent, `lr·sign(g)` for
    # each of ~6M weights, and weights with noise gradients move as far as those
    # with signal (rl5: KL 0.035 per step). The gradient direction itself
    # weights parameters by signal. No optimizer involved; `lr`, `epochs`,
    # `clip_range`, `target_kl` are not read in this mode.
    step_mode: str = "ppo"
    line_kl: float = 0.02         # target KL(π_old ‖ π_new) of the step
    line_alpha0: float = 0.02     # first step-length probe (Euclidean norm)
    line_iters: int = 6
    line_holdout: float = 0.1     # share of GROUPS outside the gradient: KL and gain probe
    line_probe: int = 16384       # turns in the probe (from held-out)

    # ⚠ H3: recompute in fp32, no autocast. bf16 gives |Δ log π| p99 0.295,
    # i.e. 6.4% of turns exceed clip 0.2 from precision alone; the archive on
    # A100 rejected a bf16 run for the same reason (max drift 0.169). Field kept
    # so that "enable bf16" is a visible decision, not a default.
    fp32: bool = True
    # H3(2): `log π_old` is recomputed by the trainer with the same function at
    # the start of the iteration rather than taken from the rollout. Then the
    # CPU/GPU mismatch ~1e-5 never enters the ratio, and the rollout `log π`
    # stays as a control: at step zero the ratio must be exactly 1.
    recompute_old_logp: bool = True

    # CUDA graph on the TRAINING step: one minibatch (policy and anchor forward,
    # `log π`, losses, backward, zeroing non-finite gradients, norm clipping)
    # is recorded as a graph and then only replayed. This is plumbing, not
    # algorithm: same arithmetic in the same order, and the training parity
    # check requires BITWISE-identical weights after N steps. Reason — same as
    # in inference: constant kernel-launch overhead (there 12.4 ms per pass
    # regardless of batch, ×6.9 with a graph).
    #
    # ⚠ The graph only takes FULL minibatches: the graph has a fixed shape and
    # the last minibatch of an epoch is partial. The tail goes the old path —
    # silently dropping it is not allowed, that would change the algorithm.
    # ⚠ Incompatible with `grad_accum > 1` (the graph zeroes gradients itself)
    # and with `kl_target > 0` (the leash coefficient then changes between
    # updates, while in the graph it is a constant). In both cases the graph is
    # silently not enabled, and a line is printed about it.
    cuda_graph: bool = False
    # ANCHOR output is computed once per update, in the `log π_old` pre-pass,
    # and kept on the GPU until the update ends (`ppo.Trainer._ref`). The anchor
    # is frozen, its output on buffer states does not depend on the epoch, yet
    # previously its forward pass ran on EVERY minibatch of every epoch. Saving
    # — (epochs − 1) anchor passes per turn: at `epochs = 1` it is exactly zero.
    # Parity is BITWISE (`--test ppo.anchor_cache=true`): the pre-pass anchor
    # uses the same batch size as the minibatch, and an output row does not
    # depend on its batch neighbours (measured: 0 mismatches with different
    # composition of the same size, ~7e-6 with a different SIZE). The partial
    # epoch tail computes the anchor the old way, at its own size. Memory:
    # 33.6 KB per turn (9 logit heads, fp32), 0.39 GB per 11.5k turns, 3.1 GB
    # per 92k.
    anchor_cache: bool = False
    # Whole rollout buffer on the GPU, minibatch via `index_select` there,
    # instead of `np.stack` + transfer on EVERY minibatch (`Batch.pin`).
    # 39.8 KB per turn. Values bitwise identical: in both cases the network
    # gets a freshly allocated tensor.
    resident_buffer: bool = False
    # Relative attention shift (`model.RelBias`) for unit pairs — one-hot by
    # MULTIPLICATION instead of `tab[:, idx]`. Forward is bitwise identical
    # (output, `log π`, entropy, KL), and backward becomes a GEMM instead of
    # `index_put_` with accumulation over B·H·n·m indices (21-27% of the GPU
    # step). The shift-table gradient differs in summation ORDER (rel.
    # 2.6-5.2e-07) and becomes deterministic; other policy gradients are bitwise
    # identical. Weights after N steps are therefore not bitwise equal to the
    # old path — checked against the old path's own run-to-run spread
    # (`--test ppo.relbias_onehot=true`).
    relbias_onehot: bool = False
    # Group breakdown of `clip_frac` (H11, `group_diag`) inside the step's CUDA
    # graph rather than after it. Only meaningful with `cuda_graph`. Computed on
    # every minibatch, read on the first epoch; values bitwise identical.
    graph_groups: bool = False
    # AdamW `fused=True` (one kernel per step) instead of foreach. GPU only.
    fused_adamw: bool = False
    # V(s) for GAE — from the `log π_old` pre-pass, not a separate
    # `collect.score` pass after collection (R6, second round, 24.09). The
    # pre-pass already computes the tower (`critic.wants_value`) and threw it
    # away; `score` ran THE SAME trunk over the same turns with the same weights
    # again (6.4 s of the rl1 cycle per timeline). The rollout is then collected
    # without a critic, `Batch` defers GAE until `set_values`,
    # `diagnostics.summarize` is called after the update. Parity checked: V,
    # GAE and post-update weights against the old path. With `overlap`, V is
    # computed by the current critic too, i.e. the same quantity `score` gave.
    prepass_value: bool = False
    # Critic phase (`value_epochs`) over a CACHE of the trunk output. The policy
    # does not step in this phase, so `h` (tower input) does not change across
    # epochs, yet previously each step ran the full network (trunk, heads,
    # corrector) for a tower on `h.detach()`. The cache is computed once after
    # the policy epochs with batch `minibatch` (a row does not depend on batch
    # neighbours, but does on SIZE, so the partial tail goes the old path).
    # Memory: 73 KB per turn (146 tokens × 128 × fp32), 4.4 GB per 60k turns.
    value_cache: bool = False

    lr: float = 1e-5         # archive in production: 1e-5 (BC trained at 8.5e-4)
    # CRITIC lr (second AdamW in `ppo.DualAdamW`). < 0 — same as `lr` (as
    # before, bitwise, checked by a probe). Why separate (rl1): in phase B
    # `lr` was tripled for the policy, and the critic on a Monte Carlo target
    # drifted with it — policy and critic have different step costs, and one
    # number cannot set both.
    value_lr: float = -1.0
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    # H10: start from `ck["model"]` (that is the EMA) with a FRESH optimizer —
    # `ck["opt"]` belongs to the raw weights, not the EMA.
    fresh_optimizer: bool = True


@dataclass
class PPGCfg:
    """Auxiliary phase for the critic (Cobbe et al., arXiv:2009.04416).

    ⚠ Off by default, and not out of caution: PPG fixes the policy/critic
    conflict over a SHARED trunk, while our critic (R1) is a separate tower on
    `h.detach()` with its own parameters. Nothing to fix. The phase is ported
    verbatim so it can be enabled if the critic ever moves onto the shared
    trunk.

    A pitfall from a previous project of ours to remember: the aux phase there
    silently never ran if the policy phase was cut by the `target_kl` gate —
    with target_kl 0.03 and 0.05 the aux-update counter was EXACTLY ZERO on
    every update. So here the phase runs regardless of the cut, and
    `ppg_updates` is always logged (the health metric is gated on the key's
    PRESENCE, not its truthiness, otherwise it vanishes exactly in the mode it
    warns about).
    """

    aux_epochs: int = 0
    clone_coef: float = 1.0
    # The previous project estimates KL to the reference as 0.5·(Δ log π)² —
    # to second order that is what it is, computed without a second pass over
    # the distribution. Ported as is. The analytic `policy.kl` cannot be
    # plugged in directly: it measures distance to the ANCHOR, whereas the
    # phase needs a reference taken AFTER the policy phase, i.e. another weight
    # snapshot. Until it exists, `exact_kl = True` fails loudly rather than
    # silently computing the wrong quantity.
    exact_kl: bool = False


@dataclass
class RewardCfg:
    """Reward. Kept separate from the game's "true outcome", as in the archive.

    `episode.outcome` (±1 by bank) is ALWAYS computed and feeds win rate, the
    league and history admission. `training_reward` is what GAE sees, and it
    may differ. The archive explicitly warns about confusion here and separates
    the fields by name; we do the same.
    """

    # Form: r_T = ±1 + α·tanh(Δmoney/σ). ±1 — a bet on wins (the leaderboard is
    # a skill rating, not money), tanh — gradation between rollouts, without
    # which a terminal ±1 has no resolution.
    #
    # ⚠ α and σ are taken FROM `value_head`, not written as numbers here. Until
    # 23.09 this had 0.25 and $30 000 while the critic trained on 0.3 and
    # $25 000 — the advantage would have come out at the wrong scale and no
    # measurement would have shown it (exactly why the reward formula, the
    # critic target and σ must live in ONE file). Change them there. The check
    # is in `ppo.Trainer._check_scale`, and it takes σ not from the default but
    # from the critic checkpoint's `cfg`: `bc_train.py --value-sigma` overrides
    # the default.
    outcome_scale: float = 1.0
    cash_alpha: float = VH.RL_ALPHA
    cash_sigma: float = VH.RL_SIGMA

    # Dense shaping on cash margin (archive, `cash_delta_reward`):
    # Φ(s) = cash_shaping · margin(s) / σ, bonus on transition
    # `γ·Φ(s') − Φ(s)`. Opponent money is public
    # (`obs["farms"][1-p]["money"]`), so the quantity is computable at decision
    # time.
    #
    # ⚠ FIXED 23.09. Before, this said "the sum telescopes to (final margin −
    # initial margin)/scale, so it is potential shaping in Ng's sense and does
    # not shift the optimum" — WRONG twice. (1) Ng's theorem (Ng, Harada,
    # Russell, ICML 1999) for episodic tasks requires Φ(terminal) = 0;
    # telescoping "to the final margin" is exactly the case where it is NOT
    # zero, and a term LINEAR in the bank difference is added to the reward on
    # top of `α·tanh(Δ/σ)`. So with cash_shaping > 0 the objective changes and
    # no measurement would show it — default 0 was the only thing saving us.
    # (2) The old code also computed the last bonus from FINAL banks, i.e.
    # counted the terminal money difference TWICE.
    # Now Φ is cut to zero at the terminal (`collect._turn_rewards`), the sum
    # over a game is −Φ(s₀) — a constant, and the optimum does not shift by the
    # theorem (checked by a probe).
    #
    # ⚠ What remains: with shaping the correct estimate is `V(s) − Φ(s)`, while
    # our critic is trained on the UNshaped target: enabling shaping on a warm
    # critic means starting from a knowingly biased baseline (it converges, but
    # the first updates follow a wrong advantage).
    # Default 0: first measure the loop on a clean terminal signal.
    cash_shaping: float = 0.0
    # Dense reward from the ARCHIVE for the same game (rl3, 24.09;
    # `Kaggriculture-no-data.zip`, docs/RL_GAE719_CASH_SHAPING.md and
    # RL_GAE64_PURE_CASH.md — PPO worked there): each turn r_t = (margin_{t+1}
    # − margin_t) / scale, margin = own money − opponent's; at the end also
    # `outcome_scale`·sign(Δ). With γ = 1 the game sum = final margin / scale
    # + ±1: the target is the outcome, not money along the way. Density puts the
    # real money increment into each turn's δ — that is how it differs from the
    # potential-based `cash_shaping` above, which with a consistent critic does
    # not change δ at all. Cost: bias toward the near term with a poor critic
    # (a purchase immediately shows a minus; the archive's "investment-credit
    # bias"). 0 = off (old terminal reward). Requires a critic with its own
    # scale (`--critic tower_scalar`); with the ±1.3 tower —
    # refused in `ppo.Trainer._check_scale`.
    margin_delta_scale: float = 0.0
    # Local reward on farm net worth (25.09):
    # r_t = (Φ(s_{t+1}) − Φ(s_t)) / scale, Φ = `networth.phi_obs` — OWN farm
    # with shed (money, stock at revenue value, seeds, crops, animals, land);
    # the opponent is excluded (its shed is invisible, the visible part is
    # twice as noisy). On the last turn Φ(s') = own money, so the game sum
    # telescopes to (own money − Φ(s₀)) / scale: the target is own money at
    # season end, while a turn's credit shows up within 6-24 turns (sign holds
    # in 32/36 pairs vs 11/36 for money), when money is still zero.
    # Scale: std ΔΦ per turn in self-play ~$650, at 1000 the reward is ~N(0, 0.65). Only meaningful together with λ ≈ 0.9: a turn's effect on
    # the outcome 72+ turns later is a coin flip. Requires a critic with free
    # scale (`--critic tower_scalar`). Mutually exclusive with the two rewards above.
    phi_delta_scale: float = 0.0


@dataclass
class RolloutCfg:
    """Collection."""

    episodes: int = 8          # games per update
    seed0: int = 9000
    # Evaluation panel: games where we also play argmax (`collect.collect_eval`).
    # Only on these is it fair to compare ourselves with a checkpoint opponent,
    # which always plays argmax (H1/R4: decoder asymmetry costs −$1 784, 16/48
    # on the SAME weights, i.e. equal strength gives a rollout win rate ≈ 0.33).
    # The history admission threshold is applied on them too (`league.promote`).
    # 0 = no panel, and then `promote` uses the biased rollout win rate and says
    # so in the log rather than staying silent.
    eval_episodes: int = 0
    eval_seed0: int = 20000    # own seed block: do not share with the rollout
    # Evaluation IN PARALLEL with the update (`train.Run.step`): plays the
    # weights snapshot BEFORE update k (the actor saved at the end of k−1) while
    # the GPU trains k. League admission lags by one update, evaluation numbers
    # are the same. Measured on rl1 23.09: evaluation of 8 games is 15-18 s of
    # wall time per update out of ~140, of which server start ~3 s, the rest
    # the games themselves.
    eval_async: bool = False
    # Whether to move the evaluation seed block from update to update.
    # False (default) — move: a gate that repeatedly looks at ONE block starts
    # selecting for it (the archive explicitly warns about a panel that is also
    # used for selection). True — keep one block: comparisons between updates
    # become PAIRED by seed and resolve a 2-3x smaller shift (measurement
    # protocol, `pair.py`), at the cost of selecting for the panel. League
    # admission uses the former; for "did the arm improve over 50 updates" the
    # latter is more honest, and then it is a separate run, not the same gate.
    eval_seed_fixed: bool = False
    # Processes for collection (`collect.collect_parallel`); the core quota is
    # read by `play.cpu_quota` (15 on our training machine, not the 28 from
    # `nproc`). Games are independent by construction, so the split must give a
    # BITWISE identical rollout — checked by a probe.
    #
    # MEASURED 23.09 (A100 node, 15 cores, load 2.1-2.8, GPU idle), 8 full
    # self-play games = 11 504 decisions, `--critic zero`, one torch thread per
    # process:
    #   jobs= 1   345.5 s   43.2 s per game     33 decisions/s
    #   jobs= 4    93.7 s   11.7 s              123        (×3.69)
    #   jobs= 8    55.3 s    6.9 s              208        (×6.25)
    #   jobs=15    55.3 s    — same: eight games, no more chunks possible
    # Parity bitwise in all three cases: max |Δ log π| = 0, max |Δ bank| = 0.
    # The gap to ×8 is worker startup cost (torch import plus weight loading)
    # and game length spread: one game per chunk, and the longest one sets the
    # wall time.
    # ⚠ Setting jobs above the number of games in a rollout is pointless.
    # ⚠ Do not combine with `device = "cuda"` without checking: J processes
    # would create J CUDA contexts, and batch 1 on the GPU is already 4x worse
    # than a CPU core (R6).
    jobs: int = 1
    # ⚠ With `serve = true` set this MUCH HIGHER than the core count. Profile of
    # a live worker (23.09): 49% of its time it WAITS for the server reply
    # (11.98 s of 24.37), and a waiting process does not occupy a core.
    # Measured (15 cores), seconds per game, games split evenly per worker:
    #   J = 14 -> 1.94   J = 28 -> 1.47   J = 42 -> 1.37   J = 56 -> 1.39
    # So 3x over the core quota gives ×1.42, then it plateaus. Not applicable to
    # the per-process split (`collect_parallel`): there the worker runs the
    # network itself and occupies a core all the time.
    device: str = "cpu"        # ⚠ on cuda batch 1 is 4x worse than a CPU core (R6)
    # What runs the game: the competition harness or the Rust port
    # (`engines/kaggriculture_rs`). The port was checked by OUR measurement on
    # OUR rollouts — 43 130 transitions and 86 400 observations, 0 mismatches
    # besides two declared ones (`remainingOverageTime`, which is not a network
    # input, and an extra `step` for player 1 with the same value). MEASURED
    # 23.09 (A100, load 3.0-3.5 of 15, OMP_NUM_THREADS=1), 4 full self-play
    # games, 5 752 decisions: 174.1 s on the harness vs 138.0 s on the port
    # (×1.26), and the banks MATCHED to the dollar — $85 193 in both, i.e. the
    # very same game was played.
    #
    # Where ×1.26 comes from when the interpreter is only 7% of the profile: the
    # cost is not it but the harness around it. `kaggle_environments` does a
    # `deepcopy` of the whole state (`__get_shared_state`) and `structify` of
    # all dicts on every agent call — together 28% of the profile vs 7% for the
    # mechanics itself.
    #
    # Default is still the harness: the port changes the environment under ALL
    # runs at once, and switching must be deliberate, via a separate
    # `--set rollout.env=rust`.
    env: str = "kaggle"        # "kaggle" | "rust"
    # Native feature encoder (`engines/kaggriculture_rs/src/encode.rs`) instead
    # of Python `features.encode`. Only together with `env = "rust"`: it reads
    # engine state DIRECTLY, bypassing the observation.
    #
    # Checked element-wise on ALL nine output keys, over whole games (there is
    # an accumulator `FlowTracker`, and a mismatch in it would not show up
    # immediately): 51 768 observations, 471 451 176 numbers, 0 mismatches (24
    # stress games and 12 with our decoder, 23.09). Separately proven that with
    # the flag off `features.encode` computes bitwise the same as before the
    # change, and that the cache fingerprint change does not change the DATA
    # (0 differences on all 28 keys).
    native_encode: bool = False
    # Inference server (`infer_client.collect_served`): `jobs` processes without
    # weights plus one GPU that batches them via a CUDA graph. The worker plays
    # the same `collect_serial`; only the network object is swapped.
    #
    # MEASURED 23.09 (A100, 15 cores, load 2-4, OMP_NUM_THREADS=1),
    # MIRROR self-play, 28 games / 40 264 decisions over 14 processes:
    #   harness, split (as before)              317.4 decisions/s
    #   port, split                             376.7
    #   port, server WITHOUT graph (8 procs)    233.8  ← WORSE than split
    #   port, server + graph                    828.9  ×2.61 vs first row
    # The third row is not a typo and is the main finding: cross-process
    # batching without a CUDA graph loses to not batching at all. The worker
    # just waits out the constant ~12.4 ms per pass. Hence `serve_graph`
    # defaults to `True`, and disabling it only makes sense for bitwise checks.
    #
    # ⚠ With a FROZEN opponent the gain is 3x smaller — ×1.30 (16 games,
    # 8 processes: 78.9 s vs 60.6 s). The server serves ONLY the training
    # policy; the league opponent loads its own weights in the worker and plays
    # batch 1 on a CPU core, and that is half the decisions of a game. Mirror
    # games in the league plan gain 3x more than the rest.
    # How often to compute `log π` IN PLAY. There it is a CONTROL, not training:
    # the trainer recomputes it itself, batched (`ppo.recompute_old_logp`,
    # H3(2)), and the recomputed value goes into the ratio. In play it costs
    # 1.60 ms per turn — more than `features.encode` (1.55 ms), and that at
    # batch 1.
    #
    # `1` — as before, every turn. `N > 1` — every N-th: the `ppo/logp_drift`
    # control does NOT lose its meaning (exactly the same two quantities are
    # compared, play `log π` vs trainer recompute), it is just measured on a
    # sample of turns.
    #
    # ⚠ Incompatible with `ppo.recompute_old_logp = False`: then the rollout
    # `log π` IS the ratio denominator and cannot be skipped. `ppo.Trainer`
    # refuses such a combination loudly.
    logp_every: int = 1
    serve: bool = False
    serve_device: str = "cuda"
    serve_graph: bool = True
    # Whether the server also serves CHECKPOINT OPPONENTS. Without it the server
    # computes only our half of the decisions, and the league opponent loads its
    # own weights in the worker and plays batch 1 on a CPU core — that is where
    # the ×2.61 in mirror self-play vs ×1.30 with a frozen opponent came from.
    #
    # ⚠ Packages (`kind == "package"`) cannot be served at all: they run THEIR
    # OWN code with their own modules, and there is no network object to swap.
    # Their share of the plan is bounded by the `frozen` category quota.
    serve_opponents: bool = True
    # How many requests the server waits for before computing. 1 — compute
    # immediately. A pass costs almost the same at batch 1 and batch 14 (1.81 vs
    # 2.7 ms with a CUDA graph), and the mean batch with 14 workers is only 3.4:
    # the server replies faster than workers come back. Waiting `serve_max_wait`
    # seconds and computing twice as much for the same cost is a good deal as
    # long as the wait is shorter than a pass.
    # ⚠ MEASURED AND NOT ADOPTED. `serve_min_batch = 7` with 14 workers did make
    # the server cheaper (in-pass 15.8 -> 14.8 s) and raised the mean batch
    # (3.5 -> 3.7), but the rollout did not move: 27.19 s vs 27.38 s, i.e.
    # within noise and if anything worse. The added waiting ate the savings one
    # to one. Left off: accumulating only makes sense if the server is
    # saturated, and it does not saturate here.
    serve_min_batch: int = 1
    serve_max_wait: float = 0.002
    # Collection core layout (`infer_client.pick_cpus`). The container quota is
    # 15 CFS cores out of 28 visible: without pinning 42+ workers run on all 28
    # at once, exhaust the quota within ~54 ms of the period, and CFS freezes
    # the WHOLE container until the period ends — including the server everyone
    # waits on. `serve_pin`: the collection tree goes on `quota` cores (one
    # logical per physical where possible), the server on ITS OWN core, workers
    # on the rest. `worker_nice`: worker niceness to the scheduler (server stays
    # at 0; lowering nice needs privileges, raising does not).
    # `serve_block_sync`: the server waits for the GPU on a blocking event
    # instead of spinning on a core.
    # None of the three changes numbers — only who gets a core and when.
    serve_pin: bool = False
    worker_nice: int = 0
    serve_block_sync: bool = False
    # Server input and output through pinned memory, asynchronous transfers
    # (`infer_server.serve`). Same values: the graph gets the same buffers.
    serve_pinned: bool = False
    # Games to workers via a QUEUE (shared counter) rather than fixed chunks
    # (`infer_client.collect_served`). Collection wall time is the slowest
    # worker: with chunks one long game holds everyone. Experience is the same:
    # a game depends on its index, not on who plays it (order is restored by
    # seed).
    serve_queue: bool = False
    # Groups of one server wake-up (policy + league checkpoint opponents, each
    # with its own weights) run on THEIR OWN CUDA streams concurrently, one sync
    # per group on its event, instead of pass-then-sync in sequence
    # (`infer_server.serve`). Requires the graph and `serve_pinned`. Pass
    # numbers bitwise identical; batch composition with J > 1 depends on timing
    # anyway.
    serve_streams: bool = False
    # Collection processes are forked from a forkserver with modules
    # pre-imported (`infer_client._context`), instead of `spawn` importing torch
    # in each of 42 workers on every collection. Does not change numbers
    # (`collect_parity --what server --set rollout.serve_forkserver=true`).
    serve_forkserver: bool = False
    # Reading worker results with the garbage collector disabled
    # (`infer_client.collect_served`). Does not change numbers.
    load_nogc: bool = False
    # Both seats in mirror self-play are experience of the training policy, and
    # both are on-policy. The archive and a previous project of ours take only
    # the actor's seat because their opponent is almost always frozen. In
    # mirror mode this doubles the sample for free, hence the flag: it has no
    # effect against a frozen opponent.
    both_seats: bool = True
    # GAE chunking (archive, `--gae-horizon 64`). ⚠ Not adopted: the bootstrap
    # at a chunk boundary comes from a frozen `values[end]`; the authors
    # themselves write "adds bootstrap bias" and did not measure the effect. Our
    # reward is terminal, and with horizon = 64 the signal reaches the end of
    # the game ONLY through the critic, which does not exist yet (R1).
    # 0 = one backward pass over the whole game.
    gae_horizon: int = 0
    # Do not keep each turn's `obs` in memory: recomputation needs only `enc`,
    # `legal0` and the latent. `obs` is needed by the log π parity check, not training.
    keep_obs: bool = False
    # ⚠ `max_turns` truncates the RECORD, not the game: the env still plays its
    # 720 steps (truncation happens after `runner.run`), so it saves NO time —
    # measured 23.09: a game is 9.0 s both with `max_turns = 24` and without.
    # Worse, the truncated stream receives the terminal reward for the FULL game
    # at turn 24, so credit assignment in such a rollout is knowingly wrong.
    # Only for mechanics checks, never for training or speed
    # measurement.
    max_turns: int = 0         # 0 = whole game; >0 — RECORD truncation for smoke
    # Group collection for the group advantage (`ppo.advantage`). K > 1: each
    # drawn opponent plays 2K games in a row — K with us in seat 0 and K in
    # seat 1 (seat — parity of the index, as always) with ONE env seed
    # `seed0 + index of the block's first game` and sampling seed `seed0 + game
    # index` (each its own). Advantage group — (env seed, seat, opponent).
    # 1 — as before: env seed = sampling seed = `seed0 + index`.
    # `rollout.episodes` must be divisible by 2K.
    group_size: int = 1
    # Temperatures of OUR sampling in the rollout and in the `log π`
    # recomputation — in one place (`collect.ppo_decode`). −1 = from
    # `policy.PPO` (unit τ 0.5, quantity τ 1.0) — as before. `sample_units = 0`
    # — unit target and work are argmax, `log π` and KL exclude them: sampling
    # and gradient only on market gates, "all", land and quantities ("sample a
    # subset of factors" probe, 24.09).
    sample_units: float = -1.0
    sample_qty: float = -1.0
    # Common random numbers (CRN) per turn: OUR sampling generator is reseeded
    # every turn with (sampling seed, turn), the opponent's with its own stream
    # (env seed, turn) that does not touch ours. Then games in a group share the
    # opponent's randomness at turn t, and branching (vine) at one turn does not
    # shift the random stream on later turns. False — as before: one generator
    # per game, shared with the opponent.
    crn: bool = False
    # Sparse exploration (24.09). 0 — as before: our sampling on EVERY turn.
    # N > 0 — sampling (`ppo_decode` mode) only on N random turns of a game
    # (per game, from the sampling seed); on the rest the turn is
    # deterministic: gates by threshold, units argmax, quantities by median.
    # Why: 71-88% of outcome variance is noise of OUR sampling over 719 turns
    # (`rl_diagnose gshops`); it also keeps critic EV near zero and drowns the
    # advantage of any single turn. With `group_size > 1` + `crn` the games of
    # a group play identically until the first exploration turn, and the outcome
    # difference within the group is the effect of these N decisions. The
    # advantage on non-exploration turns is zeroed (`buffer.Batch`), `log π` is
    # not recorded there (a deterministic turn has probability 1).
    explore_turns: int = 0
    # Exploration turns shared by the group (from the env seed) rather than per
    # game (from the sampling seed). With `group_size > 1` and `crn` the games
    # of a group diverge only on these turns: the advantage is the exact effect
    # of the decision.
    explore_shared: bool = False


@dataclass
class LeagueCfg:
    """Opponent league: three categories by quota, PFSP by group inside the pool.

    Quotas rather than softmax over the whole pool, because the categories
    measure different things: the BC anchor answers "did we fall apart",
    latest — "is there growth", the pool — forgetting and connection to the
    ladder. Inside the pool the choice is TWO-LEVEL: first a group, then a
    group member, both times with weight
    `difficulty_floor + difficulty**difficulty_power` (our loss share, EMA)
    and a share floor (`pool_min_share`, `member_min_share`).
    A group is a BEHAVIOUR cluster, not an author: otherwise ten versions of one
    script would eat the rollout budget. Clusters are computed offline and
    stored here as a field: the league must not compute them on the fly.
    """

    bc_anchor: float = 0.25
    latest: float = 0.25
    # PFSP pool: frozen packages (scripts AND our own weights) plus history
    # snapshots. Replaced the old `history` (0.25) and `frozen` (0.25) — same sum.
    pool: float = 0.50
    # Group → packages with THEIR OWN code (`main.py` with `agent`, like
    # `play.py --opp`). Clustered by cross-agreement of turns (≥ ~80% within,
    # 45-60% between) and action fingerprint, 23.09. History snapshots are an
    # implicit group `history` (one policy lineage = one style, like a script
    # cluster). `corn` is a package of our own ladder weights (`corn_r8off e11`)
    # as a separate group.
    # ⚠ The other entries are third-party agents from public Kaggle notebooks
    # and are NOT shipped with this repo: place them yourself under
    # `opponents/<name>/main.py`. Missing packages are skipped with a warning,
    # and an empty pool leaves the league with only the anchor, latest and
    # history snapshots.
    groups: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "corn": ("opponents/corn_pkg/main.py",),
        "script_a": ("opponents/v47/main.py",
                     "opponents/kg_gametheory/main.py",
                     "opponents/kg_shoprouter7/main.py"),
        "script_b": ("opponents/kg_herdsafe/main.py",
                     "opponents/kg_popeconomy/main.py",
                     "opponents/kg_marketshock/main.py"),
        "script_c": ("opponents/kg_v41/main.py",),
        "script_v53": ("opponents/kg_v53/main.py",),
    })
    # Share floor of a group in the pool and of a member in a group:
    # p' = (1−kε)p + ε, k — number of groups (members). Without it an unbeatable
    # group drops to `difficulty_floor / Σw` ≈ 1-2%, and forgetting against it
    # is invisible.
    pool_min_share: float = 0.05
    member_min_share: float = 0.10
    # Snapshot of the training policy into history — only if it won its own
    # rollout above the threshold (archive: 0.55 → 0.65 → 0.70 over runs).
    promote_win_rate: float = 0.60
    history_limit: int = 24
    # EMA of opponent difficulty (our loss share). Never-played opponents count
    # as NEUTRAL (`difficulty_init`), not easy: otherwise a new pool member
    # starts with minimal weight and may never get played.
    difficulty_ema: float = 0.8
    difficulty_power: float = 1.0
    difficulty_floor: float = 0.05
    difficulty_init: float = 0.5


@dataclass
class RunCfg:
    """Whole run."""

    name: str = "rl0"
    # ⚠ NOT `experiments/net/runs`: on 23.09 that folder in the main checkout
    # became a symlink to itself and all checkpoints under that path were lost.
    # The PPO loop has its own directory.
    out: str = "experiments/rl/runs"
    ckpt: str = ""             # BC checkpoint: policy start AND anchor
    updates: int = 0           # 0 = forever, until signal
    save_every: int = 1
    seed: int = 0
    # `zero` — stub (GAE degenerates to Monte Carlo without a baseline, the loop
    # is only checked with it); `tower` — R1 value tower
    # (`value_head.ValueTower`), the production option.
    critic: str = "zero"
    # Separate tower checkpoint. Empty = the tower is taken from `ckpt`, the same
    # file as the policy — that is what `bc_train.py --value-only` produces,
    # where the tower is warmed on top of a frozen policy. The field is for the
    # "one policy, critic warmed separately" case: put a file with `vtower.*`
    # here, and ONLY those are taken — policy weights from it are not touched,
    # otherwise the arm would silently become a different one.
    critic_ckpt: str = ""
    # Where the TRAINER computes. Empty = same as the rollout (as before).
    # ⚠ These are different tasks with opposite preferences (R6): the rollout
    # runs at batch 1, where the GPU is 4x worse than a CPU core (10.78 ms vs
    # 2.64); the update runs at batch 64-256, where the GPU is tens of times
    # faster than the CPU.
    # MEASURED 23.09, forward+backward+AdamW step with value tower:
    #   batch   64:  CPU (15 threads)  921 ms,   69 states/s
    #                A100               56 ms, 1 144 states/s   (×16.5)
    #   batch  256:  CPU              4 205 ms,   61 states/s
    #                A100               87 ms, 2 930 states/s   (×48)
    #   batch 1024:  A100              300 ms, 3 416 states/s
    # For a rollout of 8 full games (11 504 turns) and four epochs this is
    # roughly 14 minutes of update on CPU vs half a minute on GPU.
    # ⚠ Do NOT touch the minibatch for speed: it is an algorithm parameter (64
    # from the source config), and changing it is a separate paired measurement.
    # Splitting devices only works with `rollout.jobs > 1`: weights go to the
    # workers as a file and are loaded on `rollout.device`, while the network
    # itself stays on `train_device`. With one process the rollout would run on
    # the trainer's network, i.e. batch 1 on the GPU, and `collect` refuses.
    train_device: str = ""
    # Collection IN PARALLEL with the update (`train.Run.step`): while the GPU
    # trains update k on rollout R_k, workers collect R_{k+1} with the weights
    # before that update. The cycle becomes max(collect, update) instead of the
    # sum. Cost: data lags by one update; corrected by an importance weight
    # (`ppo.behavior_rho`, mandatory) and requires `rollout.logp_every = 1`.
    # Two more consequences, both by one update: the league difficulty EMA used
    # to plan R_{k+1} does not see the results of R_k; the critic sets V(s) on
    # R_k AFTER update k−1 (fresher than in sequential mode).
    # Only with the inference server (`rollout.serve`): the per-process split
    # occupies the CPU, which the update also needs.
    # MEASURED 23.09 (mb 1024 ep 1, graph, load ~6-8): update alone 7.9 min per
    # 1e6 turns, during collection 8.7-17.4 (mean ~14); mirror collection alone
    # 13.3, during an update 15.3. Sequential ~21.5.
    overlap: bool = False
    ppo: PPOCfg = field(default_factory=PPOCfg)
    ppg: PPGCfg = field(default_factory=PPGCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    league: LeagueCfg = field(default_factory=LeagueCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Production presets for `train.py --preset`: PLUMBING only — what changes
# speed, not the algorithm. Minibatch, epochs, lr are not here and must not be:
# those are arms of paired money measurements. Every field is proven by a
# parity check, except those marked. The preset is applied BEFORE `--set`, so
# any field can be overridden, and the run's `config.json` holds the final
# values, not the preset name.
PRESETS: Dict[str, Dict[str, str]] = {
    "fast": {
        "train_device": "cuda",
        # collection: inference server with graph, env port, native encode;
        # workers 3x the cores — they spend half their life waiting on the server (R6)
        "rollout.jobs": "42", "rollout.serve": "true", "rollout.env": "rust",
        "rollout.native_encode": "true", "rollout.logp_every": "64",
        # core layout and pinned memory: numbers unchanged;
        # mirror 11.1-11.4 -> 9.3-9.7 min per 1e6 turns (23.09)
        "rollout.serve_pin": "true", "rollout.serve_block_sync": "true",
        "rollout.serve_pinned": "true",
        # evaluation in parallel with the update: snapshot before the update,
        # league admission one update later
        "rollout.eval_async": "true",
        # trainer: step graph, buffer on GPU, group metrics in the graph —
        # bitwise (training parity check)
        "ppo.cuda_graph": "true", "ppo.resident_buffer": "true",
        "ppo.graph_groups": "true",
        # ⚠ NOT bitwise: summation order of the shift-table gradient, 1.5e-6 —
        # within the GPU's own spread on the tower (2e-6)
        "ppo.relbias_onehot": "true",
        # `anchor_cache` is not included: at `epochs = 1` it saves nothing and
        # makes the pre-pass more expensive; enable at epochs ≥ 2
    },
    # Same plus collection in parallel with the update. This is NOT pure
    # plumbing: data lags by one update, corrected by an importance weight (see
    # `RunCfg.overlap`). Compare the arm against `fast` in money, don't trust it.
    "overlap": {
        "overlap": "true", "ppo.behavior_rho": "1.0", "rollout.logp_every": "1",
    },
}
PRESETS["overlap"] = {**PRESETS["fast"], **PRESETS["overlap"]}


def override(cfg: RunCfg, dotted: Dict[str, Any]) -> RunCfg:
    """Targeted overrides by dotted path: `ppo.clip_range=0.1`.

    Needed exactly for the measurement protocol: two arms must differ in one
    field, and that field must be visible in the run log, not in a code diff.
    """
    for key, val in dotted.items():
        obj: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        cur = getattr(obj, parts[-1])
        if isinstance(cur, bool) and isinstance(val, str):
            val = val.lower() in ("1", "true", "y", "yes")
        elif isinstance(cur, (int, float)) and not isinstance(cur, bool):
            val = type(cur)(val)
        setattr(obj, parts[-1], val)
    return cfg
