# Farmcore — Kaggriculture solution

Code of team **Farmcore** for the Kaggle simulation competition
[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) (two farms, one
shared market, 720 turns). Writeup:
[kaggle.com/competitions/kaggriculture/discussion/745119](https://www.kaggle.com/competitions/kaggriculture/discussion/745119).

The final agent is **pure behaviour cloning** — no self-play, no search at play
time:

- **Labels are what the engine executed**, not what the player ordered. Strong
  players over-order and the engine silently drops the rest; every corpus game is
  replayed through the engine and only the executed state change is kept.
- **One forward pass per turn.** A transformer drafts every unit's target tile and
  action and every market decision at once; a 6-layer *corrector* then lets all
  decisions see each other's soft draft and revise. Movement is not predicted: a
  rule takes the shortest-path step to the chosen tile.
- **Pretrain on many top teams → fine-tune on one → weight soup** of two
  fine-tunes from the same pretrain.
- **Models are chosen by money on paired seeds**, never by validation metrics.

Trained weights are not included. `pipeline/README.md` reproduces the final
submission from replays.

## Layout

| path | what |
|---|---|
| `experiments/net/bc_train.py` | corpus labeling (engine replay, executed market), feature cache, losses, training |
| `experiments/net/model.py`, `backbone.py`, `value_head.py`, `counts.py`, `effects.py` | the network: per-type encoders, trunk with relative-position bias, draft heads, corrector, value tower (training only) |
| `experiments/net/features.py`, `extract_features.py`, `masks.py`, `envcfg.py` | input encoding, legality masks, environment configuration |
| `experiments/net/policy.py` | turn assembly from the network output: masks before sampling, budget caps, `log π`; rule-based extras (`end_sell`, `night_room`, feed guard) |
| `experiments/net/play.py` | the agent (`agent(obs)`), local matches, parallel evaluation with `--dump` |
| `tools/soup.py` | uniform weight soup |
| `tools/submission/` | `build.py` packs a checkpoint into a self-contained Kaggle package; `validate.py` runs it the way Kaggle does |
| `tools/fetch/` | replay download by submission id |
| `tools/bench/` | `pair.py` (paired-seed comparison), `select_epoch.py` (pick an epoch by money) |
| `pipeline/` | teacher lists and the exact commands for the final models |
| `experiments/rl/` | PPO loop on top of the BC network (rollouts through the same `play.agent`, GAE, league, critic). **Not used by the final submissions**; an earlier PPO run gave +$1.3k/game over its own BC start. |
| `engines/kaggriculture_rs/` | Rust port of the environment with Python bindings, byte-for-byte parity with the Python engine; used for RL rollouts. Apache-2.0 (derived from `kaggle-environments`). |

## Setup

```bash
python -m venv env && . env/bin/activate     # Python 3.12
pip install -r requirements.txt
pip install --no-deps kaggle-environments==1.32.7
```

`--no-deps` because `kaggle-environments` pulls in packages that do not build on
3.12 and are not needed here. The Rust port is optional:
`cd engines/kaggriculture_rs && maturin develop --release`.

Quick check that the agent plays (needs a checkpoint):

```bash
python experiments/net/play.py --ckpt S_DEEP.pt --end-sell --night-room \
  --night-room-mode value -n 4 -j 4 --opp starter
```

## Notes

- The environment is deterministic given the seed, so two agents are compared by
  playing both on the same seeds and taking the per-seed difference
  (`play.py --dump` → `tools/bench/pair.py`).
- Anything fed into the network must be computable at decision time, and training
  and play must compute the same quantity (e.g. the shed at market time). Most of
  our worst bugs were of that kind; `bc_train.py` and `play.py` share the code
  for every such quantity.
- `secrets/` is git-ignored. Replay listing needs your own Kaggle browser cookies
  in `secrets/kaggle_cookie.json` (see `tools/fetch/fetch_replays.py`).

## License

[CC BY 4.0](LICENSE), as required by the competition rules for winning
submissions. `engines/kaggriculture_rs/` is Apache-2.0 (see its `LICENSE` and
`NOTICE`); it contains and ports code from
[Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments).
