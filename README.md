# Farmcore — Kaggriculture solution

Code of team **Farmcore** for the Kaggle simulation competition
[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture). Writeup:
[Farmcore: a one-pass transformer trained on what the engine executed](https://www.kaggle.com/competitions/kaggriculture/writeups/farmcore-a-one-pass-transformer-trained-on-what-t) (4th place).

The final agent is pure behaviour cloning, with no self-play and no search at play time:

- **Labels are what the engine executed**, not what the player ordered. Every corpus game is
  replayed through the engine, and only the executed state change is kept.
- **One forward pass per turn.** A transformer drafts every unit's target and action and every
  market decision at once; a 6-layer corrector lets these decisions see each other and revise.
  Movement is a shortest-path rule toward the chosen tile.
- **Pretrain on many top teams, fine-tune on one, average two fine-tunes** (weight soup).
- **Models are selected by money on paired seeds**, not by validation metrics.

The trained agent (the final submission package, code + weights) is on
[Kaggle Models](https://www.kaggle.com/models/betmyex/farmcore-kaggriculture-agent);
`pipeline/README.md` reproduces it from replays.

## Layout

| path | what |
|---|---|
| `experiments/net/bc_train.py` | corpus labeling through the engine, feature cache, losses, training |
| `experiments/net/model.py`, `backbone.py`, `value_head.py`, `counts.py`, `effects.py` | the network |
| `experiments/net/features.py`, `extract_features.py`, `masks.py`, `envcfg.py` | input encoding, legality masks, environment config |
| `experiments/net/policy.py` | turn assembly from the network output, plus the rules (`end_sell`, `night_room`, feed guard) |
| `experiments/net/play.py` | the agent `agent(obs)`, local matches, parallel evaluation |
| `tools/soup.py` | weight soup |
| `tools/submission/` | `build.py` packs a checkpoint into a Kaggle package, `validate.py` runs it the way Kaggle does |
| `tools/fetch/` | replay download |
| `tools/bench/` | `pair.py` paired-seed comparison, `select_epoch.py` epoch selection by money |
| `pipeline/` | teacher lists and commands for the final models |
| `experiments/rl/` | PPO loop on top of the BC network (not used by the final submission) |
| `engines/kaggriculture_rs/` | Rust port of the environment with Python bindings, bit-exact with the Python engine |

## Setup

```bash
python -m venv env && . env/bin/activate     # Python 3.12
pip install -r requirements.txt
pip install --no-deps kaggle-environments==1.32.7
```

The Rust port is optional: `cd engines/kaggriculture_rs && maturin develop --release`.

Play a few games (needs a checkpoint):

```bash
python experiments/net/play.py --ckpt S_DEEP.pt --end-sell --night-room \
  --night-room-mode value -n 4 -j 4 --opp starter
```

Compare two agents on the same seeds: `play.py --dump a.json`, `play.py --dump b.json`, then
`python tools/bench/pair.py a.json b.json`.

Downloading replay lists needs your own Kaggle browser cookies in `secrets/kaggle_cookie.json`
(git-ignored).

## License

[CC BY 4.0](LICENSE). `engines/kaggriculture_rs/` is Apache-2.0 (see its `LICENSE` and `NOTICE`),
as it ports code from [Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments).
