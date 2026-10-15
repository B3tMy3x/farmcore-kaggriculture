# Reproducing the final submissions

Final submission **56721212** (`S_DEEP`) is a uniform weight soup of two fine-tunes
that share one pretrain:

```
pre_deep67   (from scratch, 67 top-ladder submissions, 2 epochs)
 ├─ deep_mm   fine-tune on 4 M & M & P & Q submissions, 12 epochs → e7
 └─ deep_mm6  fine-tune on 6 M & M & P & Q submissions,  8 epochs → e7
S_DEEP = ½ deep_mm/e7 + ½ deep_mm6/e7      (tools/soup.py)
package: --end-sell --night-room --night-room-mode value
```

The commands below are reconstructed from the `cfg` stored in each checkpoint
(every flag that differs from the default is listed). Paths are relative to the
repo root; checkpoints land in `experiments/net/runs/<run>/e<N>.pt`.

## 0. Environment

```bash
python -m venv env && . env/bin/activate      # Python 3.12
pip install -r requirements.txt
pip install --no-deps kaggle-environments==1.32.7
```

## 1. Corpus

Teacher lists live in `pipeline/teachers/` (Kaggle submission ids):

| file | used by | what |
|---|---|---|
| `pretrain67.txt` | pretrain | 67 submissions of 23 top-ladder teams |
| `ft_mm4.txt` | `deep_mm` | 4 submissions of M & M & P & Q |
| `ft_mm6.txt` | `deep_mm6` | the same 4 + 2 later ones |

```bash
bash pipeline/01_fetch.sh pipeline/teachers/pretrain67.txt
```

Listing a submission's episodes uses an undocumented Kaggle RPC that needs your
browser cookies (see `tools/fetch/fetch_replays.py`); the replays themselves are
public. Kaggle keeps only the last ~1000 episodes per submission, so a corpus
fetched today differs from ours (we had 8 583 games for the pretrain, 563 for `deep_mm` and 766 for `deep_mm6`).

The first training run builds the label cache
(`raw-content/cache/<fingerprint>/<episode>.npz`): every game is replayed through
the engine with both sides' recorded moves, money is checked bit-exactly on every
step, and market labels are what the engine **executed**, not what was ordered
(`bc_train.executed_market`).

## 2. Pretrain

```bash
bash pipeline/02_pretrain.sh        # run pre_deep67, keep e0 (= epoch 1 of 2)
```

## 3. Fine-tunes

```bash
bash pipeline/03_finetune.sh        # runs deep_mm and deep_mm6, both from pre_deep67/e0
```

## 4. Soup, package, check

```bash
bash pipeline/04_soup_and_package.sh
```

`tools/soup.py` averages float tensors and takes `prior`/`cfg` from the first
parent; on our checkpoints it reproduces the submitted weights bit for bit.
`tools/submission/validate.py` runs the built package exactly as Kaggle does and
compares money with a local run of the package's own `weights.pt` (pass that,
not the raw soup: the decoder flags live in the package weights).

## Hardware

Pretrain and the 12-epoch fine-tune ran on one A100-80GB (`--gpu-corpus 1` keeps
the corpus on the GPU); `deep_mm6` ran on a single GPU
in free cloud compute. Play is CPU-only (2 threads, a few tens of ms per turn on a laptop; the 1 s limit is never close).

## Choosing epochs

Validation metrics did not predict play (Spearman −0.69 between validation and
money across epochs). Pick epochs by money on paired seeds:
`tools/bench/select_epoch.py`, and compare two candidates with
`experiments/net/play.py --dump` + `tools/bench/pair.py`.
