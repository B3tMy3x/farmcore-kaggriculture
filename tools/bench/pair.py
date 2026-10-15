"""Paired comparison of two `play.py --dump` dumps by seed.

Medians lie: with a 3x spread and n=48, a median comparison resolves almost nothing.
The seeds are the same across runs and the measurement is deterministic, so
the per-seed difference is what should be compared.

Also computes the measurement's resolving power: the minimum shift that
such an n can distinguish from zero at 80% power. At n=48 that is ~$11k --
so all our earlier conclusions at the level of "-$580" and "-$7.4k" were indistinguishable.

    ./env/bin/python tools/bench/pair.py games_A.json games_B.json

With ONE file it computes a head-to-head: in the `--opp-ckpt` dump both sides played
each other, and the bank difference within a game is the sharpest available
statistic (one market, one seed, only the version differs).

    ./env/bin/python tools/bench/pair.py games_r1_vs_r0.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats


def load(path):
    g = np.array(json.load(open(path))["games"], dtype=float)
    return g


def head2head(path):
    """Head-to-head: one dump where the sides played EACH OTHER.

    This is the sharpest available comparison: both versions are on the same market
    in the same game, so the per-seed bank difference contains neither seed
    spread nor opponent spread -- only the version difference. Plus wins, which
    the leaderboard counts.
    """
    g = np.array(json.load(open(path))["games"], dtype=float)
    ok = g[:, 2] == 0
    dead = int((~ok).sum())
    g = g[ok]
    n = len(g)
    d = g[:, 0] - g[:, 1]
    w = int((d > 0).sum())
    print(f"{path}  (head-to-head)")
    if dead:
        print(f"  ⚠ dropped from the game: {dead}, excluded from the comparison")
    print(f"  n = {n}")
    print(f"  median: us ${np.median(g[:,0]):,.0f}  opponent ${np.median(g[:,1]):,.0f}")
    print(f"  p10:     us ${np.percentile(g[:,0],10):,.0f}  opponent "
          f"${np.percentile(g[:,1],10):,.0f}")
    print(f"  paired within game: median difference ${np.median(d):+,.0f}, "
          f"mean ${d.mean():+,.0f}")
    print(f"  wins {w}/{n}, sign test p = {stats.binomtest(w, n, 0.5).pvalue:.4g}; "
          f"wilcoxon p = {stats.wilcoxon(d).pvalue:.4g}")
    mde = 2.8 * d.std() / np.sqrt(n)
    print(f"  resolving power: sd of difference ${d.std():,.0f}, "
          f"minimum detectable shift ~${mde:,.0f} (80% power at n={n})")


def main():
    if len(sys.argv) == 2:
        return head2head(sys.argv[1])
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    A, B = load(sys.argv[1]), load(sys.argv[2])
    if len(A) != len(B):
        raise SystemExit(f"different number of games: {len(A)} and {len(B)}")
    ok = (A[:, 2] == 0) & (B[:, 2] == 0)
    dead = int((~ok).sum())
    A, B = A[ok], B[ok]
    n = len(A)
    d = B[:, 0] - A[:, 0]
    w = int((d > 0).sum())
    print(f"{sys.argv[1]}  ->  {sys.argv[2]}")
    if dead:
        print(f"  ⚠ dropped from the game: {dead}, excluded from the comparison")
    print(f"  n = {n}")
    print(f"  median: ${np.median(A[:,0]):,.0f}  ->  ${np.median(B[:,0]):,.0f}")
    print(f"  p10:     ${np.percentile(A[:,0],10):,.0f}  ->  "
          f"${np.percentile(B[:,0],10):,.0f}")
    print(f"  paired: median difference ${np.median(d):+,.0f}, "
          f"mean ${d.mean():+,.0f}")
    print(f"  sign test {w}/{n}, p = {stats.binomtest(w, n, 0.5).pvalue:.4g}; "
          f"wilcoxon p = {stats.wilcoxon(d).pvalue:.4g}")
    wa = int((A[:, 0] > A[:, 1]).sum())
    wb = int((B[:, 0] > B[:, 1]).sum())
    print(f"  wins in game: {wa}/{n} -> {wb}/{n}  "
          f"(against {'the same' if np.allclose(A[:,1], B[:,1]) else 'a different'} opponent)")
    mde = 2.8 * d.std() / np.sqrt(n)
    print(f"  resolving power: sd of difference ${d.std():,.0f}, "
          f"minimum detectable shift ~${mde:,.0f} (80% power at n={n})")


if __name__ == "__main__":
    main()
