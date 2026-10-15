"""Submission strength from episode metadata -- without downloading the games.

`episodes.json` contains the rewards of BOTH sides, so the bank median and win
rate are computed without a single gigabyte. Replays are worth downloading only for
submissions that are actually stronger than the current teacher.

Run: ./env/bin/python tools/fetch/rank_subs.py 56216119 56352346 ...
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fetch_replays import load_session, list_episodes  # noqa: E402

TEACHER = 56263693        # current teacher, as the reference point


def stats(sess, sid):
    resp = list_episodes(sess, sid)
    mine, opp, wins, n = [], [], 0, 0
    opp_ids = set()
    for ep in resp.get("episodes", []):
        if ep.get("state") != "COMPLETED":
            continue
        ags = ep.get("agents", [])
        if len(ags) != 2:
            continue
        me = next((a for a in ags if a.get("submissionId") == sid), None)
        ot = next((a for a in ags if a.get("submissionId") != sid), None)
        if me is None or ot is None or me.get("reward") is None or ot.get("reward") is None:
            continue
        mine.append(me["reward"]); opp.append(ot["reward"])
        wins += int(me["reward"] > ot["reward"]); n += 1
        opp_ids.add(ot.get("submissionId"))
    if not n:
        return None
    return dict(n=n, med=float(np.median(mine)), p10=float(np.percentile(mine, 10)),
                p90=float(np.percentile(mine, 90)), win=wins / n,
                opp_med=float(np.median(opp)), opps=len(opp_ids),
                team=(next((a.get("submission") or {} for a in
                            resp["episodes"][0].get("agents", [])
                            if a.get("submissionId") == sid), {}) or {}).get("teamName"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_ids", nargs="+", type=int)
    a = ap.parse_args()
    sess = load_session()
    ids = list(dict.fromkeys(a.submission_ids + [TEACHER]))
    rows = []
    for sid in ids:
        try:
            s = stats(sess, sid)
        except Exception as e:
            print(f"{sid}: ERROR {e}", flush=True)
            continue
        if s:
            rows.append((sid, s))
            print(f"  {sid} collected ({s['n']} episodes)", flush=True)
    rows.sort(key=lambda r: -r[1]["med"])
    print(f"\n{'submission':>10}{'median':>11}{'p10':>10}{'p90':>10}"
          f"{'wins':>8}{'games':>8}{'opp.':>8}  team")
    for sid, s in rows:
        mark = "  <- current teacher" if sid == TEACHER else ""
        print(f"{sid:>10}{s['med']:>11,.0f}{s['p10']:>10,.0f}{s['p90']:>10,.0f}"
              f"{100*s['win']:>7.0f}%{s['n']:>8}{s['opps']:>8}  "
              f"{(s['team'] or '')[:22]}{mark}")
    if rows:
        t = next((s for i, s in rows if i == TEACHER), None)
        if t:
            print(f"\ncurrent teacher: median ${t['med']:,.0f}, wins {100*t['win']:.0f}%")
            best = rows[0]
            if best[0] != TEACHER:
                print(f"best of the list: {best[0]}, +{100*(best[1]['med']/t['med']-1):.1f}% "
                      f"by median, wins {100*best[1]['win']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
