"""Download episode replays from Kaggle by submissionId.

The public API token is NOT enough: this needs the undocumented RPC
`competitions.EpisodeService/ListEpisodes` with browser cookies. Export your own
kaggle.com cookies as JSON to secrets/kaggle_cookie.json and never commit that file.
The replay itself lives on the public kaggleusercontent.com and downloads without auth.

ListEpisodes is NOT limited to your own submissionId -- it works for other teams too,
and returns at most the 1000 latest episodes per submission.

⚠ Do not download your own submissions for TRAINING: the model would close on its own mistakes.
If you download your own episode for analysis, keep it outside raw-content/replays/, otherwise
any glob over that folder will pick it up.

Usage:
    ./env/bin/python tools/fetch/fetch_replays.py 56263693            # list + replays
    ./env/bin/python tools/fetch/fetch_replays.py 56254996 --list-only  # list only
    python3 tools/fetch/fetch_replays.py 56254996 --from-list          # download only
Result: raw-content/replays/sub_<id>/<episode_id>.json + episodes.json with metadata.
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COOKIE_PATH = os.path.join(ROOT, "secrets", "kaggle_cookie.json")
OUT_ROOT = os.path.join(ROOT, "raw-content", "replays")
LIST_EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggleusercontent.com/episodes/{id}.json"
WORKERS = 16


def load_session():
    cookies = {c["name"]: c["value"] for c in json.load(open(COOKIE_PATH))}
    s = requests.Session()
    for k, v in cookies.items():
        s.cookies.set(k, v, domain=".kaggle.com")
    s.headers.update({
        "Content-Type": "application/json",
        "X-XSRF-TOKEN": cookies.get("XSRF-TOKEN", ""),
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.kaggle.com",
    })
    return s


def list_episodes(sess, submission_id, retries=6):
    delay = 2.0
    for attempt in range(retries):
        r = sess.post(LIST_EPISODES_URL, json={"submissionId": int(submission_id)})
        if r.status_code == 429:
            time.sleep(delay); delay *= 2; continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"ListEpisodes: rate limited for {submission_id}")


def fetch_one(episode_id, out_dir):
    path = os.path.join(out_dir, f"{episode_id}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return "cached"
    try:
        req = urllib.request.Request(REPLAY_URL.format(id=episode_id),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        return "ok"
    except Exception as e:
        return f"err {e}"


def download(out_dir, eps):
    """Download the replays themselves. No auth is needed here: the json lies on the public
    kaggleusercontent. This splits the work: the episode list is fetched with
    browser cookies (only where the cookies are), while the 30 MB per episode can be
    downloaded wherever they will be stored. Otherwise they would have to be
    uploaded over a 0.7 MB/s channel, i.e. a day per teacher."""
    stats = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(lambda e: fetch_one(e["id"], out_dir), eps), 1):
            k = r if r in ("ok", "cached") else "err"
            stats[k] = stats.get(k, 0) + 1
            if i % 25 == 0:
                print(f"  {i}/{len(eps)} episodes, {i/(time.time()-t0):.1f}/s "
                      f"{stats}", flush=True)
    print(f"  {stats}", flush=True)
    return stats


def pull(sess, submission_id, limit=None):
    out_dir = os.path.join(OUT_ROOT, f"sub_{submission_id}")
    os.makedirs(out_dir, exist_ok=True)
    resp = list_episodes(sess, submission_id)
    eps = []
    for ep in resp.get("episodes", []):
        if ep.get("state") != "COMPLETED":
            continue
        rec = {"id": ep["id"], "createTime": ep.get("createTime"),
               "agents": [{"submissionId": a.get("submissionId"),
                           "index": a.get("index"),
                           "reward": a.get("reward"),
                           "teamName": (a.get("submission") or {}).get("teamName")}
                          for a in ep.get("agents", [])]}
        eps.append(rec)
    if limit:
        eps = eps[:limit]
    json.dump(eps, open(os.path.join(out_dir, "episodes.json"), "w"))
    print(f"sub {submission_id}: {len(eps)} finished episodes", flush=True)
    return len(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_ids", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=None, help="episodes per submission")
    ap.add_argument("--list-only", action="store_true",
                    help="only the episode list into episodes.json (cookies needed)")
    ap.add_argument("--from-list", action="store_true",
                    help="download from an existing episodes.json (cookies NOT needed)")
    args = ap.parse_args()
    if not args.submission_ids:
        ap.error("give at least one submissionId")

    if args.from_list:
        total = 0
        for sid in args.submission_ids:
            out_dir = os.path.join(OUT_ROOT, f"sub_{sid}")
            eps = json.load(open(os.path.join(out_dir, "episodes.json")))
            if args.limit:
                eps = eps[:args.limit]
            print(f"sub {sid}: {len(eps)} episodes in the list, downloading...", flush=True)
            download(out_dir, eps)
            total += len(eps)
        print(f"total episodes: {total}")
        return

    sess = load_session()
    total = 0
    for sid in args.submission_ids:
        try:
            n = pull(sess, sid, args.limit)
            total += n
            if not args.list_only:
                out_dir = os.path.join(OUT_ROOT, f"sub_{sid}")
                download(out_dir, json.load(open(os.path.join(out_dir,
                                                              "episodes.json"))))
        except Exception as e:
            print(f"sub {sid}: ERROR {e}")
    print(f"total episodes: {total}")


if __name__ == "__main__":
    main()
