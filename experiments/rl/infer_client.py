"""Rollout collection through the inference server: J worker processes, one GPU.

A worker plays games with the UNMODIFIED `collect.collect_serial` — the same
function as serial collection and the process-pool worker (`collect_worker.py`).
The only difference: it is handed an `infer_server.RemoteNet` instead of weights.
So "through the server" cannot become a second way to play a game: there is one
way, only the object asked for logits differs.

What the worker NO LONGER NEEDS (besides speed, the main simplification):
weights. `collect_parallel` writes `state_dict` to a temp file and each of the J
processes reads it; here the weights live in one place — on the server's GPU.
The worker only needs the teacher log-frequencies (`ck["prior"]`), and for those
it would read the whole 40 MB checkpoint — so `collect_served` extracts them
up front into a separate few-KB file and points `cfg.ckpt` at it.

⚠ Processes are started via `spawn` (otherwise the server would inherit a foreign
CUDA context), and `spawn` RE-imports the caller's main module. So any script
that calls `collect_served` must guard its entry point with
`if __name__ == "__main__":` — otherwise every worker process starts by
launching yet another server, and crashes. This is a `multiprocessing`
requirement, not ours, and cannot be bypassed: `fork` is ruled out by the GPU.

⚠ Bitwise equality with serial collection is expected ONLY with one worker and
no graph: then batch size is one and the same kernels run. This was verified by
an internal parity script. With J > 1 the batch is larger than one, and a
last-digit mismatch is unavoidable — its size and whether it reaches the action
were measured by a separate batch-parity script.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))


def _worker(slot, spec, arrays_in, arrays_out, flags, req_sem, rep_sem, slots,
            task_path, mids=None, queue=None):
    """Worker process entry point. Contains not a single line of game logic."""
    import torch
    torch.set_num_threads(1)     # same parity requirement as in collect_worker
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "experiments/net"))
    import collect as COL
    import infer_server as S

    with open(task_path, "rb") as f:
        task = pickle.load(f)
    if task.get("affinity"):
        os.sched_setaffinity(0, task["affinity"])
    if task.get("nice"):
        os.nice(int(task["nice"]))

    def remote(mid):
        return S.RemoteNet(slot, spec, arrays_in, arrays_out, flags, req_sem,
                           rep_sem, slots, mid=mid, mids=mids)

    # Checkpoint opponents also go to the GPU: the parent computed their model
    # ids from the plan and put them in the task. Anything not in the registry
    # (a package with its own code, a checkpoint outside the plan) is loaded as
    # before, with its own weights.
    reg = task.get("registry") or {}
    if reg:
        COL.NET_FOR_OPPONENT = (
            lambda path, dev, _r=reg: remote(_r[path]) if path in _r
            else COL.load_net(path, dev))
    if task.get("opp_priors"):
        COL.PRIOR_PATH_FOR = dict(task["opp_priors"])
    net = remote(0)
    prof = task.get("profile")
    if prof and slot == 0:
        # Profile of a LIVE worker, not of a model of it: the Python floor has
        # shifted three times already (env port, dropped observation copy,
        # server), and each time stale shares misled us. Only slot zero is
        # profiled — the rest run normally and give it real contention for the GPU.
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        eps = COL.collect_serial(net, task["plan"], task["cfg"], None,
                                 progress=False, offset=task["offset"],
                                 tag="__PROG__ ",
                                 eval_mode=bool(task.get("eval")))
        pr.disable()
        pstats.Stats(pr).dump_stats(prof)
    elif queue is not None:
        # Queue (`rollout.serve_queue`): every worker holds the full plan, the
        # next game index is a shared counter. A game does not depend on WHO
        # plays it (seed and side come from the global index, the `FlowTracker`
        # stream is reset per game), so only the assignment to workers changes,
        # not the experience.
        eps = []
        while True:
            with queue.get_lock():
                gi = queue.value
                queue.value += 1
            if gi >= len(task["plan"]):
                break
            eps += COL.collect_serial(net, task["plan"][gi:gi + 1], task["cfg"],
                                      None, progress=False,
                                      offset=task["offset"] + gi,
                                      tag="__PROG__ ",
                                 eval_mode=bool(task.get("eval")))
    else:
        eps = COL.collect_serial(net, task["plan"], task["cfg"], None,
                                 progress=False, offset=task["offset"],
                                 tag="__PROG__ ",
                                 eval_mode=bool(task.get("eval")))
    with open(task["out"], "wb") as f:
        pickle.dump(eps, f, protocol=pickle.HIGHEST_PROTOCOL)


def pick_cpus(n: int) -> List[int]:
    """`n` logical cores from those available, one per physical core where possible.

    The CFS quota counts core-time, not cores: on 28 visible logical cores, 42
    workers burn 15 core-seconds of a period before it ends, and the whole
    container freezes. Pinning to `n` cores makes the quota unreachable and the
    time sharing fine-grained and fair.
    """
    allowed = sorted(os.sched_getaffinity(0))
    seen, first, rest = set(), [], []
    for c in allowed:
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/"
                       "thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        (rest if sib in seen else first).append(c)
        seen.add(sib)
    return (first + rest)[:max(1, n)]


def slim_prior(ckpt: str, out_dir: str) -> str:
    """Extract ONLY `prior` from a checkpoint into a small separate file. -> path.

    Plain caching: `load_prior` reads `sd["prior"]`, and with it `torch.load`
    loads all 40 MB of weights — in each of the J processes. The file below is
    a few KB and loads instantly. If the checkpoint has none, return the
    original path and let `load_prior` deal with it.
    """
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    pr = sd.get("prior") if isinstance(sd, dict) else None
    if not pr:
        return ckpt
    import hashlib
    p = os.path.join(out_dir, "prior_" +
                     hashlib.sha1(os.path.abspath(ckpt).encode()).hexdigest()[:10] + ".pt")
    torch.save({"prior": pr}, p)
    return p


def collect_served(net, plan, cfg, workers: int = 0, progress: bool = True,
                   server_device: str = "cuda", graph: bool = True,
                   offset: int = 0, serve_opponents: bool = True,
                   profile: str = "", min_batch: int = 1,
                   max_wait: float = 0.002, eval_mode: bool = False) -> List:
    """Rollout through the inference server. -> experience streams in plan order.

    `net` is needed here only to (a) probe the slot shape and (b) hand the
    weights to the server as a file. It plays no games itself.

    ⚠ The parent does NOT touch CUDA: only the server process initializes the GPU.
    Otherwise `fork`/`spawn` would inherit a foreign context.
    """
    import multiprocessing as mp
    import numpy as np
    import torch
    import play

    import collect as C
    import infer_server as S

    n = len(plan)
    if n == 0:
        return []
    quota = play.cpu_quota()
    # The server also wants a core: it copies input and output and syncs with
    # the GPU. Giving it one of the J would short the workers.
    J = max(1, min(int(workers or (quota - 1)), n))
    # ⚠ Asking for MORE workers than cores is allowed here, not a typo.
    # Measured 23.09 (profile of a live worker): a worker spends 49% of its time
    # WAITING for the server — 11.98 s of 24.37. A waiting process holds no
    # core, so beyond the quota they don't compete, they fill each other's
    # waits. The process pool (`collect_parallel`) can't do this: there the
    # worker computes the net itself and occupies a core all the time.
    if J > quota - 1:
        print(f"workers {J} with a core quota of {quota}: over quota on purpose, "
              f"a worker waits on the server half the time", flush=True)

    t_prep = time.time()
    torch.set_num_threads(1)
    spec = S.probe(net, cfg)
    # ⚠ `spawn`, not `fork`: the server process initializes CUDA, and a foreign
    # GPU context must not be inherited. This also avoids the usual
    # "fork after torch started its threads" issue.
    ctx = _context(cfg)
    # Input — one array per tensor (a flat one broke bitwise parity, see
    # `infer_server.serve`), output — ONE flat array: one bus transfer
    # instead of thirteen.
    arrays_in = S.alloc(ctx, spec["in_spec"], J)
    # Output — one flat array per slot (`infer_server.out_layout`): one
    # bus transfer instead of thirteen.
    _, ototal = S.out_layout(spec["out_spec"])
    arrays_out = ctx.RawArray("f", J * ototal)
    # Empty slots are filled by the server with the probe observation (`serve`).
    flags = ctx.RawArray("b", J)
    # Model id per slot: the worker writes it next to the input, the server
    # uses it to split arrivals into groups (the weights differ).
    mids = ctx.RawArray("i", J)
    req_sem = ctx.Semaphore(0)
    rep_sems = [ctx.Semaphore(0) for _ in range(J)]
    stop = ctx.Value("i", 0)
    stats = ctx.RawArray("d", 7)

    tmp = tempfile.mkdtemp(prefix="rl_served_")
    procs = []
    srv = None
    try:
        wpath = os.path.join(tmp, "weights.pt")
        torch.save(net.state_dict(), wpath)
        ck_slim = slim_prior(cfg.ckpt, tmp) if cfg.ckpt else cfg.ckpt

        # The model registry is built FROM THE PLAN: 0 is the trained policy,
        # then each distinct checkpoint opponent. Packages (`kind == "package"`)
        # are not and cannot be included — they run their own code with their
        # own modules, and there is no net object to swap for them.
        weights = [wpath]
        registry: Dict[str, int] = {}
        if serve_opponents:
            for m in plan:
                if m.kind == "ckpt" and m.path and m.path not in registry:
                    registry[m.path] = len(weights)
                    weights.append(m.path)
        # Checkpoint opponents' `prior` — also as small files (review 23.09,
        # finding 10): the worker only needs `prior` from their checkpoint, the
        # weights are on the server, yet it read the whole 22-40 MB file — in
        # every process.
        opp_priors = {path: slim_prior(path, tmp) for path in registry}

        rc = cfg.rollout
        srv_cpus, wrk_cpus = None, None
        if getattr(rc, "serve_pin", False):
            cpus = pick_cpus(quota)
            srv_cpus, wrk_cpus = cpus[:1], (cpus[1:] or cpus)
            if progress:
                print(f"pinning: server -> {srv_cpus}, workers -> {wrk_cpus}",
                      flush=True)
        srv = ctx.Process(target=S.serve,
                          args=(weights, spec, J, server_device, graph, arrays_in,
                                arrays_out, flags, req_sem, rep_sems, stop, stats,
                                mids, min_batch, max_wait),
                          kwargs={"affinity": srv_cpus,
                                  "pinned": bool(getattr(rc, "serve_pinned", False)),
                                  "block_sync": bool(getattr(rc, "serve_block_sync",
                                                             False)),
                                  "streams": bool(getattr(rc, "serve_streams",
                                                          False))},
                          daemon=True)
        srv.start()

        # Plan chunks — same blocks as in `collect_parallel`: the game index
        # sets both the seed and the side parity, so a chunk must know its
        # place in the plan.
        # Queue instead of chunks: collection wall time is the slowest worker,
        # and with fixed chunks one long game (a net package, a long script
        # tail) holds everyone up. With a queue each worker takes the next game
        # when free; stream order is restored by index.
        use_q = bool(getattr(cfg.rollout, "serve_queue", False))
        qctr = ctx.Value("i", 0) if use_q else None
        blocks = [(0, n)] * J if use_q else C._blocks(n, J)
        outs = []
        for bi, (off, cnt) in enumerate(blocks):
            wcfg = _clone_cfg(cfg, ck_slim)
            res = os.path.join(tmp, f"res{bi}.pkl")
            task = os.path.join(tmp, f"task{bi}.pkl")
            with open(task, "wb") as f:
                pickle.dump({"cfg": wcfg, "plan": list(plan[off:off + cnt]),
                             "offset": offset + off, "out": res,
                             "registry": registry, "profile": profile,
                             "opp_priors": opp_priors,
                             "affinity": wrk_cpus, "eval": bool(eval_mode),
                             "nice": int(getattr(rc, "worker_nice", 0))}, f)
            outs.append(res)
            p = ctx.Process(target=_worker,
                            args=(bi, spec, arrays_in, arrays_out, flags,
                                  req_sem, rep_sems[bi], J, task, mids, qctr),
                            daemon=True)
            p.start()
            procs.append(p)
        if progress:
            print(f"server up: workers {len(procs)}, batch {J}, "
                  f"device {server_device}, graph {'yes' if graph else 'no'}, "
                  f"models {len(weights)} (opponents {len(registry)}), "
                  f"setup {time.time() - t_prep:.1f} s", flush=True)

        t0 = time.time()
        done = 0
        for p in procs:
            p.join()
            done += 1
            if progress:
                print(f"__PROG__ workers done {done}/{len(procs)}, "
                      f"{time.time() - t0:.0f} s", flush=True)
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"worker crashed, exit code {p.exitcode}")

        # `rollout.load_nogc`: the garbage collector is off while reading. The
        # result is ~2.4 GB for 60k turns, millions of objects (`enc`/`lat`
        # arrays, dicts, `Turn`), and the cyclic collector walks the whole
        # growing generation at every threshold: 5.6-10.3 s vs 3.3 s without
        # it (24.09). These objects have no cycles, so there is nothing to collect.
        import gc
        nogc = bool(getattr(cfg.rollout, "load_nogc", False)) and gc.isenabled()
        if nogc:
            gc.disable()
        try:
            eps = []
            for r in outs:
                with open(r, "rb") as f:
                    eps += pickle.load(f)
        finally:
            if nogc:
                gc.enable()
        if use_q:
            # same order as with chunks: by game index, within a game — as
            # `play_episode` returned it (with mirroring, both sides in a row)
            # seed = seed0 + game index; in group collection the env seed is
            # shared by 2K games, and order is kept by `index` (global index)
            eps.sort(key=lambda e: (e.index if e.index >= 0 else e.seed))
    finally:
        stop.value = 1
        req_sem.release()
        for p in procs:
            if p.is_alive():
                p.terminate()
        if srv is not None:
            srv.join(timeout=15)
            if srv.is_alive():
                srv.terminate()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    collect_served.last_stats = {
        "batches": int(stats[0]), "states": int(stats[1]),
        "s in pass": stats[2], "s waiting": stats[3],
        "s input": stats[4], "s net": stats[5], "s output": stats[6],
        "mean batch": stats[1] / max(stats[0], 1),
        "workers": J}
    if progress:
        st = collect_served.last_stats
        print(f"server: batches {st['batches']}, states {st['states']}, "
              f"mean batch {st['mean batch']:.1f}, "
              f"in pass {st['s in pass']:.1f} s "
              f"(input {st['s input']:.1f} + net {st['s net']:.1f} + "
              f"output {st['s output']:.1f}), "
              f"waiting {st['s waiting']:.1f} s", flush=True)
    return eps


_PRELOAD = ["numpy", "torch", "collect", "infer_server", "play", "policy",
            "features", "model", "collect_lockstep", "kaggle_environments"]


def _context(cfg):
    """How to start the server and workers. -> `multiprocessing` context.

    Default is `spawn` (as before): every collection process re-imports torch,
    the game and the env. rl1 timeline (24.09): ~15 s from collection start to
    the server's first pass, the GPU idle all that time; a worker's import
    costs ~1.7-2.7 s of CPU, and there are 42 of them on 14 cores (plus 24 for
    evaluation, running alongside the update — which is starved of cores then).

    `rollout.serve_forkserver`: processes are forked from a parent server that
    imported all of this ONCE per run (`_PRELOAD`). It does not touch the GPU
    (torch without CUDA), so forking from it is fine even for the inference
    server — that one creates its CUDA context itself, after the fork. ⚠
    Consequence: the code of `_PRELOAD` modules is frozen at the run's first
    collection; syncing new code under a live run does not change it (under
    `spawn` it did — and mixed versions).
    """
    import multiprocessing as mp
    if not getattr(cfg.rollout, "serve_forkserver", False):
        return mp.get_context("spawn")
    ctx = mp.get_context("forkserver")
    if not getattr(_context, "preloaded", False):
        ctx.set_forkserver_preload(_PRELOAD)
        _context.preloaded = True
    return ctx


def _clone_cfg(cfg, ckpt: str):
    """Copy of the config for a worker with `cfg.ckpt` replaced.

    It is replaced by the log-frequencies-only file: that is ALL the worker
    needs from the checkpoint (`play_episode` reads `cfg.ckpt` for
    `load_prior` and nothing else), and it has no weights at all now — they
    are on the server's GPU.
    """
    import copy as _copy
    w = _copy.deepcopy(cfg)
    w.ckpt = ckpt
    # ⚠ The worker builds tensors on CPU and hands them to the server as is:
    # `RemoteNet` writes `tensor.numpy()` to shared memory, and a GPU tensor
    # has no such method. There is exactly one GPU in this scheme, owned by the server.
    w.rollout.device = "cpu"
    # No pool inside a worker: we already distributed the processes.
    w.rollout.jobs = 1
    return w
