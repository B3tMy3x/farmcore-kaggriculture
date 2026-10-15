"""Inference server: one GPU serves J worker processes, batched and graph-captured.

Why processes, not threads. Lockstep on threads was built, verified and MEASURED
(`collect_lockstep.py`): batching works perfectly there — pass cost drops from
16.25 to 0.65 ms per decision — but the Python part of a turn grows from 6.4 to
25 ms, because every tensor op in the decoder releases and re-acquires the GIL,
with K threads ready. The gain is eaten to exactly zero. The Python floor is
paid on a CORE, so a worker must be a process.

Why a graph. A net pass costs ~12.4 ms of fixed overhead regardless of batch
(A100: batch 1 — 12.45 ms, batch 128 — 13.96 ms) — kernel launches and Python
inside `model.forward`, not arithmetic. While that overhead is there, a worker
WAITS those 12.4 ms instead of computing, and cross-process batching doesn't
pay off. A CUDA graph cuts it to 1.81 ms at batch 1 and 5.98 ms at batch 64.

Layout. Shared memory with J slots: one slot per worker, each holding the net
input and net output, all as flat fixed-shape arrays. A worker writes its input
into its slot, raises a flag, releases the shared semaphore and sleeps on its
own. The server wakes, takes ALL raised flags as one batch, computes, scatters
the output to slots and wakes each worker.

⚠ A graph requires a FIXED batch shape, so the batch always equals J, even when
fewer arrived. Extra slots are computed for nothing, but at these sizes a pass
is bound by kernel launches, not arithmetic (same table: from 1 to 128 the cost
grows 12%), so padding costs almost nothing, while branching on batch size and
J graphs in memory do cost. At startup slots are filled with ONE real
observation (from the shape probe) so unused slots never hold garbage: a fully
closed tile mask would give `nan` in softmax.

⚠ Only the TRAINED policy is served. A checkpoint opponent in
`collect.play_episode` loads its own copy of the weights and plays locally at
batch 1 — the server doesn't know about it. For mirrored self-play (the main
mode) this doesn't matter: both sides move via the same `play.agent` and both
go through the server.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))


# ------------------------------------------------------------- shared memory

_CTYPE = {"float32": ctypes.c_float, "float64": ctypes.c_double,
          "bool": ctypes.c_bool, "int64": ctypes.c_int64,
          "int32": ctypes.c_int32, "uint8": ctypes.c_ubyte}


def alloc(ctx, spec: List[Tuple[str, tuple, str]], slots: int):
    """One shared array per spec tensor. -> {name: RawArray}.

    `mp.RawArray`, not `shared_memory`: it can simply be passed as an argument
    to a child process, and `multiprocessing` manages its lifetime, not us. No
    lock on purpose — each slot is written by exactly one process, and the
    semaphores handle synchronization.
    """
    out = {}
    for name, shape, dt in spec:
        n = slots
        for d in shape:
            n *= d
        out[name] = ctx.RawArray(_CTYPE[dt], n)
    return out


def view(arrays, spec: List[Tuple[str, tuple, str]], slots: int):
    """numpy arrays over the shared ones. -> {name: ndarray of shape (slots, *shape)}."""
    import numpy as np
    return {name: np.frombuffer(arrays[name], dtype=np.dtype(dt)).reshape(
        (slots,) + tuple(shape)) for name, shape, dt in spec}


def in_layout(in_spec):
    """Flat input layout. -> (float32 offsets, total, same for bool).

    Same argument as for `out_layout`: we pay not for bytes but for ELEVEN
    transfers per batch (2.7 s of 18.9 s of server work with 14 workers).
    Two groups, not one, because part of the input is boolean (unit, tile and
    legality masks), and mixing dtypes in one array would mean fiddling with
    byte views on both sides.

    ⚠ Slicing back into tensors happens INSIDE the graph (`call`), so it costs
    no separate kernel launches: the graph has already captured them.
    """
    fo, fn, bo, bn = {}, 0, {}, 0
    for name, shape, dt in in_spec:
        k = 1
        for d in shape:
            k *= d
        if dt == "bool":
            bo[name] = (bn, k, tuple(shape)); bn += k
        elif dt == "float32":
            fo[name] = (fn, k, tuple(shape)); fn += k
        else:
            raise SystemExit(f"flat input expects float32/bool, but {name} is {dt}")
    return fo, fn, bo, bn


def out_layout(out_spec):
    """Flat output layout. -> (offsets, total numbers per slot).

    ⚠ Why flat. Measured 23.09: the server returned output via thirteen separate
    `cpu()` calls — one tensor per head — and that was 3.3 s of 18.9 s of its
    work, while the payload is only forty KB per state. We pay not for bytes
    but for THIRTEEN bus transfers per batch. Concatenating into one tensor
    inside the graph makes them one.

    All outputs are float32, so the layout is trivial: head after head, each
    flattened.
    """
    off, n = {}, 0
    for name, shape, dt in out_spec:
        if dt != "float32":
            raise SystemExit(f"flat output expects float32, but {name} is {dt}")
        k = 1
        for d in shape:
            k *= d
        off[name] = (n, k, tuple(shape))
        n += k
    return off, n


# ------------------------------------------------------------------- probe

class _Capture:
    """Stand-in "net" that records the call signature and calls the real one.

    The slot spec cannot be made up from constants: `play.agent` calls the net
    with six positional tensors and six keyword ones, and the set must match
    THAT VERY call, not our idea of it. So the shape is captured by a real turn
    of the real code.
    """

    def __init__(self, net):
        self.net = net
        self.args = None
        self.kwargs = None
        self.out = None

    def __call__(self, *args, **kwargs):
        out = self.net(*args, **kwargs)
        if self.args is None:
            self.args, self.kwargs, self.out = args, kwargs, out
        return out


def probe(net, cfg) -> Dict[str, Any]:
    """Play ONE turn and capture the slot spec. -> spec dict.

    The spec cannot be made up from constants: `play.agent` calls the net with
    six positional tensors and six keyword ones, and the set must match THAT
    VERY call, not our idea of it. So the shape is captured by a real turn of
    the real code.

    The observation comes from the Python env, not the Rust port, regardless of
    `rollout.env`: the net input shape doesn't depend on the env (observations
    verified bitwise equal), and creating one initial state with the Python env
    is one line with no dependency on the built wheel.

    Returns: `in_spec` (names, shapes without batch, dtype), `out_spec`,
    `n_args` (how many are positional), `kw_names`, `const` (non-tensor keyword
    args — their value must match across all workers), `seed_in` (contents of
    the real observation used to fill empty slots).
    """
    import torch
    import kaggle_environments as ke
    import policy as POL
    import collect as C

    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0},
                  debug=False)
    env.reset(2)
    obs = env.state[0].observation

    mod = C.fresh_play_module()
    cap = _Capture(net)
    # Log-frequencies are required: with β > 0 (the default) `policy.family`
    # loudly refuses to play without them, and the probe would fail for nothing.
    # ⚠ The probe runs on THE device holding the net. In the loop it is on the
    # GPU (`train_device=cuda`), and `play.agent` builds input on `_NET["dev"]`:
    # put "cpu" here and the probe fails at the first `linear` with "two devices,
    # cuda:0 and cpu". The shape doesn't depend on the device, and `seed_in`
    # below is taken on CPU anyway.
    where = str(next(net.parameters()).device)
    prior = C.load_prior(cfg.ckpt, where) if cfg.ckpt else None
    C.install(mod, cap, where, C.decode_mode(POL.PPO), record=False, prior=prior)
    mod._NET["m"] = cap
    mod._reset_flow()
    with torch.no_grad():
        mod.agent(obs, env.configuration)
    if cap.args is None:
        raise SystemExit("shape probe did not catch a net call")

    def spec_of(items):
        return [(name, tuple(t.shape[1:]), str(t.dtype).replace("torch.", ""))
                for name, t in items]

    arg_names = [f"a{i}" for i in range(len(cap.args))]
    kw_t = [(k, v) for k, v in cap.kwargs.items() if torch.is_tensor(v)]
    const = {k: v for k, v in cap.kwargs.items() if not torch.is_tensor(v)}
    in_spec = spec_of(list(zip(arg_names, cap.args))) + spec_of(kw_t)
    out_spec = spec_of(list(cap.out.items()))
    seed_in = {name: t.detach().cpu().numpy()[0].copy()
               for name, t in list(zip(arg_names, cap.args)) + kw_t}
    return {"in_spec": in_spec, "out_spec": out_spec, "n_args": len(cap.args),
            "kw_names": [k for k, _ in kw_t], "const": const, "seed_in": seed_in}


# ------------------------------------------------------------------ server

def serve(weights: List[str], spec: Dict[str, Any], slots: int, device: str,
          graph: bool, arrays_in, arrays_out, flags, req_sem, rep_sems,
          stop, stats, mids=None, min_batch: int = 1,
          max_wait: float = 0.002, affinity=None, block_sync: bool = False,
          pinned: bool = False, streams: bool = False) -> None:
    """Serving loop. Runs in its own process until `stop`.

    `stats` is a shared array of seven numbers: batches, states, seconds in
    pass, seconds waiting, and the pass split into input, net and output.
    Without it "the server didn't help" is indistinguishable from "the server
    helped but something else ate it", and without the split — "server bound by
    the GPU" from "server bound by copying".
    """
    import numpy as np
    import torch
    import collect as C

    torch.set_num_threads(1)
    if affinity:
        os.sched_setaffinity(0, affinity)
    dev = torch.device(device)
    # A blocking event instead of `torch.cuda.synchronize()`: whoever waits on
    # the GPU sleeps in the OS kernel instead of spinning on a CPU core. Same
    # numbers — only what the server does while the graph runs changes.
    ev = torch.cuda.Event(blocking=True) if (block_sync and device.startswith("cuda")) else None
    # Model registry: 0 is the trained policy, then league opponents. One GPU
    # serves all of them: otherwise an opponent would load its own weights in
    # the worker and move at batch 1 on a core, and that is HALF of a game's
    # decisions (measured 23.09: with a frozen opponent the server gave ×1.30
    # vs ×2.61 in mirrored self-play).
    if isinstance(weights, str):
        weights = [weights]
    nets = [C.load_net(w, device) for w in weights]
    net = nets[0]
    in_spec, out_spec = spec["in_spec"], spec["out_spec"]
    n_args = spec["n_args"]
    kw_names = spec["kw_names"]
    const = spec["const"]

    vin = view(arrays_in, in_spec, slots)
    # Output — ONE flat array [slots, total]: see `out_layout`.
    olay, ototal = out_layout(out_spec)
    vout = np.frombuffer(arrays_out, dtype=np.float32).reshape(slots, ototal)

    # ⚠ INPUT STAYS PER-TENSOR, and that is a measured decision, not laziness.
    # A flat input was built and rejected on 23.09: it did cut transfers
    # (2.7 s -> 0.3 s with 14 workers), but BROKE bitwise parity — the server
    # parity check gave 24 mismatches over 2 876 turns, all in `log π` and all
    # around 1e-6. The cause is not the values: a slice of the flat buffer is
    # non-contiguous, `reshape` rebuilds it as a copy, the copy has different
    # alignment, and cuBLAS picks a different kernel — i.e. a different
    # reduction order. Actions, latents, input, rewards and banks all matched.
    #
    # We didn't trade proven bitwise equality for 8% speed. It could be
    # restored with a tensor-after-tensor layout aligning each block to
    # 256 bytes — then `view` works without a copy and alignment is the same —
    # but that is separate work with its own verification.
    # `rollout.serve_streams`: EACH model gets its own input buffer, GPU stream
    # and pinned memory — so groups from one wakeup (the policy and league
    # opponents) go to the GPU concurrently, not one after another with a sync
    # after each. Without the flag there is one shared buffer, as before.
    streams = bool(streams and graph and pinned and device.startswith("cuda"))

    def _static():
        st = {name: torch.empty((slots,) + tuple(shape),
                                dtype=getattr(torch, dt), device=dev)
              for name, shape, dt in in_spec}
        for name, shape, dt in in_spec:
            st[name].copy_(torch.from_numpy(
                np.broadcast_to(spec["seed_in"][name], (slots,) + tuple(shape)).copy()))
        return st
    static = _static()
    statics = [static] + [(_static() if streams else static)
                          for _ in range(len(nets) - 1)]

    def call(k: int = 0, m: int = 0):
        """Pass and CONCATENATION of output into one tensor [k, total].

        Concatenating inside the graph, not after: then it becomes part of the
        graph and costs one kernel, and the bus carries a single transfer.
        """
        k = k or slots
        st = statics[m]
        a = [st[f"a{i}"][:k] for i in range(n_args)]
        kw = {k2: st[k2][:k] for k2 in kw_names}
        kw.update(const)
        o = nets[m](*a, **kw)
        return torch.cat([o[name].reshape(k, -1).float()
                          for name, _, _ in out_spec], dim=1)

    # Not one graph but a ladder of sizes is captured: the server takes the
    # smallest one that fits what arrived. A single graph for the full batch J
    # would always pay full price, and workers arrive staggered (with 8 workers
    # the mean batch is 2.1-2.4).
    #
    # ⚠ MEASURED, NO GAIN: 8 workers 558.6 decisions/s vs 564.8 with one graph,
    # 14 workers 828.9 vs 823.7 — run noise. The breakdown shows why: the
    # ladder did make a pass cheaper (net 2.15 ms per batch vs 2.7), but the
    # server then answered faster, the queue shortened and the mean batch
    # dropped (11 200 batches vs 9 739 for the same work). Gain per batch and
    # loss on batch count cancelled out.
    #
    # Kept on purpose anyway: the cost of ONE graph grows with J (1.81 ms at
    # batch 1, 5.98 at 64), and at J above twenty a fixed full batch would start
    # to dominate. The ladder insures against that and costs nothing — 0.5 s of
    # capture at startup and a shared input buffer.
    #
    # All graphs read from ONE `static` buffer, each with its own slice `[:k]`
    # — a graph records the pointer, and a slice of the shared buffer suits it.
    # So the ladder costs only its own output tensors, the input is shared.
    # Graphs per (model, size) PAIR. This costs little memory — the net has
    # 1.99M parameters, and the input buffer is shared by all graphs.
    graphs: List[List[Tuple[int, Any, Any]]] = [[] for _ in nets]
    cap_st = [torch.cuda.Stream() for _ in nets] if streams else None
    if graph:
        if not device.startswith("cuda"):
            raise SystemExit("CUDA graph requested on a non-GPU device")
        sizes, k = [], 1
        while k < slots:
            sizes.append(k)
            k *= 2
        sizes.append(slots)
        with torch.no_grad():
            st_ = torch.cuda.Stream()
            st_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st_):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(st_)
            t0 = time.time()
            for m in range(len(nets)):
                # ⚠ `serve_streams`: a model has its own stream, and warmup and
                # capture happen ON IT. The cuBLAS workspace is keyed by a
                # (handle, stream) pair; graphs captured on one stream share it,
                # and replaying them concurrently is a race: verification caught
                # outputs diverging by 14.8 (24.09).
                ws = cap_st[m] if streams else st_
                if m or streams:
                    # Each model gets its own warmup: without it capture picks up
                    # not-yet-warmed kernels and crashes or writes garbage.
                    ws.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(ws):
                        for _ in range(3):
                            call(m=m)
                    torch.cuda.current_stream().wait_stream(ws)
                for k in sizes:
                    gk = torch.cuda.CUDAGraph()
                    if streams:
                        with torch.cuda.graph(gk, stream=ws):
                            ok = call(k, m)
                    else:
                        with torch.cuda.graph(gk):
                            ok = call(k, m)
                    graphs[m].append((k, gk, ok))
            torch.cuda.synchronize()
        print(f"__SRV__ graphs captured in {time.time() - t0:.1f} s: models "
              f"{len(nets)}, sizes {sizes}", flush=True)
    else:
        print(f"__SRV__ no graph, batch up to {slots}", flush=True)

    # Pinned memory for input and output (`rollout.serve_pinned`): slot rows
    # are gathered with `np.take` straight into it, and transfers to and from
    # the GPU are async, queued together with the graph. Without it each of the
    # eleven transfers from pageable memory is synchronous (the driver copies
    # through its own buffer). The net gets THE SAME static buffers — values and
    # alignment don't change.
    pin_in, pin_out = None, None
    if pinned and device.startswith("cuda"):
        pin_in = {name: torch.empty((slots,) + tuple(shape),
                                    dtype=getattr(torch, dt)).pin_memory()
                  for name, shape, dt in in_spec}
        pin_in_np = {k_: v.numpy() for k_, v in pin_in.items()}
        pin_out = torch.empty((slots, ototal), dtype=torch.float32).pin_memory()
        pin_out_np = pin_out.numpy()
    if streams:
        # per model: pinned input/output, stream and completion event
        m_in, m_in_np, m_out, m_out_np, m_st, m_ev = [], [], [], [], [], []
        for m in range(len(nets)):
            pi = ({name: torch.empty((slots,) + tuple(shape),
                                     dtype=getattr(torch, dt)).pin_memory()
                   for name, shape, dt in in_spec} if m else pin_in)
            po = (torch.empty((slots, ototal), dtype=torch.float32).pin_memory()
                  if m else pin_out)
            m_in.append(pi); m_in_np.append({k_: v.numpy() for k_, v in pi.items()})
            m_out.append(po); m_out_np.append(po.numpy())
            m_st.append(cap_st[m])      # replay on the capture stream
            m_ev.append(torch.cuda.Event(blocking=bool(block_sync)))
    idx = np.arange(slots)
    fl = np.frombuffer(flags, dtype=np.int8)
    # How much CORE time the server itself got and how long it sat in the
    # scheduler queue ready to run (`/proc/self/schedstat`: time on core and
    # time waiting for a core, ns). Without it "server saturated" is
    # indistinguishable from "server can't get a core": 42+ processes on 15 cores.
    def _sched():
        try:
            with open("/proc/self/schedstat") as f:
                a_, b_, _ = f.read().split()
            return int(a_) / 1e9, int(b_) / 1e9
        except OSError:
            return float("nan"), float("nan")
    t_srv0, cpu0, (run0, wait0) = time.time(), time.process_time(), _sched()
    while True:
        # The semaphore is an alarm clock, not a counter. Counting tokens would
        # need a correction for the worker raising its flag BEFORE releasing
        # the semaphore, and that debt bookkeeping only survives until the first
        # worker crash. Instead: polling flags with a short timeout — normally
        # a token wakes us, the poll is insurance against a lost one.
        t_w = time.time()
        got = req_sem.acquire(timeout=0.05)
        if got:
            while req_sem.acquire(block=False):
                pass
            # ⚠ DELIBERATE accumulation. The server answers faster than workers
            # come back, so the mean batch with 14 workers is only 3.4 — while a
            # pass costs nearly the same at batch 1 and batch 14 (1.81 vs
            # 2.7 ms). So it pays to wait a couple of ms and compute twice as
            # much for the same price. We wait by TIME, not by count: otherwise
            # at the rollout tail, with fewer live workers than `min_batch`, the
            # server would hang forever.
            if min_batch > 1:
                end = time.time() + max_wait
                while int((fl == 1).sum()) < min_batch:
                    left = end - time.time()
                    if left <= 0:
                        break
                    if req_sem.acquire(timeout=left):
                        while req_sem.acquire(block=False):
                            pass
        stats[3] += time.time() - t_w
        # ⚠ Order in `RemoteNet`: input first, then flag, then semaphore.
        # On x86 stores are not reordered with each other, so a visible flag
        # means the input is already written. On a weak memory model (ARM) the
        # polling path would need a barrier.
        ready = idx[fl == 1]
        if len(ready) == 0:
            if stop.value:
                break
            continue

        # Arrivals are split by MODEL: one wakeup may contain both our policy
        # and two different league opponents. They can't share a batch — the
        # weights differ.
        if mids is None or len(nets) == 1:
            groups = [(0, ready)]
        else:
            mv = np.frombuffer(mids, dtype=np.int32)[ready]
            groups = [(int(m), ready[mv == m]) for m in np.unique(mv)]

        if streams:
            # All groups go to their own streams without syncing between them,
            # then each is awaited on its event. Model m's graph reads and writes
            # only its own buffers, so concurrency shares nothing; pass numbers
            # are identical (verified by an internal parity script).
            t0 = time.time()
            with torch.no_grad():
                for m, grp in groups:
                    k = len(grp)
                    with torch.cuda.stream(m_st[m]):
                        for name, _, _ in in_spec:
                            np.take(vin[name], grp, axis=0, out=m_in_np[m][name][:k])
                            statics[m][name][:k].copy_(m_in[m][name][:k],
                                                       non_blocking=True)
                        gk, out = next((g, o) for kk, g, o in graphs[m] if kk >= k)
                        gk.replay()
                        m_out[m][:k].copy_(out[:k], non_blocking=True)
                        m_ev[m].record(m_st[m])
            t1 = time.time()
            for m, grp in groups:
                m_ev[m].synchronize()
                vout[grp] = m_out_np[m][:len(grp)]
                for j in grp:
                    flags[j] = 0
                    rep_sems[j].release()
            now = time.time()
            stats[0] += len(groups)
            stats[1] += len(ready)
            stats[2] += now - t0
            stats[4] += t1 - t0        # launch: gather input and enqueue
            stats[5] += now - t1       # wait for GPU and hand out replies
            continue

        for m, grp in groups:
            t0 = time.time()
            k = len(grp)
            with torch.no_grad():
                if pin_in is not None:
                    for name, _, _ in in_spec:
                        np.take(vin[name], grp, axis=0, out=pin_in_np[name][:k])
                        static[name][:k].copy_(pin_in[name][:k], non_blocking=True)
                else:
                    for name, _, _ in in_spec:
                        static[name][:k].copy_(torch.from_numpy(vin[name][grp]))
                t1 = time.time()
                if graphs[m]:
                    # Smallest graph of this model that fits the arrivals. The
                    # tail of extra slots holds the probe observation and is
                    # computed for nothing.
                    gk, out = next((g, o) for kk, g, o in graphs[m] if kk >= k)
                    gk.replay()
                    if pin_out is not None:
                        pin_out[:k].copy_(out[:k], non_blocking=True)
                    if ev is not None:
                        ev.record()
                        ev.synchronize()
                    else:
                        torch.cuda.synchronize()
                else:
                    # Without a graph the batch is cut exactly to the arrivals:
                    # with one worker that is batch 1, and only because of this
                    # the no-graph mode can match serial collection bitwise.
                    out = call(k, m)
                t2 = time.time()
                if pin_out is not None and graphs[m]:
                    vout[grp] = pin_out_np[:k]
                else:
                    vout[grp] = out[:k].cpu().numpy()
            now = time.time()
            stats[0] += 1
            stats[1] += k
            stats[2] += now - t0
            stats[4] += t1 - t0        # input: shared memory -> GPU
            stats[5] += t2 - t1        # the pass itself
            stats[6] += now - t2       # output: GPU -> shared memory

            for j in grp:
                flags[j] = 0
                rep_sems[j].release()
    run1, wait1 = _sched()
    print(f"__SRV__ cpu: wall {time.time() - t_srv0:.1f} s, process time "
          f"{time.process_time() - cpu0:.1f} s, on core {run1 - run0:.1f} s, "
          f"in scheduler queue {wait1 - wait0:.1f} s", flush=True)
    print(f"__SRV__ stopped: batches {int(stats[0])}, states {int(stats[1])}, "
          f"in pass {stats[2]:.1f} s (input {stats[4]:.1f} + net {stats[5]:.1f} "
          f"+ output {stats[6]:.1f}), waiting {stats[3]:.1f} s, "
          f"mean batch {stats[1] / max(stats[0], 1):.1f}", flush=True)


# ------------------------------------------------------------------ client

class RemoteNet:
    """Worker-side "net": writes input into its slot and waits for the reply.

    To `play.agent` it is indistinguishable from a real net — an object that is
    called and returns a dict of batch-1 tensors. No decoder, encoder or
    `log π` here, and there can't be: the object is swapped, not the code.
    """

    def __init__(self, slot: int, spec, arrays_in, arrays_out, flags,
                 req_sem, rep_sem, slots: int, timeout: float = 120.0,
                 mid: int = 0, mids=None):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.j = slot
        self.in_spec = spec["in_spec"]
        self.out_spec = spec["out_spec"]
        self.n_args = spec["n_args"]
        self.kw_names = spec["kw_names"]
        self.const = spec["const"]
        self.vin = view(arrays_in, self.in_spec, slots)
        # Output arrives as one flat row: the server concatenates it inside the
        # graph so the bus carries one transfer instead of thirteen
        # (`out_layout`). Here the row is split back into heads — a `reshape`
        # over the same memory plus one copy per head, all in-process and off
        # the bus.
        self.olay, self.ototal = out_layout(self.out_spec)
        self.vout = np.frombuffer(arrays_out, dtype=np.float32).reshape(
            slots, self.ototal)
        self.flags = flags
        self.req = req_sem
        self.rep = rep_sem
        self.timeout = timeout
        self.calls = 0
        # Model id: 0 is the trained policy, then league opponents in the order
        # `collect_served` registered them. Written to the slot with the input,
        # because BOTH sides take turns in one worker process, and the server
        # must know whose weights to use.
        self.mid = int(mid)
        self.mids = mids

    def __call__(self, *args, **kwargs):
        j = self.j
        for i in range(self.n_args):
            self.vin[f"a{i}"][j] = args[i].numpy()[0]
        for k in self.kw_names:
            self.vin[k][j] = kwargs[k].numpy()[0]
        for k, v in self.const.items():
            if kwargs.get(k) != v:
                raise RuntimeError(
                    f"worker calls the net with {k}={kwargs.get(k)!r}, but the slot was "
                    f"captured with {v!r}: these are different passes, they can't share a batch")
        if self.mids is not None:
            self.mids[j] = self.mid
        self.flags[j] = 1
        self.req.release()
        if not self.rep.acquire(timeout=self.timeout):
            raise RuntimeError(f"inference server did not reply within {self.timeout} s")
        self.calls += 1
        # Copy is mandatory: the slot is overwritten by the very next batch.
        row = self.vout[j]
        out = {}
        for name, (a, n, shape) in self.olay.items():
            out[name] = self.torch.from_numpy(
                row[a:a + n].reshape(shape)).unsqueeze(0).clone()
        return out
