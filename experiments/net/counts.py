"""Quantity heads: gate + softmax over quantity classes + "all" flag.

The structure is like the argument heads of AlphaStar / OpenAI Five and like the multi-step
choice in pokemoni-pobeda: first the decision "act or not" (gate, 2 classes),
if acting, "how many" (categorical over quantity classes), and for sells also
"the whole stock" (2 classes). At every level training is softmax/CE, play is argmax,
and argmax is the limit of sampling as tau -> 0. This is what separates oh3 from the earlier
CORN chain (25.09): its played median was NOT the limit of sampling
(internal parity analysis, section 3), and from a flat softmax over {none, bins, all} (oh1,
oh2): there "no order" does not compete with twenty quantities
individually: oh1 gave a buy recall of 0.14 and -$8,550 head-to-head.

Quantity classes are all integers 1..30 and coarse ones above (36 classes): on 60 teacher
games this covers >= 95% of exact values with p90 = 28 for sells; the budget cap
(R3) closes classes with value > m BEFORE sampling, "all" stays open at m >= 1.
The label is the nearest class, with ordinal smoothing SMOOTH to the neighbours (n = 20
teaches both 19 and 21; otherwise a rare class is learned in isolation -- exactly what
the softmax over exact numbers failed on in 2026-09-2x).

Outputs per item: 1 (gate) + C (classes) + 1 ("all" for sells). The shapes
are baked into the weights, the live configuration is `envcfg`, a mismatch is caught there (A7).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SHED_CAP = 100        # configuration.shedCapacity in training
MAX_ORDERS = 10       # configuration.maxMarketOrdersPerTurn in training
CAPS = {"sell": SHED_CAP, "buy": SHED_CAP, "seed": SHED_CAP,
        "animal": SHED_CAP, "hire": MAX_ORDERS}
HAS_ALL = ("sell",)
UNIT_CAPS = {"pickup": SHED_CAP}
ALL_CAPS = {**CAPS, **UNIT_CAPS}
SMOOTH = 0.1          # share of label mass going to the two neighbouring classes

_V = list(range(1, 31)) + [35, 40, 50, 60, 80, 100]
VALUES = {h: (_V if c > MAX_ORDERS else list(range(1, c + 1))) for h, c in ALL_CAPS.items()}


def n_logits(head):
    """Gate + quantity classes + "all" flag."""
    return 1 + len(VALUES[head]) + (1 if head in HAS_ALL else 0)


_VALUES_T = {}


def values(head, device=None):
    """Class values as a tensor. Cached by (head, device): inside a forward
    pass under CUDA-graph capture (rollout server) tensors must not be
    created: "operation not permitted when stream is capturing"."""
    key = (head, str(device) if device is not None else "cpu")
    t = _VALUES_T.get(key)
    if t is None:
        t = torch.tensor(VALUES[head], dtype=torch.long, device=device)
        _VALUES_T[key] = t
    return t


def split(logits, head):
    """Split the head output into gate, quantity classes and "all" flag."""
    C = len(VALUES[head])
    gate = logits[..., 0]
    cond = logits[..., 1:1 + C]
    allf = logits[..., 1 + C] if head in HAS_ALL else None
    return gate, cond, allf


def label(head, n):
    """Class of the nearest value (lower one on ties). n is long, n >= 1."""
    v = values(head, n.device)
    return (n.long().clamp_min(1).unsqueeze(-1) - v).abs().argmin(-1)


def _mask(head, m, device):
    """Classes with value <= m. m is long [...]; at least class "1" is open."""
    v = values(head, device)
    ok = v <= torch.as_tensor(m, device=device).long().clamp_min(1).unsqueeze(-1)
    ok[..., 0] = True
    return ok


def log_pmf(cond, m, tau=1.0, head=None):
    """log P(class | act) at cap m and temperature tau. cond [..., C]."""
    lg = cond.float() / max(float(tau), 1e-6)
    return torch.log_softmax(lg.masked_fill(~_mask(head, m, lg.device), float("-inf")), -1)


def argmax_n(cond, m, head):
    """Played quantity: value of the argmax class among those open under the cap."""
    c = log_pmf(cond, m, 1.0, head).argmax(-1)
    return values(head, cond.device)[c]


def expect(logits, head):
    """E[n] = sigmoid(gate) * sum_c P(c) * value_c (without the budget mask -- the corrector
    sees what the net WANTS)."""
    gate, cond, _ = split(logits, head)
    p = torch.softmax(cond.float(), -1)
    return torch.sigmoid(gate.float()) * (p * values(head, logits.device).float()).sum(-1)


def median(cond, m=None, head=None):
    """Compatibility by name: deterministic quantity = argmax of the class."""
    if head is None:
        raise ValueError("counts.median: head is required")
    mm = torch.full(cond.shape[:-1], ALL_CAPS[head], dtype=torch.long,
                    device=cond.device) if m is None else torch.as_tensor(m, device=cond.device)
    return argmax_n(cond, mm, head)


def log_prob(cond, n, m, tau=1.0, head=None):
    """log P(n | act) at cap m: label class n. tau = 0 is argmax, 0."""
    if tau == 0:
        return torch.zeros(n.shape, device=cond.device)
    lp = log_pmf(cond, m, tau, head)
    return lp.gather(-1, label(head, n).unsqueeze(-1)).squeeze(-1)


def pmf(cond, m, tau=1.0, head=None):
    """P over classes at cap m, for sampling. -> [C]; the chosen class
    is converted to a number via `values`."""
    return torch.exp(log_pmf(cond, torch.tensor(int(m)), tau, head))


def loss(cond, n, head=None):
    """CE over the label class with ordinal smoothing SMOOTH."""
    C = cond.shape[-1]
    c = label(head, n)
    tgt = F.one_hot(c, C).float()
    if SMOOTH > 0:
        left = F.one_hot((c - 1).clamp_min(0), C).float()
        right = F.one_hot((c + 1).clamp_max(C - 1), C).float()
        tgt = (1 - SMOOTH) * tgt + SMOOTH * 0.5 * (left + right)
    ls = torch.log_softmax(cond.float(), -1)
    return -(tgt * ls).sum(-1)


def decode(logits, head, gate_thr=0.5, have=None, sample=False, gen=None):
    """Number from the head outputs WITHOUT the budget mask, for validation and probes:
    gate by threshold (or sampling), quantity by argmax, "all" substitutes
    the number with the stock `have`."""
    gate, cond, allf = split(logits, head)
    if sample:
        act = torch.bernoulli(torch.sigmoid(gate.float()), generator=gen).bool()
    else:
        act = torch.sigmoid(gate.float()) > gate_thr
    n = median(cond, head=head)
    n = torch.where(act, n, torch.zeros_like(n))
    if allf is not None and have is not None:
        pa = torch.sigmoid(allf.float())
        take_all = (torch.bernoulli(pa, generator=gen).bool() if sample
                    else pa > 0.5) & act
        n = torch.where(take_all, have, torch.minimum(n, have))
    return n
