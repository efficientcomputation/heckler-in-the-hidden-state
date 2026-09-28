#!/usr/bin/env python
"""Reproduce the PUBLISHED flash 'jump' with the study's EXACT noise seed, then show 5 other
seeds do not. Proves the dip-then-recover is a single-realization artifact of the refit probe.

Study perturb draw (pert_ext.perturb_hook): per batch of bs, eps = torch.randn(B,1,D, gen),
normalized; gen = Generator().manual_seed(100 + int(round(mag*1000))).  Analytic pooled read at
peak (validated Δauc=0.0000 vs real forward): pooled' = pooled[:,peak] + mag*meannorm[:,peak]*eps.
"""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "8"
sys.path.insert(0, "/scratch/jspace_pert")
import numpy as np, torch
import pert_ext as PE

Z = np.load("/scratch/jspace_neuro/dumps/clean_llada_flash.npz", allow_pickle=True)
pooled = Z["pooled"]; meannorm = Z["meannorm"]; y = Z["y"].astype(float)
prob = Z["prob"]; kind = Z["kind"]; peak = int(Z["peak"])
uq = np.unique(prob); n, _, D = pooled.shape
wauc = PE.make_wauc(y, prob, kind, uq)
pP = pooled[:, peak]; mnP = meannorm[:, peak][:, None]


def center(M):
    Mc = M.astype(np.float32).copy()
    for p in uq:
        m = prob == p; Mc[m] -= Mc[m].mean(0, keepdims=True)
    return Mc


def refit(M):
    Mc = center(M)
    return wauc(PE.fresh_scores(Mc[:, None, :], y, prob, uq, 0))[0], Mc


def fixed(Mc, d):
    return wauc(Mc @ d)[0]


d0 = None
_, Mc0 = refit(pP); d0 = (Mc0[y == 1].mean(0) - Mc0[y == 0].mean(0)); d0 /= np.linalg.norm(d0) + 1e-9


def study_eps(mag, bs):
    gen = torch.Generator().manual_seed(100 + int(round(mag * 1000)))
    eps = np.zeros((n, D), np.float32); i = 0
    while i < n:
        B = min(bs, n - i)
        e = torch.randn(B, 1, D, generator=gen)[:, 0, :]
        e = e / e.norm(dim=-1, keepdim=True)
        eps[i:i + B] = e.numpy(); i += B
    return eps


def np_eps(seed):
    rng = np.random.default_rng(1_000_000 + seed)
    e = rng.standard_normal((n, D)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


PUB = {0.2: 0.527, 0.4: 0.375, 0.8: 0.603}
print(f"n={n} D={D} peak=L{peak}  clean refit={refit(pP)[0]:.3f}")
for bs in [8, 6]:
    print(f"\n--- study exact seed (bs={bs}) vs published ---")
    for mag in [0.2, 0.4, 0.8]:
        eps = study_eps(mag, bs)
        a, Mc = refit(pP + mag * mnP * eps)
        print(f"  m={mag}: study-seed refit_auc={a:.3f}  (published {PUB[mag]})  fixed={fixed(Mc, d0):.3f}")

print("\n--- 5 independent seeds (mean +- sd) ---")
for mag in [0.2, 0.4, 0.8]:
    aucs = []
    for s in range(5):
        a, _ = refit(pP + mag * np_eps(s) * mnP)
        aucs.append(a)
    aucs = np.array(aucs)
    print(f"  m={mag}: refit_auc={aucs.mean():.3f}+-{aucs.std():.3f}  per-seed={np.round(aucs,3).tolist()}")
