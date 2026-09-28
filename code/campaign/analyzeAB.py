#!/usr/bin/env python
"""Analysis for Run A (DiffuCoder) and Run B (Qwen2.5-Coder):
probe AUC vs the model's own confidence, same within-problem protocol, strict nulls.

Protocol (matches the study): per-problem centering, K-fold by problem, difference-of-means
direction, within-problem AUC averaged over problems. The probe's shuffle null re-runs the
WHOLE pipeline including argmax-over-layers, x200. Confidence baselines get the same
within-problem shuffle null (no layer choice to correct for).

Outputs JSON + prints. Run:  analyzeAB.py --run A|B --glob '~/jlens/out/runA/runA_shard*.pt'
"""
import argparse, glob, json, os
import numpy as np
import torch


def wauc_one(y, s):
    o = np.argsort(s)
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_auc(y, s, prob, rng=None):
    """Mean within-problem AUC; optional within-problem label shuffle."""
    yy = y.copy()
    if rng is not None:
        for p in np.unique(prob):
            m = prob == p
            yy[m] = rng.permutation(yy[m])
    v = []
    for p in np.unique(prob):
        m = prob == p
        if len(set(yy[m])) < 2:
            continue
        v.append(wauc_one(yy[m], s[m]))
    return float(np.nanmean(v)) if v else np.nan


def probe_scores(X, y, prob, folds=5, seed=0):
    """X [N, L, d] centered per problem -> out-of-fold projection scores [N, L]."""
    uq = np.unique(prob)
    rr = np.random.default_rng(seed)
    order = uq.copy(); rr.shuffle(order)
    S = np.full((len(y), X.shape[1]), np.nan, np.float32)
    for te_p in np.array_split(order, folds):
        te = np.isin(prob, te_p); tr = ~te
        d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        S[te] = np.einsum("nld,ld->nl", X[te], d)
    return S


def probe_with_null(X, y, prob, n_perm=200):
    """Returns (auc, bestL, curve, null_mean, null_sd, z). Null repeats argmax-over-layers."""
    S = probe_scores(X, y, prob)
    nL = X.shape[1]
    curve = [within_auc(y, S[:, L], prob) for L in range(nL)]
    bestL = int(np.nanargmax(curve))
    nulls = []
    for i in range(n_perm):
        rg = np.random.default_rng(1000 + i)
        per_layer = [within_auc(y, S[:, L], prob, rng=np.random.default_rng(1000 + i)) for L in range(nL)]
        nulls.append(np.nanmax(per_layer))
    nulls = np.array(nulls)
    z = (curve[bestL] - nulls.mean()) / (nulls.std() + 1e-9)
    return curve[bestL], bestL, curve, float(nulls.mean()), float(nulls.std()), float(z)


def conf_with_null(y, s, prob, n_perm=200):
    a = within_auc(y, s, prob)
    nulls = np.array([within_auc(y, s, prob, rng=np.random.default_rng(2000 + i)) for i in range(n_perm)])
    return float(a), float(nulls.mean()), float(nulls.std()), float((a - nulls.mean()) / (nulls.std() + 1e-9))


def paired_boot(y, s_probe, s_conf, prob, n_boot=2000):
    """Bootstrap (cluster by problem) of probe_auc - conf_auc."""
    uq = np.unique(prob)
    diffs = []
    rg = np.random.default_rng(7)
    for _ in range(n_boot):
        pick = rg.choice(uq, len(uq), replace=True)
        idx = np.concatenate([np.flatnonzero(prob == p) for p in pick])
        d = within_auc(y[idx], s_probe[idx], prob[idx]) - within_auc(y[idx], s_conf[idx], prob[idx])
        diffs.append(d)
    diffs = np.array(diffs)
    return float(np.nanmean(diffs)), float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["A", "B"])
    ap.add_argument("--glob", required=True)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.expanduser(a.glob))):
        rows.extend(torch.load(f, weights_only=False))
    print(f"[{a.run}] {len(rows)} rollouts from {len(set(r['task_id'] for r in rows))} mixed problems")
    y = np.array([1.0 if r["pass"] else 0.0 for r in rows])
    prob = np.array([r["task_id"] for r in rows])

    if a.run == "A":
        # acts: [steps, layers, d] -> last denoising step
        X = np.stack([r["acts"][-1].float().numpy() for r in rows])           # [N, L, d]
        confs = {"conf_vis": np.array([r["conf_vis"] for r in rows]),
                 "entropy_vis_neg": -np.array([r["entropy_vis"] for r in rows])}
        if rows[0].get("conf_masked") is not None:
            confs["conf_masked"] = np.array([r["conf_masked"] for r in rows])
    else:
        X = np.stack([r["act_mean"].float().numpy() for r in rows])           # [N, L+1, d]
        confs = {"conf": np.array([r["conf"] for r in rows]),
                 "entropy_neg": -np.array([r["entropy"] for r in rows])}

    # per-problem centering
    Xc = X.astype(np.float32)
    for p in np.unique(prob):
        m = prob == p
        Xc[m] -= Xc[m].mean(0, keepdims=True)

    res = {"n_rollouts": len(rows), "n_problems": int(len(np.unique(prob))),
           "n_pass": int(y.sum())}

    auc_p, bestL, curve, nm, ns, z = probe_with_null(Xc, y, prob, a.n_perm)
    res["probe"] = {"auc": auc_p, "layer": bestL, "null": [nm, ns], "z": z,
                    "curve": [None if np.isnan(c) else round(c, 4) for c in curve]}
    print(f"PROBE  auc={auc_p:.3f} @L{bestL}  null={nm:.3f}±{ns:.3f}  z={z:+.2f}")
    print("  depth: " + " ".join(f"L{L}:{curve[L]:.2f}" for L in range(0, len(curve), max(1, len(curve)//14))))

    S = probe_scores(Xc, y, prob)
    s_probe = S[:, bestL]
    res["confidence"] = {}
    for name, s in confs.items():
        ac, cm, cs, cz = conf_with_null(y, s, prob, a.n_perm)
        d, lo, hi = paired_boot(y, s_probe, s, prob)
        res["confidence"][name] = {"auc": ac, "null": [cm, cs], "z": cz,
                                   "probe_minus_this": [d, lo, hi]}
        beats = "probe BEATS it" if lo > 0 else ("probe does NOT beat it" if hi < 0 else "inconclusive")
        print(f"CONF   {name:<14} auc={ac:.3f} z={cz:+.2f}   probe-minus={d:+.3f} CI[{lo:+.3f},{hi:+.3f}]  -> {beats}")

    json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
    print(f"ANALYZE_DONE run={a.run} -> {a.out}")


if __name__ == "__main__":
    main()
