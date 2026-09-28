#!/usr/bin/env python
"""Analytic J-space reads from a clean capture (neuro_clean.py output).

Because the mu read "perturb layer L, read at L" is
    pooled'[ex,L] = pooled[ex,L] + m * meannorm[ex,L] * v[ex]      (v = one unit dir/example)
every read under noise is a pure function of the clean per-layer pooled activation + mean row
norm -- validated analytic==real in neuro_clean.py.  This script computes, from clean_<name>.npz:

  (A) layer_under_noise_<name>.json : perturb-L-read-L pooled AUC at EVERY layer for m in {0,0.4,0.8},
      5 seeds, refit + fixed-clean-direction + within-problem shuffle null.
  (B) neuro_partA_<name>.json       : the flash "jump" artifact test at the PEAK layer over the fine
      grid m in {0,.2,.4,.6,.8,1,1.5} x 5 seeds -- refit auc, fixed auc, drift, per-position auc, and
      DEGENERACY metrics (participation ratio, effective rank, collapse cosine, variance ratio).
  Dumps peak-layer perturbed pooled activations per m (seed 0) to dumps/peakact_<name>.npz for Part B.

Reuses pert_ext machinery VERBATIM (make_wauc, fresh_scores, make_wauc_pp, dir_pp, center_pp).
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = os.environ.get("JP_THREADS", "8")
sys.path.insert(0, "/scratch/jspace_pert")
import numpy as np
import pert_ext as PE

DUMP = "/scratch/jspace_neuro/dumps"
OUT = "/scratch/jspace_neuro"
GRID_A = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
GRID_L = [0.0, 0.4, 0.8]
SEEDS = [0, 1, 2, 3, 4]
NPERM = 200


def eps_for(n, D, seed):
    rng = np.random.default_rng(1_000_000 + seed)
    e = rng.standard_normal((n, D)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


def center_pooled(M, prob, uq):
    Mc = M.astype(np.float32).copy()
    for p in uq:
        m = prob == p
        Mc[m] -= Mc[m].mean(0, keepdims=True)
    return Mc


def spectrum(Mc):
    """Nonzero-spectrum eigenvalues via the n x n Gram matrix (cheap for n<<D)."""
    G = Mc @ Mc.T
    lam = np.linalg.eigvalsh(G)
    lam = np.clip(lam, 0, None)
    lam = lam[lam > 1e-9 * lam.max()] if lam.max() > 0 else lam
    return lam


def part_ratio(lam):
    s = lam.sum()
    return float(s * s / (np.square(lam).sum() + 1e-30)) if s > 0 else 0.0


def eff_rank(lam):
    s = lam.sum()
    if s <= 0:
        return 0.0
    p = lam / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def collapse_cos(M):
    """Mean pairwise cosine of RAW pooled vectors -> ~1 if collapsed toward a common vector."""
    N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    C = N @ N.T
    n = C.shape[0]
    return float((C.sum() - np.trace(C)) / (n * (n - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    Z = np.load(f"{DUMP}/clean_{a.name}.npz", allow_pickle=True)
    pooled = Z["pooled"]; meannorm = Z["meannorm"]; PP = Z["PP_peak"]; owner = Z["owner"]
    y = Z["y"].astype(float); prob = Z["prob"]; kind = Z["kind"]; dclean = Z["dclean"]
    peak = int(Z["peak"]); hidden = int(Z["hidden"]); nL = int(Z["nL"])
    uq = np.unique(prob)
    n = pooled.shape[0]
    wauc = PE.make_wauc(y, prob, kind, uq)
    prob_long = prob[owner]; y_long = y[owner]
    wauc_pp = PE.make_wauc_pp(y, prob, owner, uq)
    print(f"[{a.name}] loaded n={n} peak=L{peak} nL={nL} hidden={hidden} "
          f"val_real={float(Z['val_real_auc']):.4f} val_ana={float(Z['val_ana_auc']):.4f}", flush=True)

    def refit_auc(Mperturbed):
        Mc = center_pooled(Mperturbed, prob, uq)
        S = PE.fresh_scores(Mc[:, None, :], y, prob, uq, 0)
        return wauc(S)[0], Mc, S

    def fixed_auc(Mc, L):
        return wauc(Mc @ dclean[L])[0]

    def null_stats(S):
        nl = np.array([wauc(S, seed=7000 + i)[0] for i in range(NPERM)])
        return float(nl.mean()), float(nl.std())

    # ---------- (A) layer_under_noise: perturb-L-read-L, every layer ----------
    t0 = time.time()
    rows = []
    for L in range(nL):
        pooledL = pooled[:, L]; mnL = meannorm[:, L][:, None]
        for m in GRID_L:
            aucs, fixes = [], []
            null_m = null_s = float("nan")
            for si, seed in enumerate(SEEDS):
                if m == 0.0:
                    Mp = pooledL
                else:
                    Mp = pooledL + m * mnL * eps_for(n, hidden, seed)
                a_ref, Mc, S = refit_auc(Mp)
                aucs.append(a_ref); fixes.append(fixed_auc(Mc, L))
                if si == 0:
                    null_m, null_s = null_stats(S)
                if m == 0.0:
                    break
            rows.append(dict(layer=L, m=m, auc=float(np.mean(aucs)),
                             auc_sd=float(np.std(aucs)), auc_fixed=float(np.mean(fixes)),
                             auc_fixed_sd=float(np.std(fixes)), auc_null_mean=null_m,
                             auc_null_std=null_s, n_seeds=len(aucs)))
        if L == peak:
            r = [x for x in rows if x["layer"] == L]
            print(f"[{a.name}] L{L}(peak): " + " ".join(f"m{r_['m']}={r_['auc']:.3f}" for r_ in r), flush=True)
    json.dump(rows, open(f"{OUT}/layer_under_noise_{a.name}.json", "w"), indent=2)
    print(f"[{a.name}] wrote layer_under_noise_{a.name}.json ({len(rows)} rows) in {time.time()-t0:.0f}s", flush=True)

    # ---------- (B) Part A: peak-layer fine grid x seeds + degeneracy ----------
    t1 = time.time()
    pooledP = pooled[:, peak]; mnP = meannorm[:, peak][:, None]
    clean_lam = spectrum(center_pooled(pooledP, prob, uq)); clean_var = float(clean_lam.sum())
    partA = []
    peak_dump = {}
    for m in GRID_A:
        for si, seed in enumerate(SEEDS):
            if m == 0.0:
                Mp = pooledP; PPp = PP
            else:
                e = eps_for(n, hidden, seed)
                Mp = pooledP + m * mnP * e
                PPp = PP + (m * meannorm[owner, peak][:, None] * e[owner])   # per-position: same v/ex
            a_ref, Mc, S = refit_auc(Mp)
            nm, ns = null_stats(S)
            a_fix = fixed_auc(Mc, peak)
            drift = float(np.mean(PE.cos_rows(Mp, pooledP)))
            # per-position read
            Cpp = PE.center_pp(PPp, prob_long, uq)
            Sp = PE.dir_pp(Cpp, y_long, prob_long, uq)
            a_pp = wauc_pp(Sp)
            npp = np.array([wauc_pp(Sp, seed=9000 + i) for i in range(NPERM)])
            zpp = (a_pp - npp.mean()) / (npp.std() + 1e-9)
            # degeneracy
            lam = spectrum(Mc)
            row = dict(model=a.name, ptype="mu", peak_layer=peak, magnitude=m, seed=seed,
                       auc=float(a_ref), auc_fixed=float(a_fix),
                       auc_null_mean=nm, auc_null_std=ns,
                       auc_z=float((a_ref - nm) / (ns + 1e-9)),
                       auc_perpos=float(a_pp), auc_perpos_z=float(zpp),
                       drift_cosine=drift,
                       part_ratio=part_ratio(lam), eff_rank=eff_rank(lam),
                       collapse_cos=collapse_cos(Mp), var_ratio=float(lam.sum() / (clean_var + 1e-30)))
            partA.append(row)
            if si == 0:
                peak_dump[f"m{m}"] = Mp.astype(np.float32)
            if m == 0.0:
                for s2 in SEEDS[1:]:                       # replicate clean row per seed for symmetry
                    r2 = dict(row); r2["seed"] = s2; partA.append(r2)
                break
    json.dump(partA, open(f"{OUT}/neuro_partA_{a.name}.json", "w"), indent=2)
    # aggregate print
    import collections
    agg = collections.defaultdict(list)
    for r in partA:
        agg[r["magnitude"]].append(r)
    print(f"[{a.name}] Part A (peak L{peak}) mean+-sd over seeds:", flush=True)
    for m in GRID_A:
        rs = agg[m]
        arr = np.array([r["auc"] for r in rs]); afx = np.array([r["auc_fixed"] for r in rs])
        pr = np.array([r["part_ratio"] for r in rs]); dr = np.array([r["drift_cosine"] for r in rs])
        print(f"   m={m}: auc={arr.mean():.3f}+-{arr.std():.3f}  fixed={afx.mean():.3f}+-{afx.std():.3f}"
              f"  drift={dr.mean():.3f}  PR={pr.mean():.1f}", flush=True)
    np.savez_compressed(f"{DUMP}/peakact_{a.name}.npz", y=y, prob=prob, kind=kind, owner=owner,
                        peak=peak, hidden=hidden, meannorm_peak=meannorm[:, peak], **peak_dump)
    print(f"[{a.name}] wrote neuro_partA_{a.name}.json ({len(partA)} rows) + peakact dump in {time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
