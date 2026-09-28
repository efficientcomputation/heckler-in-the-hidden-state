#!/usr/bin/env python
"""Run D: final-layer battery on Run A's activations (reviewer point 5).
Three tests of the compression hypothesis, all on the same LOO-by-problem protocol:
  1. NONLINEAR probes (logistic + small MLP) at the final layer vs the peak layer:
     does a fancier decoder recover the signal the linear probe loses at the end?
     (Also finally backs or kills the primer's "elaborate decoder did worse" line.)
  2. POST-FINAL-NORM probe: apply the model's final RMSNorm to last-layer acts, probe.
     Does normalization itself destroy the read?
  3. UNEMBEDDING PROJECTION: split each layer's activation into the component inside
     the top-k right-singular subspace of W_U (the token-relevant subspace) and its
     complement; probe both. If the final-layer drop is the readout squeeze, the read
     should survive better OUTSIDE the token subspace at the peak and be squeezed at
     the end.
Needs: runA shards + the model's norm weight and unembedding (extracted once).
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


def lin_scores(X, y, prob, folds=5, seed=0):
    uq = np.unique(prob)
    rr = np.random.default_rng(seed)
    order = uq.copy(); rr.shuffle(order)
    S = np.full(len(y), np.nan, np.float32)
    for te_p in np.array_split(order, folds):
        te = np.isin(prob, te_p); tr = ~te
        d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
        d /= max(np.linalg.norm(d), 1e-9)
        S[te] = X[te] @ d
    return S


def nonlin_scores(X, y, prob, kind, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    uq = np.unique(prob)
    rr = np.random.default_rng(seed)
    order = uq.copy(); rr.shuffle(order)
    S = np.full(len(y), np.nan, np.float32)
    for te_p in np.array_split(order, folds):
        te = np.isin(prob, te_p); tr = ~te
        if kind == "logistic":
            clf = LogisticRegression(max_iter=2000, C=0.05)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=600,
                                alpha=1e-2, random_state=0)
        clf.fit(X[tr], y[tr])
        S[te] = clf.predict_proba(X[te])[:, 1]
    return S


def report(tag, y, s, prob, n_perm=200):
    a = within_auc(y, s, prob)
    nulls = np.array([within_auc(y, s, prob, rng=np.random.default_rng(3000 + i))
                      for i in range(n_perm)])
    z = (a - nulls.mean()) / (nulls.std() + 1e-9)
    print(f"  {tag:<38} AUC={a:.3f}  null={nulls.mean():.3f}±{nulls.std():.3f}  z={z:+.2f}",
          flush=True)
    return {"auc": a, "null": [float(nulls.mean()), float(nulls.std())], "z": float(z)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="~/jlens/out/runA/runA_shard*.pt")
    ap.add_argument("--model", default="apple/DiffuCoder-7B-cpGRPO")
    ap.add_argument("--topk", type=int, default=64, help="top-k unembedding singular dirs")
    ap.add_argument("--out", default="~/jlens/out/runD.json")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.expanduser(a.glob))):
        rows.extend(torch.load(f, weights_only=False))
    print(f"[runD] {len(rows)} rollouts, {len(set(r['task_id'] for r in rows))} problems", flush=True)
    y = np.array([1.0 if r["pass"] else 0.0 for r in rows])
    prob = np.array([r["task_id"] for r in rows])
    X = np.stack([r["acts"][-1].float().numpy() for r in rows])   # [N, L, d] last step
    nL = X.shape[1]

    # center per problem
    Xc = X.astype(np.float32)
    for p in np.unique(prob):
        m = prob == p
        Xc[m] -= Xc[m].mean(0, keepdims=True)

    # linear depth curve -> peak & final layers
    curve = [within_auc(y, lin_scores(Xc[:, L], y, prob), prob) for L in range(nL)]
    peakL, finalL = int(np.nanargmax(curve)), nL - 1
    print(f"linear depth: peak L{peakL}={curve[peakL]:.3f}  final L{finalL}={curve[finalL]:.3f}", flush=True)
    res = {"curve_linear": [round(float(c), 4) for c in curve],
           "peakL": peakL, "finalL": finalL}

    # 1. nonlinear probes at peak and final
    print("(1) nonlinear probes:", flush=True)
    for L, name in ((peakL, "peak"), (finalL, "final")):
        for kind in ("logistic", "mlp"):
            res[f"{kind}_{name}"] = report(f"{kind} @ {name} L{L}",
                                           y, nonlin_scores(Xc[:, L], y, prob, kind), prob)

    # 2. post-final-norm probe (RMSNorm weight from the model, applied to final acts)
    from transformers import AutoModel
    m = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32, trust_remote_code=True)
    core = getattr(m, "model", m)
    g = core.norm.weight.detach().float().numpy()               # [d]
    W = m.get_output_embeddings().weight.detach().float().numpy()  # [V, d]
    del m, core
    Xf = X[:, finalL].astype(np.float32)
    rms = np.sqrt((Xf ** 2).mean(-1, keepdims=True) + 1e-6)
    Xn = (Xf / rms) * g
    for p in np.unique(prob):
        mm = prob == p
        Xn[mm] -= Xn[mm].mean(0, keepdims=True)
    print("(2) post-final-norm probe:", flush=True)
    res["post_norm_final"] = report(f"linear @ final, after RMSNorm", y, lin_scores(Xn, y, prob), prob)

    # 3. unembedding-subspace split at peak and final
    print(f"(3) unembedding top-{a.topk} subspace split:", flush=True)
    Wc = W - W.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    B = Vt[:a.topk].T                                            # [d, k] token-relevant basis
    for L, name in ((peakL, "peak"), (finalL, "final")):
        XL = Xc[:, L]
        Xin = XL @ B @ B.T                                       # inside token subspace
        Xout = XL - Xin                                          # complement
        res[f"tok_subspace_{name}"] = report(f"inside W_U top-{a.topk} @ {name}",
                                             y, lin_scores(Xin, y, prob), prob)
        res[f"tok_complement_{name}"] = report(f"outside W_U top-{a.topk} @ {name}",
                                               y, lin_scores(Xout, y, prob), prob)

    json.dump(res, open(os.path.expanduser(a.out), "w"), indent=2)
    print(f"RUND_DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
