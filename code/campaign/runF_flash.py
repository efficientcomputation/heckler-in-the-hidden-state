#!/usr/bin/env python
"""Run F: re-measure LLaDA-flash's per-layer correctness read, settling whether the
final-entry cliff (0.81 -> 0.64) is real or a capture artifact.

Design: forward-only (no generation). Corpus = the runC mutant set (MBPP+ references,
one-character mutations, pass/fail by unit test, both sides mutated). For every text:
pooled hidden state at EVERY layer via output_hidden_states, PLUS the last decoder
block's raw output captured by hook, PLUS that same raw output with the model's final
norm applied explicitly. Probe all of them identically. If the cliff is real, raw-last
and normed-last both crater. If it is a capture artifact, the properly-handled entries
stay near the peak.

Batch=1, no padding, no attention mask (sidesteps every mask-convention trap).
fp32 if RAM allows, else bf16 (slow but correct).
"""
import argparse, ast, copy, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/jlens/code"))
sys.path.insert(0, os.path.expanduser("~/jlens"))
from jcode import load_mbppplus, grade_task
import gradefix  # noqa: F401
from ladder_subtle2 import Mut, all_mutants  # mutation machinery (import-safe)
from transformers import AutoModel, AutoTokenizer


def build_corpus(n_tasks=378, cap_each=6):
    rng = random.Random(0)
    tasks = load_mbppplus(n_tasks, 0)
    T, Y, P, K = [], [], [], []
    used = 0
    for bi, t in enumerate(tasks):
        try:
            ref = ast.unparse(ast.parse(t["code"]))
        except SyntaxError:
            continue
        if not grade_task(ref, t):
            continue
        good, bad = [], []
        for src, kind in all_mutants(ref):
            if src.strip() == ref.strip():
                continue
            try:
                compile(src, "<m>", "exec")
            except Exception:
                continue
            (good if grade_task(src, t) else bad).append((src, kind))
        if not good or not bad:
            continue
        rng.shuffle(good); rng.shuffle(bad)
        used += 1
        pre = f"# Problem:\n{t['text']}\n\n# Candidate solution:\n"
        for src, kind in good[:cap_each]:
            T.append(pre + src); Y.append(1); P.append(f"{t['task_id']}#{bi}"); K.append(kind)
        for src, kind in bad[:cap_each]:
            T.append(pre + src); Y.append(0); P.append(f"{t['task_id']}#{bi}"); K.append(kind)
    print(f"[corpus] {used} base programs -> {len(T)} mutants "
          f"({sum(Y)} pass / {len(Y)-sum(Y)} fail)", flush=True)
    return T, np.array(Y, float), np.array(P), np.array(K)


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


def probe(X, y, prob, folds=5, seed=0):
    Xc = X.astype(np.float32)
    for p in np.unique(prob):
        m = prob == p
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    uq = np.unique(prob)
    rr = np.random.default_rng(seed)
    order = uq.copy(); rr.shuffle(order)
    S = np.full(len(y), np.nan, np.float32)
    for te_p in np.array_split(order, folds):
        te = np.isin(prob, te_p); tr = ~te
        d = Xc[tr & (y == 1)].mean(0) - Xc[tr & (y == 0)].mean(0)
        d /= max(np.linalg.norm(d), 1e-9)
        S[te] = Xc[te] @ d
    return S


def auc_with_null(y, S, prob, n_perm=200, seed0=5000):
    a = within_auc(y, S, prob)
    nulls = np.array([within_auc(y, S, prob, rng=np.random.default_rng(seed0 + i))
                      for i in range(n_perm)])
    return a, float(nulls.mean()), float(nulls.std()), float((a - nulls.mean()) / (nulls.std() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_tasks", type=int, default=378)
    ap.add_argument("--maxlen", type=int, default=640)
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--out", default=os.path.expanduser("~/jlens/out/runF_flash.json"))
    ap.add_argument("--acts_out", default=os.path.expanduser("~/jlens/out/runF_acts.npz"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    T, y, prob, kind = build_corpus(a.n_tasks)

    free_gb = int(os.popen("free -g | awk 'NR==2{print $7}'").read().strip() or 0)
    dtype = torch.float32 if free_gb > 440 else torch.bfloat16
    print(f"[load] free={free_gb}GB -> dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, torch_dtype=dtype,
                                      trust_remote_code=True, low_cpu_mem_usage=True).eval()
    core = getattr(model, "model", model)
    layers = core.layers
    nL = len(layers)
    print(f"[load] ok: {nL} blocks, hidden={model.config.hidden_size}", flush=True)

    raw_last = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        raw_last["h"] = h.detach()
    layers[-1].register_forward_hook(hook)

    HS, RAW, NORMED = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, txt in enumerate(T):
            ids = tok(txt, return_tensors="pt", truncation=True,
                      max_length=a.maxlen).input_ids
            out = model(input_ids=ids, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 0)[:, 0]        # [nH, seq, d]
            HS.append(hs.float().mean(1).numpy())               # [nH, d] pooled
            r = raw_last["h"][0]                                # [seq, d] raw last block
            RAW.append(r.float().mean(0).numpy())
            NORMED.append(core.norm(r).float().mean(0).numpy() if hasattr(core, "norm")
                          else r.float().mean(0).numpy())
            if i % 25 == 0:
                print(f"  fwd {i}/{len(T)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    X = np.stack(HS)                                            # [N, nH, d]
    Xr, Xn = np.stack(RAW), np.stack(NORMED)
    np.savez_compressed(a.acts_out, X=X.astype(np.float16), Xr=Xr.astype(np.float16),
                        Xn=Xn.astype(np.float16), y=y, prob=prob, kind=kind)
    print(f"[acts] saved {X.shape} -> {a.acts_out}", flush=True)

    res = {"n_mutants": len(T), "n_hidden_entries": int(X.shape[1]), "dtype": str(dtype)}
    curve = []
    for L in range(X.shape[1]):
        curve.append(within_auc(y, probe(X[:, L], y, prob), prob))
    res["curve"] = [round(float(c), 4) for c in curve]
    print("depth curve: " + " ".join(f"{c:.2f}" for c in curve), flush=True)
    print(f"last 5 hidden-state entries: {[round(c,3) for c in curve[-5:]]}", flush=True)
    for tag, XX in (("raw_last_block", Xr), ("normed_last_block", Xn)):
        A, nm, ns, z = auc_with_null(y, probe(XX, y, prob), prob)
        res[tag] = {"auc": round(A, 4), "null": [nm, ns], "z": round(z, 2)}
        print(f"{tag}: AUC={A:.3f} null={nm:.3f}±{ns:.3f} z={z:+.2f}", flush=True)
    pk = int(np.nanargmax(curve))
    A, nm, ns, z = auc_with_null(y, probe(X[:, pk], y, prob), prob)
    res["peak"] = {"layer": pk, "auc": round(A, 4), "z": round(z, 2)}
    print(f"peak: L{pk} AUC={A:.3f} z={z:+.2f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"RUNF_DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
