#!/usr/bin/env python
"""J-space perturbation study -- EXTENSIONS 1 & 2 (forward-only, one diffusion code model).

Reuses the pert_run.py machinery VERBATIM for the POOLED read (auc/wauc, build_4d, offset-calibrated
peak-layer residual hook, per-problem centering, 5-fold OOF refit + fixed clean-direction read,
permutation null, L0-at-chance sanity, matched-norm random baseline, bf16 + asserted 4D bias mask).
Adds the perturbation FAMILIES and a second READ:

EXT1  Gaussian decomposition at the peak layer, measured with TWO reads:
  mu     : ONE per-example random unit dir (correlated across positions), added m*mean||h||.
           == the study's original "noise" perturbation (mu-style mean shift).
  sigma  : per-position i.i.d. zero-mean Gaussian, std sigma*||h_pos|| (independent draw at every
           position). sigma largely CANCELS under mean pooling.
  reads  : auc_pooled  = mean-pooled residual over the scored region (the study default), and
           auc_perpos  = per-position read: score EVERY scored position on the correctness
           direction and rank pass-positions vs fail-positions within each problem (example-level
           permutation null). NOT averaged before scoring, so it exposes sigma's LOCAL corruption
           even when the pooled read survives. L0-clean (within-problem mutants are near-identical
           programs, so per-position token embeddings do NOT leak the label -- unlike a last-token
           read, which does).

EXT2  causal-link disruption at the FEATURE level (alt reading of "within the latent space"):
  feat   : zero a random fraction f of the hidden DIMENSIONS (same mask across positions per
           example). Severs cross-FEATURE structure (vs. the attention-connectivity `causal`
           variant which severs cross-POSITION structure). Written to __causal_feat.json so it
           overlays the existing __causal.json.

Every AUC (pooled refit, per-position refit, fixed-direction, kind-matched) carries a within-problem
label-shuffle null (mean/std/z). Inert-hook guard: at every mag>0 require min(pooled,perpos) drift
< 0.999 (pooled alone FALSE-fires on sigma -- that IS the finding: sigma is invisible to the pooled
read). Peak layer is PINNED (--peak_layer) to the study's value so the new curves overlay the old.
"""
import os, sys, json, math, argparse, time
NTH = int(os.environ.get("JP_THREADS", "16"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(NTH)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np                                   # noqa: E402
import torch, torch.nn as nn                         # noqa: E402
torch.set_num_threads(NTH)
from transformers import AutoModel, AutoTokenizer    # noqa: E402

DT = torch.bfloat16
NEG = torch.finfo(DT).min
_STATE = {"ptype": None, "mag": 0.0, "am": None, "gen": None}

MU = [0.05, 0.1, 0.2, 0.4, 0.8]
SIGMA = [0.05, 0.1, 0.2, 0.4, 0.8]
FEAT = [0.1, 0.25, 0.5, 0.75]


# ---------- AUC / weighted-AUC (identical to pert_run.py) ----------
def auc(yy, ss):
    o = np.argsort(ss)
    r = np.empty(len(ss), float); r[o] = np.arange(1, len(ss) + 1)
    n1, n0 = (yy == 1).sum(), (yy == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_wauc(y, prob, kind, uq):
    def wauc(score, seed=None, same_kind=False):
        rg = np.random.default_rng(seed) if seed is not None else None
        yy = y.copy()
        if rg is not None:
            for p in uq:
                m = prob == p
                yy[m] = rg.permutation(yy[m])
        v = []
        for p in uq:
            idx = np.flatnonzero(prob == p)
            groups = ([np.flatnonzero((prob == p) & (kind == kk)) for kk in set(kind[idx])]
                      if same_kind else [idx])
            for m in groups:
                if len(m) < 2 or len(set(yy[m])) < 2:
                    continue
                v.append(auc(yy[m], score[m]))
        return (float(np.nanmean(v)), len(v)) if v else (np.nan, 0)
    return wauc


# ---------- per-position weighted-AUC (rank pass-positions vs fail-positions within problem) ----------
def make_wauc_pp(y_ex, prob_ex, owner, uq):
    """owner[k] = example index of position k. Null permutes labels at the EXAMPLE level within
    problem (preserving each example's within-position correlation), then propagates to positions."""
    prob_long = prob_ex[owner]
    grp = {p: np.flatnonzero(prob_long == p) for p in uq}

    def wauc_pp(scores_long, seed=None):
        yy = y_ex.copy()
        if seed is not None:
            rg = np.random.default_rng(seed)
            for p in uq:
                m = prob_ex == p
                yy[m] = rg.permutation(yy[m])
        y_long = yy[owner]
        v = []
        for p in uq:
            m = grp[p]
            if len(m) < 2 or len(set(y_long[m])) < 2:
                continue
            v.append(auc(y_long[m], scores_long[m]))
        return float(np.nanmean(v)) if v else np.nan
    return wauc_pp


# ---------- attention mask (identical) ----------
def build_4d(am):
    bsz, slen = am.shape
    m = torch.zeros(bsz, 1, slen, slen, dtype=DT)
    m.masked_fill_(~am.bool()[:, None, None, :], NEG)
    return m


# ---------- peak-layer residual perturbation (EXTENDED) ----------
def perturb_hook(module, inp, out):
    pt = _STATE["ptype"]
    if pt not in ("mu", "sigma", "feat") or _STATE["mag"] == 0:
        return out
    h = out[0] if isinstance(out, tuple) else out
    od = h.dtype
    hf = h.float()                                   # (B,S,D)
    am = _STATE["am"]; gen = _STATE["gen"]; mag = _STATE["mag"]
    B, S, Dm = hf.shape
    keepf = am.bool().float()[..., None]             # (B,S,1)
    rown = hf.norm(dim=-1, keepdim=True)             # (B,S,1) per-position norm
    cnt = keepf.sum(1, keepdim=True).clamp_min(1)    # (B,1,1)
    meannorm = (rown * keepf).sum(1, keepdim=True) / cnt   # (B,1,1) mean row norm/example
    if pt == "mu":                                   # per-example CORRELATED mean shift
        eps = torch.randn(B, 1, Dm, generator=gen)   # ONE dir per example
        eps = eps / eps.norm(dim=-1, keepdim=True)
        hf2 = hf + mag * meannorm * eps               # broadcast over positions
    elif pt == "sigma":                              # per-position I.I.D. spread
        eps = torch.randn(B, S, Dm, generator=gen)   # fresh dir at EVERY position
        eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        hf2 = hf + mag * rown * eps                   # std sigma*||h_pos||, independent per pos
    else:                                            # feat: zero a fraction f of hidden dims
        keepdim = (torch.rand(B, 1, Dm, generator=gen) >= mag).float()  # same mask over positions
        hf2 = hf * keepdim
    hf2 = torch.where(am.bool()[..., None], hf2, hf)
    hn = hf2.to(od)
    assert not torch.isnan(hn).any(), "NaN in perturbed residual"
    if isinstance(out, tuple):
        return (hn,) + tuple(out[1:])
    return hn


def encode(model, tok, texts, bs, pp_layer, maxlen=640):
    """Return (pooled[n,nL,d], PP[Npos,d]@pp_layer, owner[Npos], seqlens[n], secs).
    pooled = mean over non-pad positions (study default); PP = every non-pad position's activation
    at hidden_states[pp_layer], concatenated across examples; owner maps each position to its example."""
    pooled_out, pp_out, owner_out, seqlens = [], [], [], []
    t0 = time.time()
    ex = 0
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen)
        am = enc.attention_mask
        seqlens += am.sum(1).tolist()
        m4 = build_4d(am)
        assert m4.dim() == 4, "attention bias must be 4D (batch,1,seq,seq)"
        _STATE["am"] = am
        with torch.no_grad():
            o = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        h = torch.stack(o.hidden_states, 1).float()          # (B,nL,S,D)
        mm = am[:, None, :, None].float()
        pooled_out.append(((h * mm).sum(2) / mm.sum(2)).numpy())
        hp = h[:, pp_layer]                                  # (B,S,D) at pinned peak
        amb = am.bool()
        for b in range(hp.size(0)):
            sel = hp[b][amb[b]].numpy()                      # (S_b, D) scored positions
            pp_out.append(sel)
            owner_out.append(np.full(sel.shape[0], ex, np.int64))
            ex += 1
        del o, h
    return (np.concatenate(pooled_out), np.concatenate(pp_out).astype(np.float32),
            np.concatenate(owner_out), np.array(seqlens), time.time() - t0)


def center(X, prob, uq):
    Xc = X.astype(np.float32).copy()
    for p in uq:
        m = prob == p
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    return Xc


def center_pp(PP, prob_long, uq):
    C = PP.astype(np.float32).copy()
    for p in uq:
        m = prob_long == p
        C[m] -= C[m].mean(0, keepdims=True)
    return C


def fresh_scores(Xc, y, prob, uq, L, folds=5, seed=0):
    rr = np.random.default_rng(seed)
    order = uq.copy(); rr.shuffle(order)
    S = np.full(len(y), np.nan, np.float32)
    for te_p in np.array_split(order, folds):
        te = np.isin(prob, te_p); tr = ~te
        d = Xc[tr & (y == 1), L].mean(0) - Xc[tr & (y == 0), L].mean(0)
        d /= max(np.linalg.norm(d), 1e-9)
        S[te] = Xc[te, L] @ d
    return S


def dir_pp(Cpp, y_long, prob_long, uq, folds=5, seed=0):
    """5-fold OOF per-position difference-of-means direction; returns per-position scores.
    Folds are split by PROBLEM (same partition granularity as the pooled read)."""
    rr = np.random.default_rng(seed)
    order = uq.copy(); rr.shuffle(order)
    S = np.full(len(y_long), np.nan, np.float32)
    for te_p in np.array_split(order, folds):
        te = np.isin(prob_long, te_p); tr = ~te
        d = Cpp[tr & (y_long == 1)].mean(0) - Cpp[tr & (y_long == 0)].mean(0)
        d /= max(np.linalg.norm(d), 1e-9)
        S[te] = Cpp[te] @ d
    return S


def load_curve(Xc, y, prob, uq, wauc, nL):
    return [wauc(fresh_scores(Xc, y, prob, uq, L))[0] for L in range(nL)]


def cos_rows(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--peak_layer", type=int, required=True,
                    help="PINNED peak layer (hidden_states index) to overlay the study")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--mu", default=",".join(map(str, MU)))
    ap.add_argument("--sigma", default=",".join(map(str, SIGMA)))
    ap.add_argument("--feat", default=",".join(map(str, FEAT)))
    ap.add_argument("--ptypes", default="mu,sigma,feat")
    ap.add_argument("--n_mut", type=int, default=0, help="cap #mutants (keep WHOLE problems); 0=all")
    ap.add_argument("--outdir", default="/scratch/jspace_pert/results")
    a = ap.parse_args()
    mu = [float(x) for x in a.mu.split(",") if x != ""]
    sigma = [float(x) for x in a.sigma.split(",") if x != ""]
    feat = [float(x) for x in a.feat.split(",") if x != ""]
    ptypes = [p for p in a.ptypes.split(",") if p]
    peakL = a.peak_layer

    def loadavg():
        return round(os.getloadavg()[0], 1)

    D = json.load(open("/scratch/jspace_pert/mutants.json"))
    T = D["T"]; y = np.array(D["Y"], float); prob = np.array(D["P"]); kind = np.array(D["K"])
    if a.n_mut and len(T) > a.n_mut:            # keep WHOLE problems up to n_mut (same as study)
        keep = []
        for p in list(dict.fromkeys(prob.tolist())):
            idx = [i for i in range(len(T)) if prob[i] == p]
            if keep and len(keep) + len(idx) > a.n_mut:
                break
            keep += idx
        T = [T[i] for i in keep]; y = y[keep]; prob = prob[keep]; kind = kind[keep]
        print(f"[{a.name}] n_mut cap -> {len(T)} mutants / {len(set(prob))} problems", flush=True)
    uq = np.unique(prob)
    wauc = make_wauc(y, prob, kind, uq)
    print(f"[{a.name}] mutants={len(T)} problems={len(uq)} load={loadavg()} threads={NTH} "
          f"pinned_peak=L{peakL}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModel.from_pretrained(a.model, torch_dtype=DT, trust_remote_code=True).eval()
    param_count = int(sum(p.numel() for p in model.parameters()))
    nL_cfg = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    layers = None; lname = None
    for nm, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == nL_cfg:
            layers = mod; lname = nm; break
    if layers is None:
        raise SystemExit(f"could not find decoder ModuleList of len {nL_cfg}")
    print(f"[{a.name}] params={param_count/1e9:.2f}B hidden={hidden} layers={nL_cfg} "
          f"module='{lname}' loaded_in={time.time()-t0:.0f}s", flush=True)

    # ---- CLEAN encode (pooled all layers + per-position at pinned peak) ----
    _STATE["ptype"] = None; _STATE["mag"] = 0.0
    Xclean, PPclean, owner, seqlens, dt = encode(model, tok, T, a.bs, peakL)
    nL_h = Xclean.shape[1]
    if not (0 <= peakL < nL_h):
        raise SystemExit(f"peak_layer {peakL} out of range 0..{nL_h-1}")
    prob_long = prob[owner]; y_long = y[owner]
    wauc_pp = make_wauc_pp(y, prob, owner, uq)
    Xc_clean = center(Xclean, prob, uq)
    curve = load_curve(Xc_clean, y, prob, uq, wauc, nL_h)
    auto_peak = int(np.nanargmax(curve))
    auc0 = curve[peakL]; auc_L0 = curve[0]
    seq_med = int(np.median(seqlens))
    print(f"[{a.name}] CLEAN pinned_peak=L{peakL} auc={auc0:.3f} (auto-argmax=L{auto_peak} "
          f"auc={curve[auto_peak]:.3f}) L0={auc_L0:.3f} encode={dt:.0f}s seq_med={seq_med} "
          f"npos={len(owner)} curve=" + " ".join(f"{c:.2f}" for c in curve), flush=True)
    if abs(auto_peak - peakL) > 2:
        print(f"[{a.name}] WARNING pinned peak L{peakL} far from auto L{auto_peak}", flush=True)
    if auc_L0 > 0.60:
        print(f"[{a.name}] WARNING L0 pooled AUC {auc_L0:.3f} > 0.60 possible leak", flush=True)
    # clean per-position read at peak
    Cpp_clean = center_pp(PPclean, prob_long, uq)
    Spp_clean = dir_pp(Cpp_clean, y_long, prob_long, uq)
    auc_pp0 = wauc_pp(Spp_clean)
    # per-position L0 leak check (token embeddings only, no forward -> cheap). Must be ~chance:
    # unlike a last-token read, the expanded per-position read should NOT leak label identity
    # because within-problem mutants are near-identical programs.
    emb = model.get_input_embeddings()
    pp0, own0, exq = [], [], 0
    for i in range(0, len(T), a.bs):
        enc = tok(T[i:i + a.bs], return_tensors="pt", padding=True, truncation=True, max_length=640)
        amb = enc.attention_mask.bool()
        with torch.no_grad():
            e = emb(enc.input_ids).float()
        for b in range(e.size(0)):
            sel = e[b][amb[b]].numpy(); pp0.append(sel)
            own0.append(np.full(len(sel), exq, np.int64)); exq += 1
    PP0 = np.concatenate(pp0).astype(np.float32); own0 = np.concatenate(own0)
    assert np.array_equal(own0, owner), "L0 per-position owner map mismatch"
    Spp_L0 = dir_pp(center_pp(PP0, prob_long, uq), y_long, prob_long, uq)
    auc_pp_L0 = wauc_pp(Spp_L0)
    del PP0, pp0
    print(f"[{a.name}] CLEAN per-position auc@peak={auc_pp0:.3f} auc@L0={auc_pp_L0:.3f} "
          f"(L0 must be ~chance)", flush=True)
    if auc_pp_L0 > 0.60:
        print(f"[{a.name}] WARNING per-position L0 AUC {auc_pp_L0:.3f} > 0.60 -> per-position "
              f"read leaks token identity", flush=True)

    # fixed clean directions at peak
    dclean = (Xc_clean[y == 1, peakL].mean(0) - Xc_clean[y == 0, peakL].mean(0))
    dclean = dclean / max(np.linalg.norm(dclean), 1e-9)
    dclean_pp = (Cpp_clean[y_long == 1].mean(0) - Cpp_clean[y_long == 0].mean(0))
    dclean_pp = dclean_pp / max(np.linalg.norm(dclean_pp), 1e-9)

    def metrics(Xc, X, PP, seed_null_base):
        # ---- pooled read (study default) ----
        S = fresh_scores(Xc, y, prob, uq, peakL)
        a_fresh = wauc(S)[0]
        nl = np.array([wauc(S, seed=seed_null_base + i)[0] for i in range(a.n_perm)])
        z = (a_fresh - nl.mean()) / (nl.std() + 1e-9)
        ak, ng = wauc(S, same_kind=True)
        nk = np.array([wauc(S, seed=seed_null_base + 500 + i, same_kind=True)[0]
                       for i in range(a.n_perm)])
        zk = (ak - nk.mean()) / (nk.std() + 1e-9)
        sf = Xc[:, peakL] @ dclean
        a_fix = wauc(sf)[0]
        nf = np.array([wauc(sf, seed=seed_null_base + 800 + i)[0] for i in range(a.n_perm)])
        zf = (a_fix - nf.mean()) / (nf.std() + 1e-9)
        drift = float(np.mean(cos_rows(X[:, peakL], Xclean[:, peakL])))
        # ---- per-position read (expanded, L0-clean) ----
        Cpp = center_pp(PP, prob_long, uq)
        Sp = dir_pp(Cpp, y_long, prob_long, uq)
        a_pp = wauc_pp(Sp)
        npp = np.array([wauc_pp(Sp, seed=seed_null_base + 1200 + i) for i in range(a.n_perm)])
        zp = (a_pp - npp.mean()) / (npp.std() + 1e-9)
        sfp = Cpp @ dclean_pp
        a_pp_fix = wauc_pp(sfp)
        nfp = np.array([wauc_pp(sfp, seed=seed_null_base + 1600 + i) for i in range(a.n_perm)])
        zpf = (a_pp_fix - nfp.mean()) / (nfp.std() + 1e-9)
        drift_pp = float(np.mean(cos_rows(PP, PPclean)))   # mean over ALL scored positions
        return dict(auc=a_fresh, auc_null_mean=float(nl.mean()), auc_null_std=float(nl.std()),
                    auc_z=float(z), auc_kind=ak, auc_kind_z=float(zk), n_kind_groups=ng,
                    auc_fixed=a_fix, auc_fixed_z=float(zf), drift_cosine=drift,
                    auc_pooled=a_fresh,
                    auc_perpos=a_pp, auc_perpos_null_mean=float(npp.mean()),
                    auc_perpos_null_std=float(npp.std()), auc_perpos_z=float(zp),
                    auc_perpos_fixed=a_pp_fix, auc_perpos_fixed_z=float(zpf),
                    drift_perpos=drift_pp)

    clean_m = metrics(Xc_clean, Xclean, PPclean, 500)
    base_row = dict(model=a.name, param_count=param_count, hidden_size=hidden,
                    peak_layer=peakL, auto_peak_layer=auto_peak, seq_len=seq_med,
                    n_pairs=len(T), n_problems=len(uq), n_positions=int(len(owner)),
                    auc_L0=float(auc_L0), auc_perpos_L0=float(auc_pp_L0), notes="")

    def row(ptype, mag, m):
        r = dict(base_row); r.update(m)
        r.update(ptype=ptype, magnitude=mag)
        return r

    # ---- offset calibration (arch-dependent; identical to pert_run.py) ----
    def _calib():
        c0 = peakL - 1
        def th(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + 100.0,) + tuple(out[1:])
            return out + 100.0
        enc = tok(T[:4], return_tensors="pt", padding=True, truncation=True, max_length=640)
        m4 = build_4d(enc.attention_mask)
        _STATE["ptype"] = None
        with torch.no_grad():
            b0 = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        base = [h.float() for h in b0.hidden_states]
        hh = layers[c0].register_forward_hook(th)
        with torch.no_grad():
            b1 = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        hh.remove()
        for k in range(len(base)):
            if (b1.hidden_states[k].float() - base[k]).abs().max().item() > 1e-3:
                return k, c0
        raise SystemExit("calibration: no hidden_states changed under +100 hook")
    F0, c0 = _calib()
    hook_layer = 2 * peakL - 1 - F0
    if not (0 <= hook_layer < nL_cfg):
        raise SystemExit(f"calibrated hook_layer {hook_layer} out of range (peakL={peakL} F0={F0})")
    print(f"[{a.name}] offset-calib: +100@layer[{c0}] -> first hidden[{F0}] (offset {F0-c0}); "
          f"perturb hook_layer={hook_layer} so read hidden[{peakL}] is affected", flush=True)
    base_row["hook_layer"] = hook_layer
    handle = layers[hook_layer].register_forward_hook(perturb_hook)

    results = {"mu": [], "sigma": [], "causal_feat": []}
    for pt in results:
        r = row(pt, 0.0, clean_m); r["notes"] = "clean baseline"
        results[pt].append(r)
    mu_curve = {0.0: clean_m["auc"]}   # matched-norm random-direction control (pooled rel-disp)

    def run(ptype, mag):
        _STATE["ptype"] = ptype; _STATE["mag"] = mag
        _STATE["gen"] = torch.Generator().manual_seed(
            {"mu": 100, "sigma": 250, "feat": 400}[ptype] + int(round(mag * 1000)))
        X, PP, own2, _, dt = encode(model, tok, T, a.bs, peakL)
        assert np.array_equal(own2, owner), "position owner map drifted between encodes"
        Xc = center(X, prob, uq)
        m = metrics(Xc, X, PP, 1000 + int(round(mag * 1000)))
        # inert-hook guard: pooled alone FALSE-fires on sigma -> require min(pooled,perpos).
        if mag > 0 and min(m["drift_cosine"], m["drift_perpos"]) >= 0.999:
            raise SystemExit(
                f"HOOK INERT: {ptype} m={mag} drift_pooled={m['drift_cosine']:.5f} "
                f"drift_perpos={m['drift_perpos']:.5f} both>=0.999 -> perturbation not applied")
        print(f"[{a.name}] {ptype} m={mag}: pooled={m['auc']:.3f} z{m['auc_z']:+.1f} "
              f"| perpos={m['auc_perpos']:.3f} z{m['auc_perpos_z']:+.1f} "
              f"| fixed={m['auc_fixed']:.3f} | drift_pool={m['drift_cosine']:.3f} "
              f"drift_pp={m['drift_perpos']:.3f} enc={dt:.0f}s load={loadavg()}", flush=True)
        return m

    os.makedirs(a.outdir, exist_ok=True)

    def write(key):
        json.dump(results[key], open(f"{a.outdir}/{a.name}__{key}.json", "w"), indent=2)
        print(f"[{a.name}] wrote {a.name}__{key}.json ({len(results[key])} rows)", flush=True)

    def ctrl_auc(rel):                       # matched-norm random-direction baseline (mu curve)
        xs = sorted(mu_curve); ys = [mu_curve[x] for x in xs]
        return float(np.interp(rel, xs, ys))

    # EXT1a: mu-shift (also populates the matched-norm control curve)
    if "mu" in ptypes:
        for m in mu:
            met = run("mu", m)
            mu_curve[m] = met["auc"]
            results["mu"].append(row("mu", m, met))
        write("mu")

    # EXT1b: sigma-spread
    if "sigma" in ptypes:
        for s in sigma:
            met = run("sigma", s)
            r = row("sigma", s, met)
            rel = float(np.sqrt(max(0.0, 2 - 2 * met["drift_cosine"])))
            r["disp_rel_norm"] = round(rel, 4)
            r["randctrl_auc"] = ctrl_auc(rel)
            r["notes"] = "per-position i.i.d.; randctrl=mu@pooled-disp-matched"
            results["sigma"].append(r)
        write("sigma")

    # EXT2: feature (hidden-dim) ablation -> overlays __causal.json
    if "feat" in ptypes:
        for f in feat:
            met = run("feat", f)
            rel = float(np.sqrt(max(0.0, 2 - 2 * met["drift_cosine"])))
            r = row("causal_feat", f, met)
            r["disp_rel_norm"] = round(rel, 4)
            r["randctrl_auc"] = ctrl_auc(rel)
            r["notes"] = ("hidden-DIM ablation (sever cross-feature structure); "
                          "randctrl=mu@drift-matched-disp")
            results["causal_feat"].append(r)
        write("causal_feat")

    handle.remove()
    print(f"[{a.name}] DONE ptypes={ptypes}", flush=True)


if __name__ == "__main__":
    main()
