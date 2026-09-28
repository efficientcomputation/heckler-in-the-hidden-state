#!/usr/bin/env python
"""Clean forward capture for the J-space neuro study (one diffusion code model).

Runs ONE clean forward over the study mutant set and captures, per layer:
  - pooled[ex, L, D]      mean over non-pad positions (the study's pooled read input)
  - meannorm[ex, L]       mean per-position L2 norm per example (the mu perturbation scale)
  - PP_peak[Npos, D]      per-position activations at the pinned peak layer + owner map
This is EVERYTHING needed to reconstruct the study's mu perturbation ANALYTICALLY at
ANY layer, because the mu read "perturb layer L, read at L" is:
      h'_pos = h_pos + m * meannorm[ex,L] * v[ex]      (v = one unit dir per example)
  ->  pooled'[ex,L] = pooled[ex,L] + m * meannorm[ex,L] * v[ex]      (v const over positions)
i.e. a pure function of the clean per-layer pooled activation + mean row norm.  Perturb-at-L /
read-at-L is unaffected by downstream layers, so the analytic form is EXACT up to bf16 rounding
of the per-position add (which averages out under pooling).  This script VALIDATES that claim by
running one REAL perturbed forward with a CONTROLLED eps and asserting the analytic pooled read
matches the real pooled read (AUC + max abs diff).

Reuses pert_ext.py machinery VERBATIM (auc/make_wauc/build_4d/center/fresh_scores + the exact
offset calibration).  bf16 mandatory. 4D bias mask asserted.
"""
import os, sys, json, time, argparse
NTH = int(os.environ.get("JP_THREADS", "16"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(NTH)
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/scratch/jspace_pert")
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(NTH)
from transformers import AutoModel, AutoTokenizer
import pert_ext as PE                     # verbatim machinery: auc, make_wauc, build_4d, center, fresh_scores

DT = torch.bfloat16


def clean_encode(model, tok, texts, bs, peak, maxlen=640):
    """Return pooled[n,nL,D], meannorm[n,nL], PP_peak[Npos,D], owner[Npos], seqlens[n]."""
    pooled_out, mn_out, pp_out, owner_out, seqlens = [], [], [], [], []
    ex = 0
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen)
        am = enc.attention_mask
        seqlens += am.sum(1).tolist()
        m4 = PE.build_4d(am)
        assert m4.dim() == 4, "attention bias must be 4D (batch,1,seq,seq)"
        PE._STATE["ptype"] = None; PE._STATE["mag"] = 0.0; PE._STATE["am"] = am
        with torch.no_grad():
            o = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        h = torch.stack(o.hidden_states, 1).float()          # (B,nL,S,D)
        mm = am[:, None, :, None].float()                    # (B,1,S,1)
        pooled_out.append(((h * mm).sum(2) / mm.sum(2)).numpy())   # (B,nL,D)
        rown = h.norm(dim=-1)                                 # (B,nL,S)
        amb2 = am[:, None, :].float()                        # (B,1,S)
        mn_out.append(((rown * amb2).sum(2) / amb2.sum(2)).numpy())  # (B,nL)
        hp = h[:, peak]                                       # (B,S,D)
        amb = am.bool()
        for b in range(hp.size(0)):
            sel = hp[b][amb[b]].numpy(); pp_out.append(sel)
            owner_out.append(np.full(sel.shape[0], ex, np.int64)); ex += 1
        del o, h
    return (np.concatenate(pooled_out), np.concatenate(mn_out).astype(np.float32),
            np.concatenate(pp_out).astype(np.float32), np.concatenate(owner_out),
            np.array(seqlens))


# ---- controlled-eps real perturbation hook (validation only) ----
_VAL = {"eps": None, "ptr": 0, "mag": 0.0}


def val_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    od = h.dtype; hf = h.float()
    am = PE._STATE["am"]; B, S, Dm = hf.shape
    keepf = am.bool().float()[..., None]
    rown = hf.norm(dim=-1, keepdim=True)
    cnt = keepf.sum(1, keepdim=True).clamp_min(1)
    meannorm = (rown * keepf).sum(1, keepdim=True) / cnt              # (B,1,1)
    eps = torch.from_numpy(_VAL["eps"][_VAL["ptr"]:_VAL["ptr"] + B]).float()[:, None, :]  # (B,1,D)
    _VAL["ptr"] += B
    hf2 = hf + _VAL["mag"] * meannorm * eps
    hf2 = torch.where(am.bool()[..., None], hf2, hf)
    hn = hf2.to(od)
    assert not torch.isnan(hn).any()
    return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--peak_layer", type=int, required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_mut", type=int, default=0)
    ap.add_argument("--outdir", default="/scratch/jspace_neuro/dumps")
    a = ap.parse_args()
    peak = a.peak_layer

    def load():
        return round(os.getloadavg()[0], 1)

    D = json.load(open("/scratch/jspace_pert/mutants.json"))
    T = D["T"]; y = np.array(D["Y"], float); prob = np.array(D["P"]); kind = np.array(D["K"])
    if a.n_mut and len(T) > a.n_mut:              # keep WHOLE problems (identical to study)
        keep = []
        for p in list(dict.fromkeys(prob.tolist())):
            idx = [i for i in range(len(T)) if prob[i] == p]
            if keep and len(keep) + len(idx) > a.n_mut:
                break
            keep += idx
        T = [T[i] for i in keep]; y = y[keep]; prob = prob[keep]; kind = kind[keep]
    uq = np.unique(prob)
    wauc = PE.make_wauc(y, prob, kind, uq)
    print(f"[{a.name}] mutants={len(T)} problems={len(uq)} load={load()} threads={NTH} peak=L{peak}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModel.from_pretrained(a.model, torch_dtype=DT, trust_remote_code=True).eval()
    pc = int(sum(p.numel() for p in model.parameters()))
    nL_cfg = model.config.num_hidden_layers; hidden = model.config.hidden_size
    layers = lname = None
    for nm, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == nL_cfg:
            layers = mod; lname = nm; break
    if layers is None:
        raise SystemExit(f"no decoder ModuleList len {nL_cfg}")
    print(f"[{a.name}] params={pc/1e9:.2f}B hidden={hidden} layers={nL_cfg} module='{lname}' "
          f"load_in={time.time()-t0:.0f}s", flush=True)

    # ---- clean capture ----
    tc = time.time()
    pooled, meannorm, PP, owner, seqlens = clean_encode(model, tok, T, a.bs, peak)
    nL_h = pooled.shape[1]
    if not (0 <= peak < nL_h):
        raise SystemExit(f"peak {peak} out of range 0..{nL_h-1}")
    prob_long = prob[owner]; y_long = y[owner]
    # per-problem-centered pooled -> clean layer curve (study read) + per-layer clean direction
    Xc = pooled.astype(np.float32).copy()
    for p in uq:
        m = prob == p; Xc[m] -= Xc[m].mean(0, keepdims=True)
    curve = [wauc(PE.fresh_scores(Xc, y, prob, uq, L))[0] for L in range(nL_h)]
    auto_peak = int(np.nanargmax(curve))
    dclean = np.zeros((nL_h, hidden), np.float32)
    for L in range(nL_h):
        d = Xc[y == 1, L].mean(0) - Xc[y == 0, L].mean(0)
        dclean[L] = d / max(np.linalg.norm(d), 1e-9)
    print(f"[{a.name}] CLEAN peak=L{peak} auc={curve[peak]:.3f} (auto=L{auto_peak} "
          f"{curve[auto_peak]:.3f}) L0={curve[0]:.3f} encode={time.time()-tc:.0f}s "
          f"seq_med={int(np.median(seqlens))} npos={len(owner)} "
          f"curve=" + " ".join(f"{c:.2f}" for c in curve), flush=True)

    # ---- offset calibration (verbatim from pert_ext) ----
    def _calib():
        c0 = peak - 1
        def th(mod, inp, out):
            return (out[0] + 100.0,) + tuple(out[1:]) if isinstance(out, tuple) else out + 100.0
        enc = tok(T[:4], return_tensors="pt", padding=True, truncation=True, max_length=640)
        m4 = PE.build_4d(enc.attention_mask); PE._STATE["ptype"] = None
        with torch.no_grad():
            b0 = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        base = [hh.float() for hh in b0.hidden_states]
        hnd = layers[c0].register_forward_hook(th)
        with torch.no_grad():
            b1 = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        hnd.remove()
        for k in range(len(base)):
            if (b1.hidden_states[k].float() - base[k]).abs().max().item() > 1e-3:
                return k, c0
        raise SystemExit("calib: nothing moved")
    F0, c0 = _calib()
    hook_layer = 2 * peak - 1 - F0
    print(f"[{a.name}] offset-calib +100@L[{c0}] -> hidden[{F0}] (offset {F0-c0}); hook_layer={hook_layer}", flush=True)

    # ---- VALIDATION: real forward w/ controlled eps at m=0.8 vs analytic ----
    mag = 0.8
    rng = np.random.default_rng(12345)
    eps = rng.standard_normal((len(T), hidden)).astype(np.float32)
    eps /= np.linalg.norm(eps, axis=1, keepdims=True)
    _VAL["eps"] = eps; _VAL["ptr"] = 0; _VAL["mag"] = mag
    hnd = layers[hook_layer].register_forward_hook(val_hook)
    real_pool = []
    for i in range(0, len(T), a.bs):
        enc = tok(T[i:i + a.bs], return_tensors="pt", padding=True, truncation=True, max_length=640)
        am = enc.attention_mask; PE._STATE["am"] = am
        m4 = PE.build_4d(am)
        with torch.no_grad():
            o = model(input_ids=enc.input_ids, attention_mask=m4, output_hidden_states=True)
        h = torch.stack(o.hidden_states, 1).float()
        mm = am[:, None, :, None].float()
        real_pool.append(((h * mm).sum(2) / mm.sum(2))[:, peak].numpy())
        del o, h
    hnd.remove()
    real_peak = np.concatenate(real_pool).astype(np.float32)                 # (n,D) real perturbed pooled@peak
    ana_peak = pooled[:, peak] + mag * meannorm[:, peak][:, None] * eps        # analytic
    # AUC of both, per-problem-centered refit
    def peak_auc(Xpeak):
        Xc1 = Xpeak.astype(np.float32).copy()
        for p in uq:
            m = prob == p; Xc1[m] -= Xc1[m].mean(0, keepdims=True)
        # emulate a single-layer refit: reuse fresh_scores by faking an (n,1,D) stack
        st = Xc1[:, None, :]
        return wauc(PE.fresh_scores(st, y, prob, uq, 0))[0]
    a_real = peak_auc(real_peak); a_ana = peak_auc(ana_peak)
    maxabs = float(np.abs(real_peak - ana_peak).max())
    relerr = maxabs / (float(np.abs(real_peak).mean()) + 1e-9)
    print(f"[{a.name}] VALIDATE m=0.8: real_auc={a_real:.4f} ana_auc={a_ana:.4f} "
          f"|Δauc|={abs(a_real-a_ana):.4f} maxabs={maxabs:.4g} relerr={relerr:.4g}", flush=True)
    if abs(a_real - a_ana) > 0.02:
        raise SystemExit(f"ANALYTIC MISMATCH |Δauc|={abs(a_real-a_ana):.4f} > 0.02 -> do not trust analytic")

    # ---- save ----
    os.makedirs(a.outdir, exist_ok=True)
    out = f"{a.outdir}/clean_{a.name}.npz"
    np.savez_compressed(out, pooled=pooled.astype(np.float32), meannorm=meannorm,
                        PP_peak=PP, owner=owner, y=y, prob=prob, kind=kind,
                        dclean=dclean, curve=np.array(curve, np.float32),
                        peak=peak, hidden=hidden, nL=nL_h, param_count=pc,
                        auto_peak=auto_peak, seq_med=int(np.median(seqlens)),
                        n_pairs=len(T), n_problems=len(uq),
                        val_real_auc=a_real, val_ana_auc=a_ana, val_maxabs=maxabs)
    print(f"[{a.name}] saved {out}  pooled={pooled.shape} PP={PP.shape} DONE", flush=True)


if __name__ == "__main__":
    main()
