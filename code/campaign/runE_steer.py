#!/usr/bin/env python
"""Run E: stronger steering tests (reviewer point 6), three arms in one batched harness.

Arms, all rows of ONE batched diffusion_generate per task (shared noise/numerics):
  base        no intervention
  +/-probe_g  probe direction, ALL response positions, every step (replication arm)
  +/-probe_c  probe direction, CONDITIONAL: only positions still masked at that step
  +/-rand_g   fresh random unit direction per task (matched norm) global  -> the null
  +rand_c     fresh random, conditional
  +/-train_g  direction TRAINED for causal effect (teacher-forced logprob of the
              reference under steering), global
  +train_c    trained direction, conditional
Metrics per row: pass (unit tests), compiles. Results saved per task (crash-safe).

Phases: --phase fit    fit probe dir at peak layer from runA shards -> dirs.pt
        --phase train  optimize v from probe init on train tasks -> dirs.pt (adds trained)
        --phase eval   sharded evaluation on held-out tasks
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/jlens/code"))
from transformers import AutoModel, AutoTokenizer
from jcode import DATASETS, build_prompt, extract_code, grade_task
import gradefix  # noqa: F401  fork-free grading

PEAK_L = 18
ALPHA = 0.4
MASK_ID = 151666


def load_model(dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained("apple/DiffuCoder-7B-cpGRPO", trust_remote_code=True)
    model = AutoModel.from_pretrained("apple/DiffuCoder-7B-cpGRPO", torch_dtype=dtype,
                                      trust_remote_code=True).to("cpu").eval()
    return tok, model


def fit_dir(out):
    rows = []
    for f in sorted(glob.glob(os.path.expanduser("~/jlens/out/runA/runA_shard*.pt"))):
        rows.extend(torch.load(f, weights_only=False))
    y = np.array([1.0 if r["pass"] else 0.0 for r in rows])
    prob = np.array([r["task_id"] for r in rows])
    X = np.stack([r["acts"][-1][PEAK_L].float().numpy() for r in rows]).astype(np.float32)
    for p in np.unique(prob):
        m = prob == p
        X[m] -= X[m].mean(0, keepdims=True)
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    d /= max(np.linalg.norm(d), 1e-9)
    torch.save({"probe": torch.tensor(d)}, out)
    print(f"FIT_DONE |d|=1 layer={PEAK_L} n={len(rows)} -> {out}", flush=True)


def train_dir(dirs_path, n_train=12, steps=25, lr=5e-3, threads=32):
    torch.set_num_threads(threads)
    tok, model = load_model()
    layers = model.model.layers
    D = torch.load(dirs_path, weights_only=False)
    v = D["probe"].clone().float().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    tasks = DATASETS["mbppplus"](n_train, 0)
    state = {"plen": 0, "v": None, "alpha": ALPHA}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if state["v"] is None:
            return o
        p = state["plen"]
        nrm = h[:, p:, :].detach().norm(dim=-1).mean()
        h = torch.cat([h[:, :p, :],
                       h[:, p:, :] + state["alpha"] * nrm * state["v"]], dim=1)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    hd = layers[PEAK_L].register_forward_hook(hook)
    enc_cache = []
    for t in tasks:
        prompt = build_prompt(tok, t)
        pids = tok(prompt, return_tensors="pt").input_ids
        ref = "```python\n" + t["code"] + "\n```"
        rids = tok(ref, return_tensors="pt").input_ids
        enc_cache.append((pids, rids))
    t0 = time.time()
    for s in range(steps):
        tot = 0.0
        opt.zero_grad()
        for pids, rids in enc_cache[:4] if s % 2 == 0 else enc_cache[4:8]:
            plen = pids.shape[1]
            full = torch.cat([pids, torch.full_like(rids, MASK_ID)], dim=1)
            state["plen"] = plen
            state["v"] = v / v.norm().clamp_min(1e-9)
            out = model(input_ids=full)
            lg = out.logits if hasattr(out, "logits") else None
            if lg is None:
                h = model.model.norm(out.last_hidden_state)
                lg = F.linear(h, model.get_output_embeddings().weight)
            lp = torch.log_softmax(lg[0, plen:].float(), -1)
            loss = -lp.gather(-1, rids[0][:, None]).mean() / 4
            loss.backward()
            tot += float(loss) * 4
        opt.step()
        print(f"  train step {s+1}/{steps} loss={tot/2:.4f} ({time.time()-t0:.0f}s)", flush=True)
    hd.remove()
    D["trained"] = (v / v.norm().clamp_min(1e-9)).detach()
    cos = float(F.cosine_similarity(D["trained"], D["probe"], dim=0))
    torch.save(D, dirs_path)
    print(f"TRAIN_DONE cos(trained,probe)={cos:.3f} -> {dirs_path}", flush=True)


def evaluate(dirs_path, shard, n_shards, n_tasks, skip, out_dir, threads=16,
             max_new=256, steps=32):
    torch.set_num_threads(threads)
    os.makedirs(os.path.expanduser(out_dir), exist_ok=True)
    tok, model = load_model()
    layers = model.model.layers
    D = torch.load(dirs_path, weights_only=False)
    vp, vt = D["probe"].float(), D["trained"].float()
    tasks = DATASETS["mbppplus"](n_tasks, skip)
    mine = [(i, t) for i, t in enumerate(tasks) if i % n_shards == shard]

    ARMS = [("base", None, 0.0, False), ("probe_g+", "p", +ALPHA, False),
            ("probe_g-", "p", -ALPHA, False), ("probe_c+", "p", +ALPHA, True),
            ("probe_c-", "p", -ALPHA, True), ("rand_g+", "r", +ALPHA, False),
            ("rand_g-", "r", -ALPHA, False), ("rand_c+", "r", +ALPHA, True),
            ("train_g+", "t", +ALPHA, False), ("train_g-", "t", -ALPHA, False),
            ("train_c+", "t", +ALPHA, True)]
    B = len(ARMS)
    state = {"plen": 0, "ids": None, "V": None}

    def embhook(m, inp):
        state["ids"] = inp[0]                       # current token ids [B, seq]
    eh = model.model.embed_tokens.register_forward_pre_hook(embhook)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if state["V"] is None or h.shape[0] != B:
            return o
        p = state["plen"]
        resp = h[:, p:, :]
        nrm = resp.detach().norm(dim=-1).mean(dim=1)              # [B]
        add = state["V"] * nrm[:, None]                           # [B, d] scaled dirs
        ids = state["ids"]
        for b, (_, kind, a, cond) in enumerate(ARMS):
            if kind is None or a == 0.0:
                continue
            if cond and ids is not None:
                mpos = (ids[b, p:] == MASK_ID).to(resp.dtype)[:, None]
                resp[b] = resp[b] + a * add[b][None, :] * mpos
            else:
                resp[b] = resp[b] + a * add[b][None, :]
        h = torch.cat([h[:, :p, :], resp], dim=1)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hh = layers[PEAK_L].register_forward_hook(hook)

    res_path = os.path.expanduser(f"{out_dir}/runE_shard{shard}.jsonl")
    fout = open(res_path, "a")
    t00 = time.time()
    for n_done, (ti, t) in enumerate(mine):
        t0 = time.time()
        rng = np.random.default_rng(10_000 + ti)
        vr = torch.tensor(rng.standard_normal(vp.shape[0]), dtype=torch.float32)
        vr /= vr.norm().clamp_min(1e-9)
        V = torch.stack([torch.zeros_like(vp) if k is None else
                         (vp if k == "p" else vr if k == "r" else vt)
                         for _, k, _, _ in ARMS])
        prompt = build_prompt(tok, t)
        inp = tok(prompt, return_tensors="pt")
        plen = inp.input_ids.shape[1]
        ids = inp.input_ids.repeat(B, 1)
        mask = inp.attention_mask.repeat(B, 1).bool()
        state["plen"], state["V"] = plen, V
        with torch.no_grad():
            out = model.diffusion_generate(
                ids, attention_mask=mask, max_new_tokens=max_new,
                steps=steps, temperature=0.0, alg="entropy", alg_temp=0.,
                output_history=False, return_dict_in_generate=True)
        state["V"] = None
        rec = {"task": ti}
        for b, (name, _, _, _) in enumerate(ARMS):
            txt = tok.decode(out.sequences[b][plen:], skip_special_tokens=False) \
                     .replace("<|dlm_pad|>", "").replace("<|im_end|>", "")
            code = extract_code(txt)
            ok = bool(grade_task(code, t))
            try:
                compile(code, "<s>", "exec"); comp = bool(code.strip())
            except Exception:
                comp = False
            rec[name] = {"pass": ok, "compile": comp}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"  [{n_done+1}/{len(mine)}] task {ti} "
              f"base={int(rec['base']['pass'])} {time.time()-t0:.0f}s "
              f"(elapsed {(time.time()-t00)/60:.1f}m)", flush=True)
    eh.remove(); hh.remove(); fout.close()
    print(f"RUNE_DONE shard={shard} -> {res_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["fit", "train", "eval"])
    ap.add_argument("--dirs", default=os.path.expanduser("~/jlens/out/runE_dirs.pt"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--n_tasks", type=int, default=48)
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out_dir", default="~/jlens/out/runE")
    a = ap.parse_args()
    if a.phase == "fit":
        fit_dir(a.dirs)
    elif a.phase == "train":
        train_dir(a.dirs, threads=a.threads)
    else:
        evaluate(a.dirs, a.shard, a.n_shards, a.n_tasks, a.skip, a.out_dir,
                 threads=a.threads)


if __name__ == "__main__":
    main()
