#!/usr/bin/env python
"""Run A: DiffuCoder MBPP+ corpus — all-layer pooled activations (per denoising step),
pass/fail labels, and THREE self-confidence scores per rollout:
  conf_vis    mean logprob of the completion, all tokens visible (bidirectional; naive)
  conf_masked mean logprob of the completion with response positions MASKED (the model's
              generative prior confidence given the prompt — the honest diffusion analog)
  entropy_vis mean per-position token entropy (visible forward)
Diffusion convention: logits at position i predict token i (no AR shift).
Shardable: --shard K --n_shards N (task index mod N == K).
"""
import argparse, os, sys, time, json
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/jlens/code"))
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoTokenizer
from jcode import DATASETS, build_prompt, extract_code, grade_task


def get_logits(model, ids, attn):
    out = model(input_ids=ids, attention_mask=attn)
    lg = getattr(out, "logits", None)
    if lg is not None:
        return lg
    h = out.last_hidden_state
    core = getattr(model, "model", model)
    if hasattr(core, "norm"):
        h = core.norm(h)
    W = model.get_output_embeddings()
    return F.linear(h, W.weight)


@torch.no_grad()
def selfconf(model, tok, full_ids, plen, dev):
    """full_ids: [seq] prompt+completion. Returns (conf_vis, entropy_vis, conf_masked)."""
    ids = full_ids.unsqueeze(0).to(dev)
    attn = torch.ones_like(ids)
    tgt = full_ids[plen:].to(dev)
    lp = torch.log_softmax(get_logits(model, ids, attn).float(), -1)[0, plen:]
    conf_vis = float(lp.gather(-1, tgt[:, None]).mean())
    ent = float(-(lp.exp() * lp).sum(-1).mean())
    conf_masked = None
    mid = tok.mask_token_id
    if mid is None:
        for cand in ("<|mask|>", "<mask>", "[MASK]"):
            i = tok.convert_tokens_to_ids(cand)
            if i is not None and i >= 0 and i != tok.unk_token_id:
                mid = i
                break
    if mid is not None:
        ids_m = ids.clone()
        ids_m[0, plen:] = mid
        lm = torch.log_softmax(get_logits(model, ids_m, attn).float(), -1)[0, plen:]
        conf_masked = float(lm.gather(-1, tgt[:, None]).mean())
    return conf_vis, ent, conf_masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="apple/DiffuCoder-7B-cpGRPO")
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--n_tasks", type=int, default=180)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n_roll", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--tok_per_step", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/jlens/out/runA"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    dev = "cpu"

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to(dev).eval()
    core = getattr(model, "model", model)
    layers = core.layers
    nL = len(layers)
    steps = a.max_new // a.tok_per_step
    tasks = DATASETS[a.dataset](a.n_tasks, a.skip)
    mine = [(i, t) for i, t in enumerate(tasks) if i % a.n_shards == a.shard]
    ex = ThreadPoolExecutor(max_workers=8)
    print(f"[runA shard {a.shard}/{a.n_shards}] {len(mine)} tasks x {a.n_roll} rolls, "
          f"{nL} layers, {steps} steps, threads={a.threads}", flush=True)

    plen = {"p": 0}
    bufs = {L: [] for L in range(nL)}
    def mk(L):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            bufs[L].append(h[:, plen["p"]:, :].mean(1).half().cpu())
        return hook
    for L in range(nL):
        layers[L].register_forward_hook(mk(L))

    store = []
    t_start = time.time()
    for n_done, (ti, t) in enumerate(mine):
        t0 = time.time()
        prompt = build_prompt(tok, t)
        inp = tok(prompt, return_tensors="pt").to(dev)
        plen["p"] = inp.input_ids.shape[1]
        ids = inp.input_ids.repeat(a.n_roll, 1)
        mask = inp.attention_mask.repeat(a.n_roll, 1)
        for L in bufs:
            bufs[L].clear()
        with torch.no_grad():
            out = model.diffusion_generate(
                ids, attention_mask=mask, max_new_tokens=a.max_new, steps=steps,
                temperature=a.temp, top_p=0.95, alg="entropy", alg_temp=0.,
                output_history=False, return_dict_in_generate=True)
        txts = [tok.decode(out.sequences[k][plen["p"]:], skip_special_tokens=False)
                .replace("<|dlm_pad|>", "").replace("<|im_end|>", "") for k in range(a.n_roll)]
        flags = list(ex.map(lambda g: bool(grade_task(extract_code(g), t)), txts))
        solve = float(np.mean(flags))
        mixed = 0.0 < solve < 1.0
        if mixed:
            ns = min(len(bufs[L]) for L in bufs)
            allacts = torch.stack([torch.stack(bufs[L][:ns]) for L in range(nL)], dim=1)
            for k in range(a.n_roll):
                cv, ev, cm = selfconf(model, tok, out.sequences[k].cpu(), plen["p"], dev)
                store.append({"task_id": str(ti), "rollout": f"{ti}_{k}", "pass": flags[k],
                              "conf_vis": cv, "entropy_vis": ev, "conf_masked": cm,
                              "acts": allacts[:, :, k, :].contiguous()})
        dt = time.time() - t0
        el = time.time() - t_start
        print(f"  [{n_done+1}/{len(mine)}] task {ti} solve={solve:.2f}"
              f"{' MIXED-kept' if mixed else ''} {dt:.0f}s (elapsed {el/60:.1f}m)", flush=True)

    outp = os.path.join(a.out, f"runA_shard{a.shard}.pt")
    torch.save(store, outp)
    nt = len(set(s["task_id"] for s in store))
    npass = sum(s["pass"] for s in store)
    print(f"RUNA_DONE shard={a.shard} stored={len(store)} ({npass}P/{len(store)-npass}F) "
          f"mixed_tasks={nt} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
