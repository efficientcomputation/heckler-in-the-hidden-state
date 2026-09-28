#!/usr/bin/env python
"""Run B: AR matched-pair probe — Qwen2.5-Coder-7B-Instruct on the SAME MBPP+ pipeline
as the diffusion study. Per rollout: pass/fail label, all-layer pooled residuals
(mean-over-completion + last-token), AR self-confidence (mean logprob, shifted) and
mean token entropy. Mixed problems only, matching the diffusion protocol.
Shardable: --shard K --n_shards N.
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/jlens/code"))
from concurrent.futures import ThreadPoolExecutor
import transformers
from jcode import DATASETS, build_prompt, extract_code, grade_task, reduced_capture


@torch.no_grad()
def selfconf_ar(model, tok, prompt_ids, comp_list, dev, batch=8):
    """AR shift: logits at t-1 predict token t. Returns [(mean_logprob, mean_entropy)]."""
    res = []
    p = prompt_ids.shape[0]
    for b0 in range(0, len(comp_list), batch):
        chunk = comp_list[b0:b0 + batch]
        seqs = [torch.cat([prompt_ids, c]) for c in chunk]
        maxlen = max(s.shape[0] for s in seqs)
        ids = torch.full((len(seqs), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.shape[0]] = s
            attn[i, :s.shape[0]] = 1
        logits = model(ids.to(dev), attention_mask=attn.to(dev), use_cache=False).logits
        lp = torch.log_softmax(logits.float(), -1)
        for i, c in enumerate(chunk):
            clen = c.shape[0]
            tgt = ids[i, p:p + clen].to(dev)
            sl = lp[i, p - 1:p + clen - 1]
            conf = float(sl.gather(-1, tgt[:, None]).mean())
            ent = float(-(sl.exp() * sl).sum(-1).mean())
            res.append((conf, ent))
        del logits, lp
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--dataset", default="mbppplus")
    ap.add_argument("--n_tasks", type=int, default=180)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n_roll", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/jlens/out/runB"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    dev = "cpu"

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    nL = model.config.num_hidden_layers
    tasks = DATASETS[a.dataset](a.n_tasks, a.skip)
    mine = [(i, t) for i, t in enumerate(tasks) if i % a.n_shards == a.shard]
    ex = ThreadPoolExecutor(max_workers=8)
    print(f"[runB shard {a.shard}/{a.n_shards}] {len(mine)} tasks x {a.n_roll}, "
          f"{nL} layers, threads={a.threads}", flush=True)

    store = []
    t_start = time.time()
    for n_done, (ti, t) in enumerate(mine):
        t0 = time.time()
        tok.padding_side = "left"
        enc = tok(build_prompt(tok, t), return_tensors="pt").to(dev)
        plen = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=a.temp, top_p=0.95,
                                 max_new_tokens=a.max_new, num_return_sequences=a.n_roll,
                                 pad_token_id=tok.pad_token_id)
        comps, flags = [], []
        txts = []
        for seq in gen:
            c = seq[plen:]
            c = c[c != tok.pad_token_id]
            comps.append(c.cpu())
            txts.append(tok.decode(c, skip_special_tokens=True))
        flags = list(ex.map(lambda g: bool(grade_task(extract_code(g), t)), txts))
        solve = float(np.mean(flags))
        mixed = 0.0 < solve < 1.0
        if mixed:
            prompt_ids = enc.input_ids[0].cpu()
            valid = [i for i, c in enumerate(comps) if c.numel() >= 2]
            vc = [comps[i] for i in valid]
            last, mean = reduced_capture(model, tok, prompt_ids, vc, dev, batch=4)
            confs = selfconf_ar(model, tok, prompt_ids, vc, dev, batch=4)
            for j, i in enumerate(valid):
                store.append({"task_id": str(ti), "rollout": f"{ti}_{i}", "pass": flags[i],
                              "conf": confs[j][0], "entropy": confs[j][1],
                              "act_last": last[j].contiguous(),
                              "act_mean": mean[j].contiguous()})
        dt = time.time() - t0
        el = time.time() - t_start
        print(f"  [{n_done+1}/{len(mine)}] task {ti} solve={solve:.2f}"
              f"{' MIXED-kept' if mixed else ''} {dt:.0f}s (elapsed {el/60:.1f}m)", flush=True)

    outp = os.path.join(a.out, f"runB_shard{a.shard}.pt")
    torch.save(store, outp)
    nt = len(set(s["task_id"] for s in store))
    npass = sum(s["pass"] for s in store)
    print(f"RUNB_DONE shard={a.shard} stored={len(store)} ({npass}P/{len(store)-npass}F) "
          f"mixed_tasks={nt} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
