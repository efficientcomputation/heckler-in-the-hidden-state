# Experiment notes

The study separates three questions: whether activations predict correctness, whether the representation changes under perturbation, and whether interventions improve generation. The files in this repository include compact results and the available research scripts for these questions.

## Models and tasks

The primary diffusion model is DiffuCoder-7B-cpGRPO. The broader probe/perturbation results include Dream-Coder-7B, Stable-DiffCoder-8B, LLaDA2.0-mini, LLaDA-flash, and UltraLLaDA-8B. The Qwen2.5-Coder capture uses the autoregressive `Qwen/Qwen2.5-Coder-7B-Instruct` checkpoint. MBPP+ supplies code-generation tasks, and the mutant checks include MBPP+ and HumanEval+.

The result files do not include immutable model or dataset revision hashes for every run. Model sizes and variant names differ in some historical filenames; for example, `LLaDA-flash-100B` denotes the model described as 102B in the study. The [data guide](DATA_GUIDE.md) identifies each stored result and distinguishes reconstructed depth curves from later measurements.

## Available scripts and dependencies

| Scripts | Purpose | Additional requirements |
| --- | --- | --- |
| `code/campaign/runA_diffu.py`, `runB_qwen.py` | Capture activations and confidence measures during generation | Model weights, dataset/prompt/grading helpers from `jcode`, output storage |
| `code/campaign/analyzeAB.py` | Problem-level probe and confidence analysis | Trusted original activation shards, NumPy and PyTorch |
| `code/campaign/runD_final.py` | Final-layer, token-subspace and nonlinear-readout comparisons | DiffuCoder weights, run-A shards, scikit-learn |
| `code/campaign/runE_steer.py` | Fit, optimize and evaluate steering directions | Run-A shards, model weights, `jcode`, `gradefix` |
| `code/campaign/runF_flash.py`, `runF_stable.py` | Mutant capture and final-layer measurements | Model weights, `jcode`, `gradefix`, `ladder_subtle2` |
| `code/perturbations/pert_ext.py`, `neuro_clean.py` | Perturbation extensions and clean activation capture | Model weights and the original mutant corpus |
| `code/perturbations/neuro_analytic.py`, `repro_seed.py` | Analysis of stored activation captures | Original clean activation dumps and `pert_ext.py` |

The local modules `jcode`, `gradefix`, and `ladder_subtle2` are not supplied. They define data loading, prompting, code extraction, grading or mutation behavior, so replacing them would change the experiment. The original run-C capture and logit-lens-generation source are also not included in this repository. Some original clean all-layer dumps were reported lost in the research notes.

`requirements/experiments.in` lists dependencies visible in the available code. Transformers 4.51.3 is recorded in the original campaign notes; the other full-experiment package versions are not frozen. The file is not a tested environment lockfile. `requirements/figures.txt` is the separate, tested figure environment.

The original campaign scripts use `~/jlens/` paths; perturbation scripts use `/scratch/jspace_pert/` and `/scratch/jspace_neuro/`. Those paths must be adapted together with the missing inputs before running new experiments. The Python source files are preserved without changing their experimental behavior.

## Probe analysis

`analyzeAB.py` centers representations by problem, forms difference-of-means directions in five problem-level folds, scores held-out problems and reports mean within-problem AUC. The default fold seed is 0. At 200 permutations, the probe-null seeds are 1000–1199 and confidence-null seeds are 2000–2199. The bootstrap seed is 7.

Three implementation details matter when interpreting its uncertainty estimates:

1. The original docstring describes refitting the full pipeline for each permutation, but the function fits scores once, permutes the evaluation labels and repeats layer selection over those fixed scores. It does not refit the probe inside the permutation loop.
2. The bootstrap samples problem IDs with replacement, then groups by the original IDs when calculating within-problem AUC. Repeated copies of a problem are therefore regrouped rather than treated as separate weighted bootstrap clusters.
3. The AUC helper assigns ranks after sorting without averaging tied ranks. With tied scores, its result can differ from a tie-aware AUC.

The supplied numerical summaries have not been recomputed to change these choices. A new statistical analysis should specify the intended protocol before comparing results.

With trusted original activation shards and the correct environment, the analysis interface is:

```bash
python code/campaign/analyzeAB.py --run A --glob '/path/to/runA/runA_shard*.pt' --n_perm 200 --out analyzeA32_recomputed.json
python code/campaign/analyzeAB.py --run B --glob '/path/to/runB/runB_shard*.pt' --n_perm 200 --out analyzeB_recomputed.json
```

These experiment commands have not been run from the compact repository. The loader uses `torch.load(..., weights_only=False)`, so the shard files must be trusted.

## Steering settings

The available run-E script uses layer 18, intervention magnitude 0.4, and eleven arms evaluated together. It distinguishes all-position and still-masked-position interventions, matched random directions, and an optimized direction. Random control directions use seed `10000 + task_index`. The supplied per-task outcomes correspond to these eleven conditions; original generation seeds and a complete immutable environment have not been reconstructed.

New model runs would be new measurements. They should record model/data revisions, seeds, hardware, precision, package versions and the exact commands alongside their results. No new model inference was performed to prepare this repository.
