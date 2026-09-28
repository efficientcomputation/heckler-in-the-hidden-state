# Data guide

The files below map the paper's graphs and experimental sections to compact results. Filenames retain the run and model identifiers used when the results were produced. The separate provenance map lists their original locations.

## Graph inputs

| Output figure | Input files or values | Interpretation |
| --- | --- | --- |
| `g3_jspace_depth.pdf` | `data/probing/decode_*.json`, `layer_curve` | Original per-layer correctness-read curves for six models |
| `jspace_readable_vs_interpretable.pdf` | The same six files, `auc_at_peak` and `neg_wrong_frac` | Probe AUC versus wrongness-token count among the negative pole's top 30 tokens |
| `g1_param_correctness_noise.pdf` | `data/perturbations/noise_replicates/neuro_partA_*.json` | Mean stored AUC at each noise magnitude for DiffuCoder, LLaDA-mini and LLaDA-flash |
| `auc_vs_causal.pdf` | `data/perturbations/interventions/*__causal.json` | Attention-ablation read AUC, using the plotting code's drift check |
| `dissociation_read_vs_generation.pdf` | `DiffuCoder-7B-cpGRPO__noise.json` and `DiffuCoder-7B-cpGRPO__behavioral.json` in the interventions directory | Internal read AUC and generated-code pass@1 under perturbation |
| `steer_asymmetry.pdf` | Constants in `code/plotting_reference.py` and `scripts/plot_figures.py` | Earlier four-arm steering summary: 0.433 baseline, 0.429 random, 0.388 positive probe, 0.298 negative probe |
| `compare_ar.pdf` | Constants in the same plotting files | Earlier diffusion/autoregressive comparison: 0.789, 0.696, 0.677 and 0.559; the original 0.53 null guide is preserved |

`data/figure_values.json` stores all plotted numerical coordinates for the seven figures. It is used to check redrawing, including points, bars and error-bar geometry. It is not a replacement for the experiment inputs.

The original depth curves include final entries whose capture convention can differ from intermediate blocks. Use the dedicated final-layer summaries below for the paper's remeasurement discussion. The dissociation figure follows the manuscript's plotting inputs, which differ from an earlier blog figure.

## Probe and confidence comparisons

| File | Run and contents |
| --- | --- |
| `data/confidence/analyzeA32.json` | DiffuCoder, 32 denoising steps; 95 rollouts from 19 mixed-outcome problems; probe depth curve and confidence comparisons |
| `data/confidence/analyzeA.json` | DiffuCoder, 16 denoising steps; 324 rollouts from 54 mixed-outcome problems |
| `data/confidence/analyzeB.json` | Qwen2.5-Coder; 448 rollouts from 56 mixed-outcome problems |
| `data/tables/confidence_comparison.csv` | Supplied cross-run summary of the confidence analyses |

A mixed-outcome problem has at least one passing and one failing rollout. The analysis script is `code/campaign/analyzeAB.py`. Its required activation shards are not included; the JSON files hold the supplied analysis results.

## Mutant and final-layer tests

| File | Run and contents |
| --- | --- |
| `data/mutants/runC_ref.json` | MBPP+ mutant correctness test |
| `data/mutants/runC_hep.json` | HumanEval+ mutant correctness test |
| `data/final_layer/runD.json` | DiffuCoder final-layer and representation-subspace battery |
| `data/final_layer/runF_flash.json` | Flash mutant remeasurement, including raw and normalized last-block measurements |
| `data/final_layer/runFS.json` | Stable-DiffCoder follow-up summary |
| `data/final_layer/runFS_probe.log` | Compact console summary accompanying the Stable follow-up |
| `data/tables/bug_tests.csv` | Supplied mutant-test summary |
| `data/tables/final_layer_battery.csv` | Supplied DiffuCoder final-layer battery summary |
| `data/tables/depth_curves.csv` | Supplied original depth curves and available remeasurement rows |

The available final-layer scripts are `runD_final.py`, `runF_flash.py`, and `runF_stable.py` under `code/campaign/`. The run-C capture scripts are not included in the available source.

## Steering

`data/steering/runE_shard0.jsonl` through `runE_shard5.jsonl` contain per-task pass and compile outcomes for the eleven-arm campaign. Each line identifies a task and records the arm outcomes. `data/tables/steering_arms.csv` summarizes those outcomes. The available fitting, training and evaluation code is `code/campaign/runE_steer.py`.

This campaign is separate from the earlier four-bar steering figure. Do not substitute one result set for the other when reproducing the paper's plots.

## Noise and summary tables

`data/perturbations/noise_replicates/` contains individual stored rows across noise magnitude and seed, with additional representation measures. The redraw averages the stored AUCs by magnitude. `data/tables/noise_curves.csv` is the supplied compact noise summary. The perturbation scripts under `code/perturbations/` require additional mutant inputs and activation dumps to generate new measurements.
