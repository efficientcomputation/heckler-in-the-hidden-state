# Reproducing the figures

The seven graphs can be regenerated from the included results without running a language model. This checks the plotted values; it does not rerun the underlying experiments or recompute their statistical estimates.

## Environment and commands

The figure workflow was tested with Python 3.12.13, Matplotlib 3.10.7, and NumPy 2.5.3. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/figures.txt
.venv/bin/python scripts/manifest.py
.venv/bin/python scripts/plot_figures.py
```

The manifest command compares the code, data, documentation, dependency files, and citation metadata with their recorded sizes and SHA-256 hashes. It reports the number of verified files or names files that differ.

The figure command prints `verified frozen manuscript coordinates for all seven figures` on successful completion. It compares the plotted coordinates with both `code/plotting_reference.py` and `data/figure_values.json`, including error-bar geometry.

## Outputs

| Location | Contents |
| --- | --- |
| `build/figures/` | Seven vector PDF graphs |
| `build/checks/` | PNG previews |
| `build/checks/figure_data_manifest.json` | Input hashes and numerical comparison results |

Use `.venv/bin/python scripts/plot_figures.py --output-root PATH` to choose a different output directory. Generated files under `build/` are ignored by Git. The README's depth-curve preview is copied from `build/checks/g3_jspace_depth.png` to `docs/figures/correctness-depth.png`.

The [data guide](DATA_GUIDE.md) maps every graph to its inputs. The four-bar steering and four-model comparison graphs use summary values from the original plotting code. The separate eleven-condition steering experiment has its own per-task outcome files and must not be substituted for the four-bar data.

## Updating repository files

After an intentional change to a file covered by the manifest, review the change and refresh its hashes:

```bash
python3 scripts/manifest.py --write
python3 scripts/manifest.py
```

Commit `MANIFEST.json` with the changed files. Refreshing the manifest records the new contents; it is not a scientific validation of changed data. Changes to plotting code or result data should also pass the numerical figure check above.

## Repeating model experiments

The research scripts require additional data, model downloads, and helper modules, and retain their original run-directory assumptions. Read [EXPERIMENTS.md](EXPERIMENTS.md) before using them. `requirements/figures.txt` is a tested plotting environment; `requirements/experiments.in` is a dependency inventory, not a complete experiment environment.
