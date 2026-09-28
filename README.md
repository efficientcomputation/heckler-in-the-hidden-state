# A Heckler in the Hidden State

**Correctness Signals in Diffusion Language Models**

This repository accompanies the paper by Angad Miglani, Samrath Singh Chadha, Kevin Li, and Manas Venkata Sai Ravulapalli. It contains the manuscript, available research scripts, and compact experimental results for studying correctness probes and activation steering in diffusion code models.

## Start here

- [Paper PDF](artifacts/paper.pdf) and [arXiv source ZIP](artifacts/arxiv-source.zip)
- [Paper source](paper/main.tex) and [appendix](paper/appendix.tex)
- [Data guide](docs/DATA_GUIDE.md): which files support each figure and result
- [Reproduction instructions](docs/REPRODUCING.md): redraw the figures and compile the paper
- [Experiment notes](docs/EXPERIMENTS.md): models, settings, dependencies, and method details
- [Source provenance](docs/PROVENANCE.md): the origin and organization of the supplied code and data

The compact data are sufficient to redraw all seven graphs. Repeating model inference requires additional inputs and helper modules described in the experiment notes; model weights and activation dumps are not included.

## Redraw the figures

With Python 3.12, from this directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements/figures.txt
.venv/bin/python scripts/manifest.py
.venv/bin/python scripts/plot_figures.py
```

PDF figures are written to `build/figures/`. The script compares their numerical coordinates with the supplied plotting reference and frozen figure values. Redrawing uses only local data after the Python dependencies are installed.

## Build the paper and submission

With XeLaTeX installed:

```bash
python scripts/build_paper.py
python scripts/package_arxiv.py
```

The outputs are `artifacts/paper.pdf` and `artifacts/arxiv-source.zip`. The ZIP contains the manuscript and compact supplementary code/data under `anc/`. Copy the [submission fields](submission/arxiv-fields.md) into the arXiv form; that Markdown file is kept outside the ZIP. Check arXiv's generated preview before completing a submission.

## Layout

| Directory | Contents |
| --- | --- |
| `paper/` | LaTeX, bibliography, appendix, and manuscript figure PDFs |
| `code/campaign/` | Probe, capture, steering, and final-layer scripts |
| `code/perturbations/` | Perturbation capture and analysis scripts |
| `data/` | Result summaries and per-task steering outcomes, grouped by purpose |
| `scripts/` | Portable figure redraw, source manifest, paper build, and packaging tools |
| `requirements/` | Figure environment and experiment dependency inventory |
| `docs/` | Data map, methods, reproduction steps, and provenance |
| `submission/` | Text for the arXiv submission form |
| `build/` | Local figures, logs and build reports, excluded by `.gitignore` |
| `artifacts/` | The paper PDF and arXiv source ZIP are included in Git; other generated files are ignored |
