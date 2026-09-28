# Source provenance

This standalone repository reorganizes the available research code and compact result files while keeping the original research-script bytes and result filenames. The earlier research-source commit is `b38aabc848dae38b10eed39ab948588d06d58d06`. That identifier belongs to the source checkout from which these files were collected; it is not a commit in this new repository.

The machine-readable [source mapping](source_mapping.json) lists each imported file, its original path and SHA-256 hash. Research scripts under `code/campaign/`, `code/perturbations/`, and `code/plotting_reference.py` were copied without changes. Results were grouped under `data/` without altering their contents. Original generic run paths remain in those scripts and the compact Stable follow-up log.

`scripts/plot_figures.py` adapts the earlier vector-figure redraw to this repository's paths. It reads `data/` directly, redirects the numerical-reference script's paths in memory, and defaults to `build/figures/` and `build/checks/`. It does not depend on the parent checkout. Figure coordinates are checked against both the original plotting reference and `data/figure_values.json`.

`code/plotting_reference.py` keeps the original script's historical formatting comments. It serves as a numerical reference, not as the manuscript template. Two figures contain hardcoded summary values inherited from that script; the [data guide](DATA_GUIDE.md) identifies them explicitly.

The root `MANIFEST.json` identifies the current code/data/documentation files by size and SHA-256 hash. Run `python scripts/manifest.py` to check it. Manuscript source and generated PDF/ZIP hashes are recorded separately by the build and packaging tools, so changing the manuscript does not change the research-data manifest.

The original four-bar steering results, the eleven-arm steering campaign, and later final-layer summaries are distinct experiments. They retain their separate files. No results were inferred from graph pixels, and no missing experimental inputs were fabricated during reorganization.

No project-level licence was present among the supplied tracked research files, so this repository does not add one. Model and benchmark licences remain separate from the availability of this code. Repository initialization, a remote URL, and publication are left to the authors.
