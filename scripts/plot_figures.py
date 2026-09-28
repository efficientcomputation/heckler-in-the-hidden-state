#!/usr/bin/env python3
"""Rebuild the seven manuscript graphs as vector PDFs without rerunning experiments.

From the repository root:
  python -m pip install -r requirements/figures.txt
  python scripts/plot_figures.py

Every numeric line, scatter point, rectangle, and error-bar segment is checked
for exact equality against code/plotting_reference.py run on the same inputs.
The resulting PDF files are standalone; arXiv does not need to run this script.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import re
import subprocess

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "build")
args = parser.parse_args()
REPO_ROOT = args.repo_root.resolve()
OUTPUT_ROOT = args.output_root.resolve()
FIGURE_DIR = OUTPUT_ROOT / "figures"
QA_DIR = OUTPUT_ROOT / "checks"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)
BASELINE = {}
FINAL_DATA = {}


def as_numbers(values):
    """JSON-safe native values, preserving exact stored float values."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return [as_numbers(value) for value in values]
    return values


def numeric_artists(fig):
    result = []
    for ax in fig.axes:
        data = {
            "lines": [{"x": as_numbers(line.get_xdata()), "y": as_numbers(line.get_ydata())}
                      for line in ax.lines],
            "scatter": [], "line_segments": [], "rectangles": [],
        }
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                data["scatter"].append(as_numbers(collection.get_offsets()))
            elif isinstance(collection, LineCollection):
                data["line_segments"].append(as_numbers(collection.get_segments()))
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                data["rectangles"].append([float(patch.get_x()), float(patch.get_y()),
                                           float(patch.get_width()), float(patch.get_height())])
        result.append(data)
    return result


def capture_reference(fig, name):
    BASELINE[Path(name).stem] = numeric_artists(fig)


reference_path = REPO_ROOT / "code" / "plotting_reference.py"
reference = reference_path.read_text()
reference = reference.replace('f"{BASE}/decode/decode_*.json"', 'f"{BASE}/probing/decode_*.json"')
reference = reference.replace('f"{BASE}/raw/*.json"', 'f"{BASE}/perturbations/interventions/*.json"')
reference = reference.replace('f"{BASE}/neuro"', 'f"{BASE}/perturbations/noise_replicates"')
reference = re.sub(r'^BASE = .+$',
                   "BASE = " + repr(str(REPO_ROOT / "data")), reference,
                   count=1, flags=re.MULTILINE)
reference = reference.replace(
    'fig.savefig(f"{G}/{name}", bbox_inches="tight", facecolor="white")',
    'capture_reference(fig, name)')
reference = reference.replace('print("wrote", name)', 'pass')
reference = reference.replace('print("done")', 'pass')
exec(compile(reference, str(reference_path), "exec"), {"capture_reference": capture_reference})
assert len(BASELINE) == 7, f"Expected seven reference figures, got {len(BASELINE)}"

import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import json, glob, os

BASE = str(REPO_ROOT / "data")
G = str(FIGURE_DIR); NEU = f"{BASE}/perturbations/noise_replicates"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 300,
    "text.color": "#111111", "axes.labelcolor": "#111111",
    "xtick.color": "#111111", "ytick.color": "#111111",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})

NAME = {"DiffuCoder-7B": "DiffuCoder-7B", "Dream-Coder-7B": "Dream-7B",
        "LLaDA2.0-mini": "LLaDA-mini", "Stable-DiffCoder-8B": "Stable-8B",
        "UltraLLaDA-8B": "UltraLLaDA-8B", "LLaDA-flash-100B": "LLaDA-flash-102B"}
COL = {"DiffuCoder-7B": "#1f77b4", "Dream-Coder-7B": "#ff7f0e", "Stable-DiffCoder-8B": "#2ca02c",
       "LLaDA2.0-mini": "#d62728", "LLaDA-flash-100B": "#7b3fb5", "UltraLLaDA-8B": "#17a2b8"}
MK  = {"DiffuCoder-7B": "o", "Dream-Coder-7B": "s", "Stable-DiffCoder-8B": "^",
       "LLaDA2.0-mini": "D", "LLaDA-flash-100B": "v", "UltraLLaDA-8B": "P"}
LS  = {"DiffuCoder-7B": "-", "Dream-Coder-7B": "--", "Stable-DiffCoder-8B": "-.",
       "LLaDA2.0-mini": ":", "LLaDA-flash-100B": "-", "UltraLLaDA-8B": "--"}

def chance(ax, label_x=0.01, color="#555555"):
    ax.axhline(0.5, color="#777777", ls=(0, (4, 3)), lw=0.9)
    ax.text(label_x, 0.502, "chance", transform=ax.get_yaxis_transform(),
            fontsize=9.5, color=color, va="bottom", ha="right" if label_x > 0.5 else "left")

def save(fig, name):
    stem = Path(name).stem
    current = numeric_artists(fig)
    assert current == BASELINE[stem], f"Numeric data changed in {stem}"
    FINAL_DATA[stem] = current
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white",
                metadata={"Creator": "Matplotlib / arXiv figure preparation", "CreationDate": None})
    fig.savefig(QA_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)
    print("wrote", pdf_path.relative_to(OUTPUT_ROOT))

# ---------------- data: decode ----------------
D = {}
for f in glob.glob(f"{BASE}/probing/decode_*.json"):
    D[f.split("decode_")[1][:-5]] = json.load(open(f))

# ---------------- 1. depth curves ----------------
fig, ax = plt.subplots(figsize=(6.3, 3.6))
order = sorted(D, key=lambda m: D[m].get("auc_at_peak", 0))
for m in order:
    lc = D[m]["layer_curve"]; n = len(lc) - 1
    xs = [i / n for i in range(len(lc))]
    ax.plot(xs, lc, ls=LS.get(m, "-"), marker=MK.get(m, "o"), markevery=0.15,
            color=COL.get(m, "#888"), lw=1.1, ms=3.5, label=NAME.get(m, m))
    pk = max(range(len(lc)), key=lambda i: lc[i])
    ax.plot(pk / n, lc[pk], MK.get(m, "o"), color=COL.get(m, "#888"), ms=5, zorder=4)
chance(ax, label_x=0.985)
ax.set_xlabel("relative depth (layer / total layers)")
ax.set_ylabel("correctness read AUC")
ax.set_ylim(0, 1.0)
ax.legend(loc="lower right", ncol=2, columnspacing=1.0, handlelength=2.2)
save(fig, "g3_jspace_depth.png")

# ---------------- 2. readable vs interpretable ----------------
LOFF = {"LLaDA-flash-100B": (-8, 2, "right"), "LLaDA2.0-mini": (-8, 2, "right"),
        "DiffuCoder-7B": (0, 12, "center"), "Dream-Coder-7B": (6, -17, "center"),
        "UltraLLaDA-8B": (-12, -17, "center"), "Stable-DiffCoder-8B": (0, 12, "center")}
fig, ax = plt.subplots(figsize=(6.3, 3.6))
ax.axhspan(-1, 3, color="#999999", alpha=0.15)
ax.text(0.556, 1.0, "indistinguishable from\na random direction", fontsize=9,
        color="#555555", va="center")
for m in D:
    x = D[m].get("auc_at_peak", 0); y = D[m].get("neg_wrong_frac", 0) * 30
    ax.scatter(x, y, s=55, color=COL.get(m, "#888"), marker=MK.get(m, "o"),
               zorder=3, edgecolor="white", lw=0.9)
    dx, dy, ha = LOFF.get(m, (0, 10, "center"))
    ax.annotate(NAME.get(m, m), (x, y), (dx, dy), textcoords="offset points",
                ha=ha, fontsize=10, color=COL.get(m, "#333"), fontweight="bold")
ax.set_xlabel("correctness read AUC")
ax.set_ylabel("“wrong” words in the broken pole’s top 30")
ax.set_xlim(0.55, 0.90); ax.set_ylim(-4, 29)
save(fig, "jspace_readable_vs_interpretable.png")

# ---------------- 3. noise vs AUC across model size ----------------
def partA_mean(model, mmax=0.8):
    rows = json.load(open(f"{NEU}/neuro_partA_{model}.json")); by = {}
    for r in rows: by.setdefault(r["magnitude"], []).append(r)
    x = [m for m in sorted(by) if m <= mmax]
    y = [sum(r["auc"] for r in by[m]) / len(by[m]) for m in x]
    return x, y

G1 = [("LLaDA2.0-mini", "LLaDA-mini (1.4B active)", "#9A4D8E", "D", ":"),
      ("DiffuCoder-7B-cpGRPO", "DiffuCoder-7B (7.6B)", "#0F4D92", "o", "-"),
      ("llada_flash", "LLaDA-flash (102B)", "#B64342", "v", "--")]
fig, ax = plt.subplots(figsize=(5.8, 3.3))
for key, lab, col, mk, ls in G1:
    x, y = partA_mean(key)
    ax.plot(x, y, ls=ls, marker=mk, color=col, lw=1.1, ms=3.5, label=lab)
ax.set_xlim(0, None); ax.set_ylim(0, 1.0)
chance(ax, label_x=0.10)
ax.set_xlabel("noise magnitude $m$")
ax.set_ylabel("correctness read AUC")
ax.legend(loc="lower left")
save(fig, "g1_param_correctness_noise.png")

# ---------------- data: raw perturbation rows ----------------
def as_rows(data):
    if isinstance(data, list): return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict): return v
        if "auc" in data or "model" in data: return [data]
    return []

rows = []
for f in glob.glob(f"{BASE}/perturbations/interventions/*.json"):
    try: data = json.load(open(f))
    except Exception: continue
    for r in as_rows(data):
        r.setdefault("model", os.path.basename(f).split("__")[0])
        if "flash" in r["model"].lower(): r["model"] = "LLaDA-flash-100B"
        _pt = (r.get("ptype") or os.path.basename(f).split("__")[-1].split(".")[0]).lower()
        if _pt.startswith("behavioral"): _pt = "behavioral"
        r["ptype"] = _pt if _pt in ("noise", "rotate", "causal", "behavioral") else "_skip"
        r["x"] = next((r[k] for k in ("magnitude", "theta_deg", "frac") if r.get(k) is not None), None)
        rows.append(r)
behavioral = [r for r in rows if r["ptype"] == "behavioral"]
prows = [r for r in rows if r.get("auc") is not None and r["ptype"] in ("noise", "rotate", "causal")]

def series(model, ptype, key="auc"):
    d = [r for r in prows if r["model"] == model and r["ptype"] == ptype and r["x"] is not None]
    seen = {}
    for r in sorted(d, key=lambda r: r["x"]):
        if r["x"] not in seen or r.get("seq_len", 0) < seen[r["x"]].get("seq_len", 1e9): seen[r["x"]] = r
    d = [seen[k] for k in sorted(seen)]
    return [r["x"] for r in d], [r.get(key) for r in d]

def fired(model, ptype):
    _, dr = series(model, ptype, "drift_cosine"); dr = [d for d in dr if d is not None]
    return len(dr) >= 2 and min(dr) < 0.99

SIZE = {"DiffuCoder-7B-cpGRPO": "7.6B", "Dream-Coder-7B": "7.6B", "Stable-DiffCoder-8B": "8.2B",
        "LLaDA2.0-mini": "1.4B active", "LLaDA-flash-100B": "102B", "UltraLLaDA-8B": "8B"}
RNAME = {"DiffuCoder-7B-cpGRPO": "DiffuCoder-7B", "Dream-Coder-7B": "Dream-Coder-7B",
         "Stable-DiffCoder-8B": "Stable-DiffCoder-8B", "LLaDA2.0-mini": "LLaDA-mini",
         "LLaDA-flash-100B": "LLaDA-flash", "UltraLLaDA-8B": "UltraLLaDA"}
RMK = {"DiffuCoder-7B-cpGRPO": "o", "Dream-Coder-7B": "s", "Stable-DiffCoder-8B": "^",
       "LLaDA2.0-mini": "D", "LLaDA-flash-100B": "v", "UltraLLaDA-8B": "P"}
RLS = {"DiffuCoder-7B-cpGRPO": "-", "Dream-Coder-7B": "--", "Stable-DiffCoder-8B": "-.",
       "LLaDA2.0-mini": ":", "LLaDA-flash-100B": "-", "UltraLLaDA-8B": "--"}
RCOL = {"DiffuCoder-7B-cpGRPO": "#1f77b4", "Dream-Coder-7B": "#ff7f0e",
        "Stable-DiffCoder-8B": "#2ca02c", "LLaDA2.0-mini": "#d62728",
        "LLaDA-flash-100B": "#7b3fb5", "UltraLLaDA-8B": "#17a2b8"}
PARAM = {}
for r in prows:
    if r.get("param_count"): PARAM.setdefault(r["model"], r["param_count"])
pmodels = [m for m in sorted({r["model"] for r in prows}, key=lambda m: PARAM.get(m, 0))
           if "ultra" not in m.lower() and not m.upper().startswith("VAL")]

# ---------------- 4. attention ablation ----------------
fig, ax = plt.subplots(figsize=(6.0, 3.3))
for m in pmodels:
    x, y = series(m, "causal", "auc")
    if not x or not fired(m, "causal"): continue
    ax.plot(x, y, ls=RLS.get(m, "-"), marker=RMK.get(m, "o"), color=RCOL.get(m, "#888"),
            lw=1.1, ms=3.5, label=f"{RNAME.get(m, m)} ({SIZE.get(m, '?')})")
chance(ax)
ax.set_xlabel("attention edges ablated (fraction)")
ax.set_ylabel("correctness read AUC")
ax.set_ylim(0.34, 0.92); ax.margins(x=0.03)
ax.legend(loc="upper right", fontsize=9.5)
save(fig, "auc_vs_causal.png")

# ---------------- 5. dissociation (single axis, no twin) ----------------
if behavioral:
    bm = behavioral[0]["model"]
    bx = sorted({r["x"] for r in behavioral if r.get("x") is not None})
    p1 = [next((r.get("pass_at_1") for r in behavioral if r.get("x") == x), None) for x in bx]
    rx, ry = series(bm, "noise", "auc")
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(rx, ry, "-o", color="#c62828", lw=1.1, ms=3.5, label="internal read of correctness (AUC)")
    ax.plot(bx, p1, "--s", color="#2e7d32", lw=1.1, ms=3.5, label="generated code (pass@1)")
    chance(ax)
    ax.set_ylim(0.42, 0.86)
    ax.set_xlabel("noise magnitude ($\\times$ residual norm)")
    ax.set_ylabel("AUC / pass@1")
    ax.legend(loc="lower left", fontsize=9.5)
    save(fig, "dissociation_read_vs_generation.png")

# ---------------- 6. steering asymmetry ----------------
labels = ["baseline\n(no push)", "random\npush", "push toward\n``correct''", "push toward\n``wrong''"]
labels = [l.replace("``", "“").replace("''", "”") for l in labels]
vals = [0.433, 0.429, 0.388, 0.298]
errs = [None, 0.036, None, None]
cols = ["#9AA0A6", "#CFCECE", "#0F4D92", "#B64342"]
znote = [None, None, "z = −1.15", "z = −3.66"]
fig, ax = plt.subplots(figsize=(5.8, 3.3))
for i, (v, c) in enumerate(zip(vals, cols)):
    ax.bar(i, v, width=0.62, color=c, edgecolor="white", linewidth=0.6, zorder=3)
    if errs[i] is not None:
        ax.errorbar(i, v, yerr=errs[i], fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=5)
    label_y = v + (errs[i] + 0.011 if errs[i] is not None else 0.011)
    ax.text(i, label_y, f"{v:.3f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#111111", zorder=6)
    if znote[i]:
        ax.text(i, v - 0.025, znote[i], ha="center", va="top", fontsize=9,
                color="white", fontweight="bold", zorder=6)
ax.axhline(0.433, color="#777777", ls=(0, (4, 3)), lw=0.8, zorder=1)
ax.text(3.44, 0.437, "baseline", fontsize=9, color="#555555", va="bottom", ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("pass@1")
ax.set_ylim(0, 0.52)
save(fig, "steer_asymmetry.png")

# ---------------- 7. AR comparison ----------------
labels = ["DiffuCoder-7B\n(diffusion)", "Qwen3-Coder\n(autoregressive)",
          "Qwen2.5-Coder\n(autoregressive)", "GLM-4.5-Air\n(autoregressive)"]
vals = [0.789, 0.696, 0.677, 0.559]
cols = ["#0F4D92", "#42949E", "#5FA8B0", "#B64342"]
znote = ["+7.5$\\sigma$", "+6.3$\\sigma$", "+4.5$\\sigma$", "+0.5$\\sigma$ (n.s.)"]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
for i, (v, c) in enumerate(zip(vals, cols)):
    ax.bar(i, v, width=0.62, color=c, edgecolor="white", linewidth=0.6, zorder=3)
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#111111")
    ax.text(i, v - 0.028, znote[i], ha="center", va="top", fontsize=9, color="white",
            fontweight="bold")
ax.axhline(0.53, color="#777777", ls=(0, (4, 3)), lw=0.8, zorder=1)
ax.legend([ax.lines[-1]], ["chance (shuffled-label null)"], loc="upper right",
          bbox_to_anchor=(0.99, 0.99), fontsize=9, handlelength=2.5)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("correctness read AUC")
ax.set_ylim(0, 0.9)
save(fig, "compare_ar.png")

print("done")

# Record exact plotted values, input hashes and environment after all checks pass.
input_paths = [reference_path, Path(__file__).resolve()]
input_paths += sorted((REPO_ROOT / "data" / "probing").glob("decode_*.json"))
input_paths += sorted((REPO_ROOT / "data" / "perturbations" / "noise_replicates").glob("neuro_partA_*.json"))
input_paths += sorted((REPO_ROOT / "data" / "perturbations" / "interventions").glob("*.json"))
input_hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in input_paths}
commit = 'b38aabc848dae38b10eed39ab948588d06d58d06'
manifest = {
    "scope": "Redrawing supplied results only; no experiment reruns or statistical reanalysis",
    "numeric_reference": "code/plotting_reference.py evaluated on current repository JSON inputs",
    "exact_numeric_artist_equality": FINAL_DATA == BASELINE,
    "figure_count": len(FINAL_DATA),
    "source_commit": commit,
    "python_version": platform.python_version(),
    "matplotlib_version": mpl.__version__,
    "input_sha256": input_hashes,
    "figures": FINAL_DATA,
    "limitations": [
        "Numeric equality is to the supplied plotting script and current JSON, not experimental validity.",
        "Legacy raster images are not inputs to this standalone redraw.",
        "The dissociation plot differs numerically from graphs_original_blog; current manuscript values are retained.",
        "The comparison chart retains the existing 0.53 null line and hardcoded bar annotations.",
    ],
}
(QA_DIR / "figure_data_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
print("verified exact numeric artist equality for all seven figures")

# Artist insertion order can depend on directory order. Compare each axis as
# a multiset of exact coordinates, retaining duplicates and every numeric value.
def canonical_payload(figures):
    return {name: [{kind: sorted(json.dumps(item, sort_keys=True, allow_nan=False)
                                      for item in values)
                    for kind, values in axis.items()}
                   for axis in axes]
            for name, axes in figures.items()}
expected_path = REPO_ROOT / "data" / "figure_values.json"
expected = json.loads(expected_path.read_text())
assert canonical_payload(FINAL_DATA) == canonical_payload(expected["figures"]), \
    "Plotted coordinates differ from frozen manuscript values"
manifest["matches_frozen_manuscript_values"] = True
manifest["frozen_values_sha256"] = hashlib.sha256(expected_path.read_bytes()).hexdigest()
(QA_DIR / "figure_data_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
print("verified frozen manuscript coordinates for all seven figures")
