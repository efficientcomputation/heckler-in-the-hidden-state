#!/usr/bin/env python3
"""Bundle manuscript source and compact supplementary code/data for arXiv.

Run: python scripts/package_arxiv.py
Submission-form fields remain outside the archive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER_FILES = ('main.tex', 'appendix.tex', 'main.bbl', '00README.json')
ANC_DIRS = ('code', 'data', 'docs', 'requirements')
ANC_SCRIPTS = ('plot_figures.py', 'manifest.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/arxiv-source.zip')
    args = parser.parse_args()
    files = {}
    for name in PAPER_FILES:
        path = ROOT / 'paper' / name
        if not path.is_file():
            raise SystemExit(f'Missing required manuscript source: {path}')
        files[name] = path.read_bytes()
    figures = sorted((ROOT / 'paper/figures').glob('*.pdf'))
    if not figures:
        raise SystemExit('No manuscript figure PDFs found under paper/figures/')
    for path in figures:
        files['figures/' + path.name] = path.read_bytes()
    ancillary = {}
    for name in ANC_DIRS:
        for path in sorted((ROOT / name).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pt', '.npz'):
                ancillary[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ANC_SCRIPTS:
        ancillary['scripts/' + name] = (ROOT / 'scripts' / name).read_bytes()
    ancillary['README.md'] = b'''# Supplementary code and result data\n\nThis directory accompanies A Heckler in the Hidden State: Correctness Signals in Diffusion Language Models. It includes the available research scripts, compact result summaries, and the inputs needed to redraw seven figures.\n\nFrom this directory, install requirements/figures.txt, then run:\n\n```bash\npython scripts/manifest.py\npython scripts/plot_figures.py\n```\n\nFigures are written to build/figures/. See docs/EXPERIMENTS.md for experimental settings and dependencies, and docs/PROVENANCE.md for the source mapping. The complete manuscript sources are one directory above this one. This compact tree includes the plot and manifest scripts; its docs/REPRODUCING.md gives direct XeLaTeX commands for the enclosing manuscript. The compact supplement does not include model weights or activation shards.\n'''
    # The ancillary tree has a plotting workflow but not the full repository's
    # paper/ directory or build/package wrappers. Adapt its guide accordingly.
    guide_name = 'docs/REPRODUCING.md'
    guide = ancillary[guide_name].decode()
    start = guide.index('## Compile the manuscript')
    end = guide.index('## Repeating experiments', start)
    guide = guide[:start] + """## Compile the accompanying manuscript

This guide is distributed inside the arXiv package's `anc/` directory. The manuscript sources (`main.tex`, `appendix.tex`, `main.bbl`, and `figures/`) are in its parent directory. From `anc/`, compile them with:

```bash
cd ..
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

The full repository's paper-build and ZIP-packaging wrappers are not included in this compact ancillary tree. The figure-redraw and manifest commands above run directly from `anc/`. The enclosing archive is already the submission source package; no extra packaging step is needed here.

""" + guide[end:]
    guide = guide.replace('From the repository root:', 'From the ancillary directory:')
    ancillary[guide_name] = guide.encode()
    ancillary['MANIFEST.json'] = (json.dumps({'scope': 'Compact arXiv ancillary code, data and documentation',
        'hash_algorithm': 'sha256', 'files': {name: {'bytes': len(content),
        'sha256': hashlib.sha256(content).hexdigest()} for name, content in sorted(ancillary.items())}}, indent=2) + '\n').encode()
    files.update({'anc/' + name: content for name, content in ancillary.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(files)
        assert all(archive.read(name) == content for name, content in files.items())
    report = {'archive': str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
              'bytes': args.output.stat().st_size, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'file_count': len(files), 'ancillary_file_count': len(ancillary),
              'source_sha256': {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())},
              'submission_metadata_included': False, 'archive_round_trip_verified': True}
    (args.output.parent / 'package-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
