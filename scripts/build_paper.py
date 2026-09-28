#!/usr/bin/env python3
"""Compile paper/main.tex with XeLaTeX or Tectonic into artifacts/paper.pdf."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', default='xelatex', help='xelatex or tectonic executable')
    parser.add_argument('--only-cached', action='store_true', help='Tectonic: avoid network access')
    parser.add_argument('--support-file', type=Path, action='append', default=[],
                        help='Optional standard TeX package unavailable in the local engine cache; may repeat')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/paper.pdf')
    args = parser.parse_args()
    engine = shutil.which(args.engine)
    if not engine:
        raise SystemExit(f'Cannot find TeX engine: {args.engine}')
    paper = ROOT / 'paper'
    if not (paper / 'main.tex').is_file():
        raise SystemExit('Missing paper/main.tex')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    report = {'engine': engine, 'commands': [], 'arxiv_server_build': 'not performed'}
    with tempfile.TemporaryDirectory(prefix='paper-', dir=build) as temporary:
        work = Path(temporary)
        for name in ('main.tex', 'appendix.tex', 'main.bbl', '00README.json'):
            if (paper / name).is_file():
                shutil.copyfile(paper / name, work / name)
        shutil.copytree(paper / 'figures', work / 'figures')
        for path in args.support_file:
            if not path.is_file():
                raise SystemExit(f'Missing support file: {path}')
            shutil.copyfile(path, work / path.name)
        if 'tectonic' in Path(engine).name.lower():
            command = [engine, '--untrusted', '--keep-logs', '--keep-intermediates', '--reruns', '2']
            if args.only_cached:
                command.append('--only-cached')
            commands = [command + ['main.tex']]
        else:
            if args.only_cached:
                raise SystemExit('--only-cached applies only to Tectonic')
            commands = [[engine, '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']] * 3
        output = []
        for command in commands:
            report['commands'].append(command)
            result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output.append(result.stdout)
            (build / 'paper-console.log').write_text('\n'.join(output))
            if result.returncode:
                raise SystemExit(f'TeX exited {result.returncode}; see build/paper-console.log')
        log = (work / 'main.log').read_text(errors='replace')
        (build / 'paper.log').write_text(log)
        if re.search(r'undefined references|Citation .* undefined|Reference .* undefined', log):
            raise SystemExit('Unresolved reference or citation; see build/paper.log')
        report['latex_warnings'] = [line for line in log.splitlines() if re.search(r'Warning|Overfull|Underfull|Missing character', line)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / 'main.pdf', args.output)
    report.update({'pdf_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   'pdf_bytes': args.output.stat().st_size, 'local_build_passed': True})
    if shutil.which('pdffonts'):
        fonts = subprocess.check_output(['pdffonts', str(args.output)], text=True)
        (build / 'pdffonts.txt').write_text(fonts)
        rows = fonts.splitlines()[2:]
        report['all_fonts_embedded'] = all(re.search(r'\byes\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$', row) for row in rows)
        report['type3_fonts'] = 'Type 3' in fonts
    if shutil.which('pdfinfo'):
        info = subprocess.check_output(['pdfinfo', str(args.output)], text=True)
        (build / 'pdfinfo.txt').write_text(info)
        pages = re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)
        report['pages'] = int(pages.group(1)) if pages else None
    (build / 'paper-build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
