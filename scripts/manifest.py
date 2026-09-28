#!/usr/bin/env python3
"""Write or verify hashes for this repository's code, data and documentation."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('code', 'data', 'docs', 'requirements', 'scripts')
ROOT_FILES = ('README.md', '.gitignore', 'CITATION.cff')


def source_files(root=ROOT):
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        paths += [p for p in (root / name).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix != '.pyc']
    return sorted(paths)


def make_manifest(root=ROOT):
    return {'scope': 'Research code, supplied result data, scripts, documentation and citation metadata; generated outputs are excluded.',
            'hash_algorithm': 'sha256',
            'files': {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_files(root)}}


def verify(root=ROOT):
    expected = json.loads((root / 'MANIFEST.json').read_text())
    actual = make_manifest(root)
    if actual['files'] != expected['files']:
        changed = sorted(name for name in set(actual['files']) | set(expected['files'])
                         if actual['files'].get(name) != expected['files'].get(name))
        raise SystemExit('Manifest mismatch: ' + ', '.join(changed))
    print(f"verified {len(expected['files'])} source files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Refresh the manifest after intentional source edits')
    args = parser.parse_args()
    if args.write:
        record = make_manifest()
        (ROOT / 'MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f"wrote manifest for {len(record['files'])} source files")
    else:
        verify()


if __name__ == '__main__':
    main()
