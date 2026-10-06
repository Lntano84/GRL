"""Verify browsable archive copies and optionally downloaded Release bundles."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(archive, assets=None):
    rows = list(csv.DictReader((archive / 'manifests/files.csv').open(encoding='utf-8', newline='')))
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate logical paths')
    checked = 0
    for r in rows:
        if r['in_git'] != '1':
            continue
        p = archive / r['path']
        if p.stat().st_size != int(r['bytes']) or sha(p) != r['sha256']:
            raise ValueError('Copy mismatch: ' + r['path'])
        checked += 1
    result = {'logical_files': len(rows), 'git_copies_checked': checked, 'bundles_checked': 0, 'unique_objects_checked': 0}
    if assets:
        for a in json.loads((archive / 'assets.json').read_text(encoding='utf-8'))['assets']:
            p = assets / a['name']
            if p.stat().st_size != a['bytes'] or sha(p) != a['sha256']:
                raise ValueError('Bundle checksum mismatch: ' + a['name'])
            with tarfile.open(p, 'r:gz') as tf:
                member_rows = list(csv.DictReader(io.TextIOWrapper(tf.extractfile('manifest.csv'), encoding='utf-8')))
                expected = [r for r in rows if r['path'].split('/')[0] == a['domain']]
                if member_rows != expected:
                    raise ValueError('Bundle manifest differs: ' + a['name'])
                seen = set()
                for r in member_rows:
                    if r['object'] in seen:
                        continue
                    f = tf.extractfile(r['object'])
                    h = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(block); size += len(block)
                    if h.hexdigest() != r['sha256'] or size != int(r['bytes']):
                        raise ValueError('Raw object mismatch: ' + r['path'])
                    seen.add(r['object'])
                if len(seen) != a['unique_objects']:
                    raise ValueError('Object count mismatch')
            result['bundles_checked'] += 1
            result['unique_objects_checked'] += len(seen)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[2] / 'research_archive/decision-studies-2026')
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    verify(args.archive, args.assets)
