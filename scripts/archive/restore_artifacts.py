"""Restore a downloaded result bundle, validating every object before writing.

Usage: python scripts/archive/restore_artifacts.py bundle.tar.gz --output restored
Does not download data, run experiments, or deserialize model checkpoints.
"""
import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile


def safe_target(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or ':' in name or '\\' in name:
        raise ValueError('Unsafe path: ' + name)
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes output directory: ' + name)
    return path


def restore(bundle, output):
    root = output.resolve()
    with tarfile.open(bundle, 'r:gz') as tf:
        manifest = tf.extractfile('manifest.csv')
        if manifest is None:
            raise ValueError('Missing manifest')
        rows = list(csv.DictReader(io.TextIOWrapper(manifest, encoding='utf-8')))
        verified = set()
        for row in rows:
            safe_target(root, row['path'])
            expected = 'objects/' + row['sha256']
            if row['object'] != expected:
                raise ValueError('Object identity mismatch')
            if expected in verified:
                continue
            source = tf.extractfile(expected)
            if source is None:
                raise ValueError('Missing object: ' + expected)
            h = hashlib.sha256()
            size = 0
            for block in iter(lambda: source.read(1024 * 1024), b''):
                h.update(block)
                size += len(block)
            if h.hexdigest() != row['sha256'] or size != int(row['bytes']):
                raise ValueError('Object checksum/size mismatch: ' + expected)
            verified.add(expected)
        # All objects have been checked; refuse to overwrite differing files.
        for row in rows:
            target = safe_target(root, row['path'])
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
                    raise FileExistsError('Existing file differs: ' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(row['object'])
            with target.open('xb') as out:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    out.write(block)
    print('Restored', len(rows), 'files;', len(verified), 'unique objects')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    restore(args.bundle, args.output)
