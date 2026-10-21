"""Shared bookkeeping for the fixed train/validation data collection."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
W = Path(__file__).parent
OUT = ROOT / 'outputs/grid100_data_foundation'
RUNTIME = OUT / 'bulk_runtime'
ARCHIVE = ROOT / 'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID100_data_foundation'
COLLAB = ROOT / '论文协作资料/FRAME_学习数据第一阶段_GRID100_20261021'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify(files):
    for name, expected in files.items():
        assert sha(ROOT / name) == expected, name


def test_sealed():
    design = read(OUT / 'design.json')
    for item in design['scenario_plan']:
        if item['partition'] == 'test':
            for folder in [ROOT / design['environment'] / 'chronics' / item['scenario'],
                           OUT / 'datasets' / item['scenario'],
                           OUT / 'generation_receipts' / (item['scenario'] + '.json'),
                           OUT / 'dataset_receipts' / (item['scenario'] + '.json')]:
                assert not folder.exists(), f'Final test materialized: {folder}'


def items():
    design = read(OUT / 'design.json')
    planned = [x for x in design['scenario_plan'] if x['partition'] in ('train', 'validation')]
    assert len(planned) == 40 and planned[0] == design['pilot']
    assert sum(x['partition'] == 'train' for x in planned) == 32
    assert sum(x['partition'] == 'validation' for x in planned) == 8
    return planned


def verify_dataset(receipt):
    assert receipt['passed'] and receipt['partition'] in ('train', 'validation')
    assert receipt['model_fits'] == 0 and not receipt['final_test_materialized']
    verify(receipt['files'])
    folder = ROOT / receipt['dataset_path']
    manifest = read(folder / 'manifest.json')
    assert len(manifest['samples']) == receipt['samples']
    for sample in manifest['samples']:
        assert sample['partition'] == receipt['partition']
        assert sample['calendar_group'] == receipt['calendar_group']
        assert sample['scenario'] == receipt['scenario']
    return manifest
