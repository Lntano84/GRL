"""Freeze the remaining unaccessed date families before extracting any payload."""
import hashlib
import json
import py_compile
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/grid62'
SEED = 'GRID62_FRAMEWORK_FINAL_CONFIRMATION_20261013_V1'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
key = lambda s: hashlib.sha256((SEED + '|' + s).encode()).hexdigest()


def main():
    assert not OUT.exists(), 'Never replace a frozen stage.'
    assert read(ROOT / 'outputs/grid61/review.json')['passed_saved_data']
    index = read(ROOT / 'outputs/grid03/archive_index.json')
    by_date = {}
    for record in index['entries']:
        p = PurePosixPath(record['name']).parts
        if len(p) >= 3 and p[1] == 'chronics' and p[2].startswith('2035'):
            by_date.setdefault(p[2][:10], set()).add(p[2])
    extracted = sorted(p.name for p in (ROOT / 'work/grid03/formal_env_initialized/chronics').iterdir() if p.is_dir())
    excluded = set(read(ROOT / 'outputs/grid52/reservation.json')['excluded_dates'])
    excluded.update(w[:10] for w in read(ROOT / 'outputs/grid52/design.json')['weeks'])
    excluded.update(w[:10] for w in extracted)
    # Conservatively exclude every episode/date mentioned in earlier top-level
    # protocol metadata, including reservations that might not have run.
    metadata_receipts = []
    for folder in sorted((ROOT / 'outputs').glob('grid*')):
        for name in ('design.json', 'reservation.json', 'runtime_design.json', 'design_freeze.json', 'run_manifest.json'):
            p = folder / name
            if not p.is_file():
                continue
            value = p.read_text(encoding='utf-8')
            # Reservation/catalogue lists are handled separately above: an
            # available-date list is not evidence of outcome exposure.
            if name == 'reservation.json':
                obj = read(p)
                text = json.dumps({k: v for k, v in obj.items() if k not in ('available_dates_by_quarter', 'existing_extracted_episode_names', 'excluded_dates')})
            else:
                obj = read(p)
                text = json.dumps({k: v for k, v in obj.items() if k not in ('available_dates_by_quarter', 'archive_entries', 'chronics_catalogue')})
            found = sorted(set(re.findall(r'2035-\d{2}-\d{2}(?:_\d+)?', text)))
            excluded.update(s[:10] for s in found)
            metadata_receipts.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'mentioned_dates': sorted(set(s[:10] for s in found))})
    available = sorted(set(by_date) - excluded)
    assert available == ['2035-04-09', '2035-05-14', '2035-11-19'], available
    families = []
    for date in available:
        variants = sorted(by_date[date], key=lambda s: int(s.split('_')[1]))
        assert len(variants) == 16
        families.append({'date': date, 'primary_episode': min(variants, key=key), 'reserved_all_variants': variants})
    weeks = [f['primary_episode'] for f in families]
    policies = ['FULL', 'FIRST_SAFE_BEST', 'NO_LOCAL', 'FRAME']
    methods = [{'method': p, 'rule': 'STATIC_ID', 'policy': p, 'budget': 128,
                'checkpoint': None, 'cache_dc': True, 'cache_safe': True} for p in policies]
    order = []
    for repeat in (0, 1):
        for i, week in enumerate(weeks):
            rotated = methods[i:] + methods[:i]
            if repeat:
                rotated = rotated[::-1]
            order.extend({**m, 'week': week, 'repeat': repeat, 'stage_phase': 'confirmation'} for m in rotated)
    reservation = {'status': 'FROZEN_BEFORE_EXTRACTION_AND_EXECUTION', 'selection_seed': SEED,
        'selection_rule': 'All remaining unaccessed dates; one variant per date by SHA256(seed|episode), without outcomes.',
        'metadata_only': True, 'catalogue_dates': len(by_date), 'excluded_dates': sorted(excluded),
        'existing_extracted_episode_names': extracted, 'families': families, 'metadata_receipts': metadata_receipts,
        'source_archive_sha256': index['archive_sha256'],
        'warning': 'Only three remaining date families, two in quarter2 and one in quarter4. Not balanced seasonal or cross-grid confirmation. Author action-library training exposure unknown; no leakage accusation.'}
    design = {'stage': 'GRID62_FINAL_FRAMEWORK_CONFIRMATION', 'selection_seed': SEED,
        'weeks': weeks, 'families': families, 'methods': methods, 'execution_order': order, 'repeats': 2, 'serial_workers': 1,
        'method_freeze': {'ranking': 'Author library frequency order (STATIC_ID), same order for every arm.',
            'initial': 128, 'batch': 64, 'minimum_before_plateau': 256, 'stale_batches': 2,
            'safe_rho': 0.9, 'completion': 'All unqueried canonical representatives of incumbent substation when frozen FRAME triggers completion.',
            'tie': 'Original fp32 reward then smallest original action ID.', 'new_training': False},
        'timing': 'Two serial balanced repetitions; rotate four policies over dates and reverse each date order for repeat1. Original GRID55 controller/worker/timers/DC+safeQP caches. No concurrent heavy work.',
        'primary': 'Date-level actual full-controller time, no earlier failure, and complete-week operational cost versus same-batch FULL. Average timing repeats within each date before ratios.',
        'quality_threshold': 0.005, 'speed_threshold': 0.05, 'positive_dates_required': 2, 'minimum_comparable_dates': 2,
        'decision_rule': 'FRAME must have no earlier failure on all dates and all complete-week costs <=FULL+0.5%; mean equal-horizon control reduction >=5%, positive on >=2 dates, and >=2 complete comparable dates. Otherwise report QUALITY_FAILED, INSUFFICIENT_HORIZONS or WEAK_OR_MIXED; no retuning or replacement.',
        'first_safe_rule': 'If FIRST_SAFE_BEST also passes quality everywhere and its mean same-horizon time ratio to FRAME <=1.02, report it as competitive; no exclusive necessity claim.',
        'mechanism_rule': 'NO_LOCAL is one existing internal ablation. Report all local gains; supporting event requires removal of its quality failure or >=0.1%FULL cost improvement in at least one complete family. Not a global guarantee.',
        'baselines': 'FULL exhaustive reference and mature first-verified-safe policy with stronger best-admissible exhausted fallback. First-safe is a controlled search-policy adaptation of SoftGNN2025, not a reproduced published agent. Earlier PPO/SoftGNN/GNN tables remain supplementary and are not silently merged.',
        'statistics': 'Three date families are the evidence units; repeats are timing repeats. Show every date and both repeats. No small-sample significance or equivalence certification.',
        'permission': 'Controller receives current public state, action library and actually dispatched simulation feedback only. Test outcomes never enter ranking/control/fitting.',
        'scope': 'One existing grid; all remaining three fresh dates; four existing policies, 24 runs. No new models, parameter sweep, baseline expansion, or additional dates.',
        'caps': {'episode_s': 1200, 'wall_s': 4800, 'physical_steps': 50000, 'public_forecasts': 500000, 'new_fits': 0},
        'sealed_use': 'All variants of these dates are excluded from training/tuning. After this stage all are exposed and never reused as unseen.',
        'cache_attribution': 'Common DC/safeQP compile reuse is available to every method and not credited as unique search/GNN novelty.'}
    OUT.mkdir()
    for name, value in [('reservation.json', reservation), ('design.json', design)]:
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    # Reuse the qualified runner and policy adapter without changing search.
    source = (ROOT / 'work/grid61/run.py').read_text(encoding='utf-8')
    source = source.replace('outputs/grid61', 'outputs/grid62').replace('work/grid61', 'work/grid62')
    source = source.replace("'planned': 32", "'planned': 24").replace('len(summaries) == 32', 'len(summaries) == 24')
    source = source.replace('32-run', '24-run').replace('32 RUNS', '24 RUNS')
    (HERE / 'run.py').write_text(source, encoding='utf-8')
    (HERE / 'mechanism_adapter.py').write_text((ROOT / 'work/grid61/mechanism_adapter.py').read_text(encoding='utf-8'), encoding='utf-8')
    extraction = (ROOT / 'work/grid52/extract.py').read_text(encoding='utf-8').replace('outputs/grid52', 'outputs/grid62').replace('.grid52.partial', '.grid62.partial').replace('96serial', '24serial').replace('FROZEN8', 'FROZEN3').replace('Frozen8', 'Frozen3')
    (HERE / 'extract.py').write_text(extraction, encoding='utf-8')
    freeze = read(ROOT / 'outputs/grid61/code_freeze.json')
    for p, h in freeze.items():
        assert sha(ROOT / p) == h, p
    for p in list(HERE.glob('*.py')) + [OUT / 'design.json', OUT / 'reservation.json', ROOT / 'outputs/grid61/review.json']:
        if p.suffix == '.py':
            py_compile.compile(str(p), doraise=True)
        freeze[p.relative_to(ROOT).as_posix()] = sha(p)
    (OUT / 'code_freeze.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    (OUT / 'preflight.json').write_text(json.dumps({'passed': True, 'runs': 24, 'families': 3, 'methods': 4, 'new_fits': 0,
        'inherited_permission_checks': 104, 'inherited_original_frame_replay_checks': 90,
        'outcomes_not_accessed': True, 'reservation_before_extraction': True}, indent=2), encoding='utf-8')
    (OUT / 'RUN_STATE.md').write_text('FROZEN BEFORE EXTRACTION:24 serial runs, three remaining fresh dates, four unchanged policies, no training.\n', encoding='utf-8')
    print(json.dumps({'frozen': True, 'runs': 24, 'weeks': weeks, 'new_fits': 0}))


if __name__ == '__main__':
    main()
