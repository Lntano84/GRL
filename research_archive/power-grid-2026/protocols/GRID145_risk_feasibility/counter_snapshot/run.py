"""Read-only eligibility checks for risk-controlled sequential search; no policy selection."""
from pathlib import Path
import collections, datetime, gzip, hashlib, importlib.util, json, math, shutil, sys
import numpy as np

R = Path('D:/Codex/GRL研究')
W = Path(__file__).parent
O = R / 'outputs/grid145_risk_feasibility'
REPO = R / 'work/GRL-results-integration'
A = REPO / 'research_archive/power-grid-2026/protocols/GRID145_risk_feasibility'
sys.path.insert(0, str(R / 'work/grid144_class_mass'))
import support as S
B = S.B
MODELS = {'RF140': B.O, 'RF142': S.OTHER, 'RF144': S.O}

def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(name, data):
    (O / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
def rows_at(p):
    with gzip.open(p, 'rt', encoding='utf8') as f: return json.load(f)
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify():
    S.verify_trained()
    api = S.diagnostic_api()
    api.verify()
    for name in ['source_hashes.json', 'code_freeze.json']:
        for p, h in read(O / name).items(): assert sha(R / p) == h, p

def archive(phase):
    d = A / phase
    d.mkdir(parents=True, exist_ok=False)
    for p in [*W.glob('*.py'), *O.iterdir()]:
        if p.is_file(): shutil.copy2(p, d / p.name)
    (d / 'archive_manifest.json').write_text(json.dumps({p.name: sha(p) for p in d.iterdir()}, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

def prepare():
    assert not O.exists()
    O.mkdir()
    S.verify_trained()
    S.diagnostic_api().verify()
    save('design.json', dict(id='GRID145_RISK_CONTROL_ELIGIBILITY', utc=stamp(),
        task='Read primary early-exit/risk-control papers and diagnose existing fixed score families before any further training.',
        models=list(MODELS), partitions=['original_fit', 'original_calibration'],
        rows='386fit +198calibration prefixes,161fit +82calibration complete searches;three existing score arrays. No development-validation score loading/replay.',
        diagnostic='Analytic exhaustive piecewise policy behavior for threshold domain[0.5,1]. Only record-high chronological scores can be first exits. Enumerate these mathematical intervals, not candidate policies for selection. Keep every interval and every loss/query nonmonotonicity.',
        loss='Action mismatch to originalFRAME after unchanged completion;also joint mismatch or non-saving learned stop. Original fallback loss zero. Labels stay unchanged.',
        audit='At each interval lower boundary and immediately below upper boundary,check against existing whole-search replay. Check monotone future-sup loss envelope and all-prefix unsafe envelope domination. Inspect only existing calibration-history thresholds;no new cutoff adopted.',
        statistics='Report date clusters,not iid prefixes. Illustrative finite-sample calculations assume independent Bernoulli date-any-error units for a pre-fixed policy;not a certificate for these clustered,previously exposed records.',
        literature=['CALM NeurIPS2022 sections4.1-4.3','Conformal Risk Control ICLR2024 theorem1 and section2.3','Learn then Test arXiv2110.01052v5 sections1.1,2.1-2.3'],
        prohibited=['fit','predict','new physical simulation','new scenarios','validation threshold search','policy adoption','risk tolerance change','test8 generation','paper edits','scale-up']))
    sources = [O / 'design.json', B.O / 'fit_rows.json.gz', B.O / 'calibration_rows.json.gz', B.O / 'design.json', B.SOURCE / 'call_ledger.json', R / 'work/grid140_global_compression/common.py', R / 'work/grid143_continue_risk/run.py', R / 'work/grid144_class_mass/support.py']
    for root in MODELS.values():
        sources += [root / name for name in ['train_scores.npz', 'cutoff.json', 'calibration_history.json', 'trained_freeze.json', 'review.json']]
    save('source_hashes.json', {p.relative_to(R).as_posix(): sha(p) for p in sources})
    save('code_freeze.json', {p.relative_to(R).as_posix(): sha(p) for p in W.glob('*.py')})
    verify()
    save('state.json', dict(stage='frozen_read_only'))
    archive('frozen')
    print('GRID145_FROZEN')

def completion_at(ref, n):
    c = B.call_at(ref['source'])
    prefix = c['outcomes'][:n]
    b = B.P.best(prefix)
    initial = B.P.best(c['outcomes'][:min(128, len(c['order']))])
    extra = []
    if b is not None and n < len(c['order']) and (initial is None or initial['rho'] >= .9 or B.P.bid(b) != B.P.bid(initial)):
        station = B.subids()[b['pool_id']]
        ids = {int(rep) for alias, rep in c['aliases'].items() if B.subids()[int(alias)] == station}
        extra = [v for v in c['outcomes'][n:] if v['pool_id'] in ids]
        b = B.P.best(prefix + extra)
    q = n + len(extra)
    original_q = ref['global_n'] + ref['local_queries']
    return dict(global_queries=n, local_queries=len(extra), queries=q,
                selected=B.P.bid(b), original_selected=ref['frame_choice'],
                action_loss=int(B.P.bid(b) != ref['frame_choice']),
                joint_loss=int(B.P.bid(b) != ref['frame_choice'] or q >= original_q))

def regimes(ref, checkpoints):
    lower = .5
    out = []
    for row, score in checkpoints:
        if score <= lower: continue
        n = row['meta']['visible_count']
        assert n < ref['global_n']
        outcome = completion_at(ref, n)
        assert (outcome['joint_loss'] == 0) == bool(row['target']['stop_label'])
        out.append(dict(lo_inclusive=lower, hi_exclusive=score, early_stop=True, **outcome))
        lower = score
    # tau=1 always disables learned exits, including any unit score.
    out.append(dict(lo_inclusive=lower, hi_exclusive=None, early_stop=False,
                    global_queries=ref['global_n'], local_queries=ref['local_queries'],
                    queries=ref['global_n'] + ref['local_queries'],
                    selected=ref['frame_choice'], original_selected=ref['frame_choice'],
                    action_loss=0, joint_loss=0))
    for i, item in enumerate(out):
        item['future_sup_action_loss'] = max(x['action_loss'] for x in out[i:])
        item['future_sup_joint_loss'] = max(x['joint_loss'] for x in out[i:])
    return out

def at(intervals, t):
    return next(x for x in intervals if t >= x['lo_inclusive'] and (x['hi_exclusive'] is None or t < x['hi_exclusive']))

def run():
    verify()
    assert not (O / 'started.json').exists()
    save('started.json', dict(utc=stamp(), fits=0, predicts=0, physical=0))
    fit = rows_at(B.O / 'fit_rows.json.gz')
    cal = rows_at(B.O / 'calibration_rows.json.gz')
    assert len(fit) == 386 and len(cal) == 198
    D = read(B.O / 'design.json')
    summaries, all_calls, violations, history_rows, group_curves = [], [], [], [], []
    replay_checks = 0
    for model, root in MODELS.items():
        scores = np.load(root / 'train_scores.npz')
        for part, rows in [('fit', fit), ('calibration', cal)]:
            pp = scores[part]
            assert len(pp) == len(rows)
            sm = {(r['meta']['call_id'], r['meta']['visible_count']): float(p) for r, p in zip(rows, pp)}
            grouped = collections.defaultdict(list)
            for r, p in zip(rows, pp): grouped[r['meta']['call_id']].append((r, float(p)))
            dates = set(D['fit_dates'] if part == 'fit' else D['calibration_dates'])
            refs = B.refs('train', dates)
            entries = []
            action_up, joint_up, query_down, gain_lost = [], [], [], []
            for ref in refs:
                rr = sorted(grouped[ref['call_id']], key=lambda x: x[0]['meta']['visible_count'])
                seg = regimes(ref, rr)
                entry = dict(model=model, partition=part, call_id=ref['call_id'], date=ref['date'], scenario=ref['scenario'], intervals=seg)
                entries.append(entry)
                for x in seg:
                    ts = [x['lo_inclusive']]
                    if x['hi_exclusive'] is not None:
                        ts.append(float(np.nextafter(x['hi_exclusive'], -np.inf)))
                    else: ts.append(1.)
                    for t in ts:
                        assert x['lo_inclusive'] <= t <= 1.
                        actual = B.replay(ref, 'MODEL', t, sm)
                        assert actual['selected'] == x['selected'] and actual['queries'] == x['queries']
                        assert int(not actual['action_match']) == x['action_loss']
                        assert int(actual['bad_learned_stop']) == x['joint_loss']
                        # This envelope ignores reachability and is also monotone.
                        unsafe = any(p > t and not r['target']['stop_label'] for r, p in rr)
                        assert int(unsafe) >= x['joint_loss']
                        replay_checks += 1
                for before, after in zip(seg, seg[1:]):
                    aa = before['action_loss'] < after['action_loss']
                    jj = before['joint_loss'] < after['joint_loss']
                    qq = before['queries'] > after['queries']
                    if aa: action_up.append(ref['call_id'])
                    if jj: joint_up.append(ref['call_id'])
                    if qq: query_down.append(ref['call_id'])
                    if before['queries'] < ref['global_n'] + ref['local_queries'] and after['action_loss']: gain_lost.append(ref['call_id'])
                    if aa or jj or qq:
                        violations.append(dict(model=model, partition=part, call_id=ref['call_id'], date=ref['date'],
                            threshold_boundary=after['lo_inclusive'], action_loss_increases=aa,
                            joint_loss_increases=jj, query_count_decreases=qq, before=before, after=after))
                assert all(a['future_sup_action_loss'] >= b['future_sup_action_loss'] for a, b in zip(seg, seg[1:]))
                assert all(a['future_sup_joint_loss'] >= b['future_sup_joint_loss'] for a, b in zip(seg, seg[1:]))
            # Exact group loss function, diagnostic only. No threshold selected from it.
            bounds = sorted({x['lo_inclusive'] for e in entries for x in e['intervals']} | {1.})
            curve = []
            for t in bounds:
                selected = [(e, at(e['intervals'], t)) for e in entries]
                date_means = [sum(x['action_loss'] for e, x in selected if e['date'] == d) / sum(e['date'] == d for e, x in selected) for d in sorted(dates)]
                curve.append(dict(threshold=t, mismatching_calls=sum(x['action_loss'] for e, x in selected),
                                  date_macro_mismatch=sum(date_means)/len(date_means),
                                  dates_with_any_mismatch=sum(any(x['action_loss'] for e, x in selected if e['date'] == d) for d in dates)))
            group_curves.append(dict(model=model, partition=part, diagnostic_only=True, curve=curve))
            summaries.append(dict(model=model, partition=part, prefixes=len(rows), calls=len(refs), dates=len(dates),
                calls_with_eligible_prefixes=len(grouped), interval_count=sum(len(e['intervals']) for e in entries),
                action_loss_nonmonotone_calls=len(set(action_up)), action_loss_up_transitions=len(action_up),
                joint_loss_nonmonotone_calls=len(set(joint_up)), query_nonmonotone_calls=len(set(query_down)),
                pooled_error_count_up_transitions=sum(a['mismatching_calls'] < b['mismatching_calls'] for a, b in zip(curve, curve[1:])),
                date_macro_error_up_transitions=sum(a['date_macro_mismatch'] < b['date_macro_mismatch'] - 1e-12 for a, b in zip(curve, curve[1:]))))
            all_calls.extend(entries)
            if part == 'calibration':
                for h in read(root / 'calibration_history.json'):
                    t = h['threshold']
                    cc = [dict(call_id=e['call_id'], date=e['date'], **at(e['intervals'], t)) for e in entries]
                    assert sum(c['action_loss'] for c in cc) == h['summary']['action_differences']
                    history_rows.append(dict(model=model, threshold=t, action_differences=sum(c['action_loss'] for c in cc),
                        future_sup_action_failures=sum(c['future_sup_action_loss'] for c in cc),
                        future_sup_joint_failures=sum(c['future_sup_joint_loss'] for c in cc), calls=cc))
    save('family_summary.json', summaries)
    save('all_call_threshold_intervals.json', all_calls)
    save('all_nonmonotone_transitions.json', violations)
    save('group_loss_diagnostic_curves.json', group_curves)
    save('existing_calibration_threshold_checks.json', history_rows)
    # Planning illustration only: fixed-policy iid Bernoulli date-any-error loss.
    illustrations = [dict(independent_date_units=n, zero_failure_one_sided_95_upper=1-math.exp(math.log(.05)/n),
                          crc_unit_bound_correction=1/(n+1)) for n in [4, 20, 32, 59, 100, 299]]
    for x in illustrations:
        assert abs((1-x['zero_failure_one_sided_95_upper'])**x['independent_date_units']-.05) < 1e-12
    save('sample_size_illustration.json', dict(not_current_confidence_certificate=True, adopted_risk_tolerance=None,
        loss_unit='Any action mismatch in a date for a pre-fixed policy. iid dates assumed ONLY for illustration.',
        rows=illustrations, minimum_zero_failure_dates_95pct=[dict(illustrative_upper=r, n=math.ceil(math.log(.05)/math.log(1-r))) for r in [.1,.05,.01]],
        zero_future_risk='No finite sample of zero errors excludes every positive Bernoulli failure rate. Identity fallback needs no empirical certification.'))
    verify()
    save('review.json', dict(passed=True, models=3, prefixes_per_model=584, fit_calls=161, calibration_calls=82,
        interval_boundary_whole_search_checks=replay_checks, existing_histories_checked=len(history_rows),
        original_sources_models_labels_cutoffs_unchanged=True, loss_envelopes_checked=True,
        no_validation_scoring_or_threshold_replay=True, fits=0, predicts=0, physical=0, final8_absent=True))
    save('decision.json', dict(calibration_is_not_a_rank_repair=True,
        basic_crc_directly_eligible=not any(x['action_loss_nonmonotone_calls'] for x in summaries),
        formal_future_risk_guarantee_established=False, new_cutoff_adopted=False, risk_tolerance_changed=False,
        new_training_authorized_or_started=False, ready_to_scale=False,
        next='Keep original task and fixed models. Improve distinction of high-scoring continue cases before further fitting;future risk certification needs a pre-specified complete-search/date unit and fresh appropriate calibration sampling.'))
    save('state.json', dict(stage='analysed_pending_report'))
    print(json.dumps(dict(summary=summaries, audit=read(O/'review.json')), ensure_ascii=False))

if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()
