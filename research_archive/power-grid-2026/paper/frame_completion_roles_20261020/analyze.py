"""Describe completion roles using frozen saved feedback; never invokes a simulator."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, importlib.util, json, math, statistics

ROOT = Path(__file__).resolve().parents[2]
W = Path(__file__).parent
OUT = ROOT / 'outputs/frame_completion_roles_20261020'
SOURCES = {}
EPS = 1e-6  # Existing GRID92 diagnostic reporting tolerance, not a policy setting.
ROLES = ['no_incumbent', 'overload_remaining', 'below_limit_without_buffer', 'buffer_reached']


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def source(p):
    SOURCES[p.relative_to(ROOT).as_posix()] = digest(p)
    return p


def read(p): return json.loads(source(p).read_text(encoding='utf-8'))


def readcsv(p):
    with source(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csvsave(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)


def module(name, rel):
    sp = importlib.util.spec_from_file_location(name, source(ROOT/rel))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def role(rho):
    if rho is None: return ROLES[0]
    return ROLES[1] if rho >= 1 else ROLES[2] if rho >= .9 else ROLES[3]


def mean(xs): return statistics.mean(xs) if xs else None


def best(rows, P): return max((r for r in rows if r['strict_admissible']), key=P.key, default=None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save('plan.json', read(W/'plan.json')); source(Path(__file__))
    P = module('roles_policy', 'work/grid91_mechanism/policies.py')
    I = module('roles_reference', 'work/grid89_review/reference_logic.py')
    refs = read(ROOT/'outputs/grid91_mechanism/design.json')['references']
    subs = read(ROOT/'outputs/grid28/offline.json')['public_subids']
    review91=read(ROOT/'outputs/grid91_mechanism/review.json')
    assert review91['passed_saved_data']
    assert read(ROOT/'outputs/grid93_trigger/review.json')['passed_saved_data']
    prior_full = {(r['cohort'],r['week'],r['call']):r for r in readcsv(ROOT/'outputs/grid92_completion_analysis/per_state.csv')}
    prior_screen = {(r['cohort'],r['week'],r['call']):r for r in readcsv(ROOT/'outputs/grid93_trigger/actual_frame_state_screen.csv')}
    rows = []; checks = defaultdict(int); field_intersection=None; field_union=set()
    for ref in refs:
        for population, field in [('FULL','full'),('FRAME','frame')]:
            for path in sorted((ROOT/ref[field]).glob('call[0-9][0-9][0-9].json')):
                c = read(path); truth = {r['pool_id']:r for r in c['outcomes']}
                for o in c['outcomes']:
                    field_union.update(o)
                    field_intersection=set(o) if field_intersection is None else field_intersection.intersection(o)
                    assert o['strict_admissible']==bool(0<o['rho']<c['rho_limit'] and not o['done'] and not o['illegal'] and not o['ambiguous'] and not o['exceptions'])
                    checks['candidate_feedback_validity']+=1
                aliases = {int(k):v for k,v in c['aliases'].items()}
                got = {}
                for policy in ['NO_LOCAL','FRAME_GUARD'] + (['GLOBAL_MATCH'] if population=='FULL' else []):
                    seen = []
                    def reveal(i, phase):
                        assert i not in seen and i in truth; seen.append(i)
                        return dict(truth[i], phase=phase)
                    z = P.run(c['order'], reveal, policy, subs, aliases)
                    independent, chosen, budget = I.replay(c['order'], truth, subs, aliases, policy, dict(P.DEFAULT,qmin=0))
                    assert seen == independent and chosen == z['chosen_pool_id'] and budget == z['completion_budget']
                    got[policy] = z; checks['independent_policy_replays'] += 1
                nl, lc = got['NO_LOCAL'], got['FRAME_GUARD']
                before, after = best(nl['outcomes'],P), best(lc['outcomes'],P)
                rb, ra = (None if v is None else v['rho'] for v in [before,after])
                gain = None if before is None else P.key(after)[0]-P.key(before)[0]
                q = lc['completion_budget']; assert q == lc['public_queries']-nl['public_queries']
                if population=='FRAME':
                    assert lc['chosen_pool_id']==c['chosen_pool_id'] and lc['public_queries']==c['public_queries']
                    assert [r['pool_id'] for r in lc['outcomes']]==[r['pool_id'] for r in c['outcomes']]
                    checks['actual_frame_reproductions'] += 1
                row = dict(population=population,cohort=ref['cohort'],week=ref['week'],call=path.stem,step=c['step'],
                    source=path.relative_to(ROOT).as_posix(),state_hash=c['state_hash'],role=role(rb),after_role=role(ra),
                    incumbent_rho=rb,after_rho=ra,global_queries=nl['public_queries'],completion_queries=q,
                    completion_forecast_s=sum(r['forecast_s'] for r in lc['outcomes'] if r['phase']=='local_completion'),
                    triggered=bool(q),choice_changed=nl['chosen_pool_id']!=lc['chosen_pool_id'],
                    before_choice=nl['chosen_pool_id'],after_choice=lc['chosen_pool_id'],reward_gain=gain,
                    reward_improved=bool(gain is not None and gain>EPS),
                    crossed_1=bool(rb is not None and rb>=1 and ra<1),
                    crossed_09=bool(rb is not None and rb>=.9 and ra<.9),
                    global_gain=None,local_minus_global=None,full_rho=None,full_can_cross_1=None,full_can_cross_09=None,
                    veto=False,veto_score=None,veto_saved_queries=0,veto_lost_gain=None)
                if population=='FULL':
                    prev=prior_full[(ref['cohort'],ref['week'],path.stem)]
                    assert (prev['triggered']=='True')==bool(q)
                    if q: assert abs(float(prev['local_gain'])-gain)<1e-12
                    gl=got['GLOBAL_MATCH']; assert gl['public_queries']==lc['public_queries']
                    bg=best(gl['outcomes'],P); bf=best(c['outcomes'],P)
                    row['full_rho']=None if bf is None else bf['rho']
                    if before is not None:
                        row.update(global_gain=P.key(bg)[0]-P.key(before)[0],
                            local_minus_global=P.key(after)[0]-P.key(bg)[0],
                            full_can_cross_1=bool(rb>=1 and bf['rho']<1),
                            full_can_cross_09=bool(rb>=.9 and bf['rho']<.9))
                    checks['grid92_rows_reconciled'] += 1
                else:
                    prev=prior_screen[(ref['cohort'],ref['week'],path.stem)]
                    assert q==int(prev['completion_budget'])
                    assert gain is None or abs(float(prev['local_gain'])-gain)<1e-12
                    row.update(veto=prev['veto']=='True',veto_score=float(prev['score']) if prev['score'] else None,
                        veto_saved_queries=int(prev['saved_queries']),
                        veto_lost_gain=float(prev['reward_lost']) if prev['reward_lost'] else None)
                    checks['grid93_rows_reconciled'] += 1
                if before is None: assert not q and after is None
                else: assert gain >= 0
                rows.append(row)
    assert sum(r['population']=='FULL' for r in rows)==185
    assert sum(r['population']=='FRAME' for r in rows)==179
    summary=[]
    for population in ['FRAME','FULL']:
        for cohort in ['old13','G1','G2','ALL']:
            for rr_role in ROLES:
                rr=[r for r in rows if r['population']==population and (cohort=='ALL' or r['cohort']==cohort) and r['role']==rr_role]
                tr=[r for r in rr if r['triggered']]
                summary.append(dict(population=population,cohort=cohort,role=rr_role,calls=len(rr),triggered=len(tr),
                    triggered_scenarios=len({r['week'] for r in tr}),queries=sum(r['completion_queries'] for r in tr),
                    measured_forecast_s=sum(r['completion_forecast_s'] for r in tr),
                    reward_improved=sum(r['reward_improved'] for r in tr),choice_changed=sum(r['choice_changed'] for r in tr),
                    gain_sum=sum(r['reward_gain'] for r in tr),mean_gain=mean([r['reward_gain'] for r in tr]),
                    crossed_1=sum(r['crossed_1'] for r in tr),crossed_09=sum(r['crossed_09'] for r in tr),
                    no_reward_gain_queries=sum(r['completion_queries'] for r in tr if not r['reward_improved']),
                    local_wins=sum(r['local_minus_global']>EPS for r in tr) if population=='FULL' else None,
                    global_wins=sum(r['local_minus_global'] < -EPS for r in tr) if population=='FULL' else None,
                    equal=sum(abs(r['local_minus_global'])<=EPS for r in tr) if population=='FULL' else None,
                    full_can_cross_1=sum(bool(r['full_can_cross_1']) for r in tr) if population=='FULL' else None,
                    full_can_cross_09=sum(bool(r['full_can_cross_09']) for r in tr) if population=='FULL' else None,
                    veto_calls=sum(r['veto'] for r in tr),veto_removes_improvement=sum(r['veto'] and r['reward_improved'] for r in tr),
                    veto_saved_queries=sum(r['veto_saved_queries'] for r in tr)))
    csvsave('per_call.csv', rows); csvsave('role_summary.csv',summary)
    cases=readcsv(ROOT/'outputs/grid92_completion_analysis/closed_loop_cases.csv')
    evidence=read(ROOT/'outputs/grid92_completion_analysis/first_divergence_evidence.json')
    cases_lookup={(r['cohort'],r['week']):r for r in cases}; joined=[]
    for e in evidence:
        src=e['frame_source']; matches=[r for r in rows if r['population']=='FRAME' and r['source']==src]
        assert len(matches)==1; r=matches[0]; c=cases_lookup[(r['cohort'],r['week'])]
        assert abs(r['incumbent_rho']-e['old_outcome']['rho'])<1e-12
        assert abs(r['after_rho']-e['new_outcome']['rho'])<1e-12
        assert c['divergence_step'] and int(c['divergence_step'])==r['step']
        same_delivery=c['predicted_combined_equals_delivered_frame']=='True' and c['predicted_combined_equals_delivered_no']=='True'
        joined.append(dict(cohort=r['cohort'],week=r['week'],step=r['step'],role=r['role'],rho_before=r['incumbent_rho'],
            rho_after=r['after_rho'],reward_gain=r['reward_gain'],extra_queries=r['completion_queries'],
            both_complete=c['complete']=='True',frame_minus_no_cost=float(c['frame_minus_no_cost']),
            no_local_vs_frame_pct=float(c['no_local_vs_frame_pct']) if c['no_local_vs_frame_pct'] else None,
            candidate_is_delivered_in_both=same_delivery,later_different_actions=int(c['later_different_actions']),
            full_week_cost_direction=('FRAME_lower' if float(c['frame_minus_no_cost']) < -1e-6 else 'FRAME_higher' if float(c['frame_minus_no_cost'])>1e-6 else 'equal') if c['complete']=='True' else 'incomparable'))
        checks['first_divergences_reconciled'] += 1
    assert len(joined)==14
    csvsave('first_divergence_roles.csv',joined)
    quality_lost=[]
    for comp in review91['comparisons']:
        if comp['frame_quality'] and not comp['no_local_quality']:
            case=next(x for x in joined if x['cohort']==comp['cohort'] and x['week']==comp['week'])
            quality_lost.append(dict(**case,frame_quality=True,no_local_quality=False,
                frame_vs_full_cost_pct=100*comp['frame_cost_delta_full'],
                no_local_vs_full_cost_pct=100*comp['no_local_cost_delta_full']))
    assert len(quality_lost)==4 and sum(x['role']=='buffer_reached' for x in quality_lost)==3
    csvsave('quality_loss_first_divergences.csv',quality_lost)
    failure=read(ROOT/'outputs/grid93_trigger/failure_analysis.json')[0]
    fsource=failure['sources'][0]['path']; fr=next(r for r in rows if r['source']==fsource and r['population']=='FRAME')
    assert fr['role']=='overload_remaining' and fr['veto'] and fr['reward_improved']
    july=next(r for r in joined if r['week']=='2036-07-07_10')
    assert july['role']=='buffer_reached' and july['full_week_cost_direction']=='FRAME_higher'
    # Class boundaries are definitions, not fitted bins. Check boundary treatment separately.
    assert [role(x) for x in [None,.899999,.9,.999999,1.,1.1]]==[ROLES[0],ROLES[3],ROLES[2],ROLES[2],ROLES[1],ROLES[1]]
    checks['classification_boundaries']=6
    # Candidate feedback currently cannot identify the economic value of a new action.
    save('evidence_summary.json',dict(groups=summary,first_divergence_roles=joined,
        quality_loss_first_divergences=quality_lost,
        rejected_trigger_failure=fr,negative_cost_case=july,common_outcome_fields=sorted(field_intersection),all_outcome_fields=sorted(field_union),
        interpretation='Forecast risk categories describe why further search may matter. They do not define a validated gate or guarantee downstream safety or cost.'))
    save('verification.json',dict(passed=True,checks=dict(checks),actual_calls=179,full_diagnostic_calls=185,
        new_forecasts=0,new_closed_loop_runs=0,new_training=0,new_trigger_policies_tested=0,
        historical_outcomes_reused=True,limitations='All scenes exposed; calls correlated; FULL and FRAME populations must not be pooled; no AC re-solve or fresh timing comparison.'))
    for p in [ROOT/'work/grid91_mechanism/adapter.py',ROOT/'work/grid03/score_adapter.py']:source(p)
    assert all(digest(ROOT/p)==h for p,h in SOURCES.items())
    save('source_receipt.json',dict(sources=SOURCES,all_source_hashes_unchanged=True))
    print(json.dumps(dict(actual=[x for x in summary if x['population']=='FRAME' and x['cohort']=='ALL'],
        full=[x for x in summary if x['population']=='FULL' and x['cohort']=='ALL'],first_divergences=joined,checks=dict(checks)),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
