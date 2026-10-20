"""Freeze a small label-validity study before any physical replay."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid95_labels'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'design.json').exists()
    refs=read(ROOT/'outputs/grid91_mechanism/design.json')['references']
    chosen=[('C0_SAME_ACTION','2035-07-09_2',100,'same-choice query omission control'),
        ('C1_LOW_RHO','2035-07-09_2',709,'first divergence in historically completion-favored whole-policy comparison'),
        ('C2_OVERLOAD','2035-07-09_2',1901,'overload case omitted by rejected GRID93 gate'),
        ('C3_COST_REVERSE','2036-07-07_10',113,'first divergence in historically no-completion-favored whole-policy comparison')]
    ss=importlib.util.spec_from_file_location('g95_original',ROOT/'work/grid91_mechanism/policies.py');P=importlib.util.module_from_spec(ss);ss.loader.exec_module(P)
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];cases=[]
    for cid,week,step,why in chosen:
        ref=next(x for x in refs if x['week']==week)
        p=next(p for p in sorted((ROOT/ref['frame']).glob('call[0-9][0-9][0-9].json')) if read(p)['step']==step)
        c=read(p);truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):v for k,v in c['aliases'].items()}
        no=P.run(c['order'],lambda i,phase:dict(truth[i],phase=phase),'NO_LOCAL',subs,aliases)
        assert no['completion_budget']>0
        with np.load(p.with_name(p.stem+'_state.npz')) as z:
            state=z['observation'];base=z['base_action']
        assert digest(state)==c['state_hash']
        cases.append(dict(id=cid,week=week,cohort=ref['cohort'],step=step,reason_for_selection=why,
            environment=ref['environment'],reference=ref['frame'],source=p.relative_to(ROOT).as_posix(),
            state_hash=c['state_hash'],base_action_hash=digest(base),completion_budget=no['completion_budget'],
            before_choice=no['chosen_pool_id'],after_choice=c['chosen_pool_id']))
    assert cases[0]['before_choice']==cases[0]['after_choice']
    assert all(c['before_choice']!=c['after_choice'] for c in cases[1:])
    shared=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,policy='FRAME_GUARD',order_mode='STATIC_ID',repeat=0,stage_phase='all')
    items=[]
    for week in ['2035-07-09_2','2036-07-07_10']:
        cc=[c for c in cases if c['week']==week]
        items.append(dict(shared,week=week,environment=cc[0]['environment'],method='FRAME_LABEL_QUAL',watch_cases=[c['id'] for c in cc],omit_case=None,purpose='full factual qualification'))
    for c in cases:
        items.append(dict(shared,week=c['week'],environment=c['environment'],method='SKIP_'+c['id'],watch_cases=[c['id']],omit_case=c['id'],purpose='single completion omission, original FRAME continuation'))
    d=dict(stage='GRID95_SINGLE_COMPLETION_LABEL_VALIDITY',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='2026-10-20 user: 我们先研究标签这一块吧，学习先放一下。',
        learning_paused=True,new_fits=0,cases=cases,execution_order=items,
        reconstruction='Fresh identical seeded environment and agent, full unmodified physical/controller prefix replay. No observation-only reset or opaque environment/agent copying.',
        intervention='Exactly one original completion is omitted at the fixed step and verified state/base hashes. Global prefix queries unchanged; all subsequent searches use unchanged FRAME_GUARD.',
        qualification='First two fresh factual runs must reproduce all historical actions, observations and raw costs. Branch prefix and reviewed controller/optimizer/RNG witnesses must match fresh factual before the intervention. The same-choice omission must reproduce the entire physical trajectory.',
        common_future='Same frozen chronics files, environment/agent seeds0 and original external process. Realized opponent attack fields are recorded every step. Divergence of external attack realizations is reported, never described as an identical external-event pair.',
        horizon=dict(primary='until original episode end, including intervention transition',diagnostic_steps=[1,12,288],dt_minutes=5,
            rule='All horizons fixed before results; clip only to original episode end. Short-window labels never substituted for remaining-week labels.'),
        label=dict(raw_vector=['survival/complete','cost_SKIP_minus_COMPLETE','all_forecasts_COMPLETE_minus_SKIP','candidate_queries_COMPLETE_minus_SKIP'],
            rules='Survival differences reported first; cumulative cost compared only when both reach the requested common horizon. When cost and queries favor different branches, label TRADEOFF rather than force a binary class. No money/time conversion coefficient fitted.',
            numeric_equality='Only sum-rounding tolerance max(0.01,32*float64_eps*sum_absolute_costs) for reported-cost equality. Not a safety or practical-equivalence threshold.',
            external_mismatch='Raw outcomes remain reported; fixed-external-event label marked UNRESOLVED_EXTERNAL_DIFFERENCE if realized opponent attacks differ over common live steps.',
            scope='Single-intervention effect under original FRAME continuation on four selected exposed states; not value under arbitrary future learned gates and not population-wide training labels.'),
        timing='Instrumentation and reconstruction study only. No online speed comparison; query counts retained separately.',
        caps=dict(episode_s=1200,wall_s=7200,physical_steps=6*2017,public_forecasts=100000),
        forbidden=['training','threshold tuning','new gate','changing controller/reward/order/parameters','selecting extra cases after results','independent AC audit claim'])
    save('design.json',d)
    codes=read(ROOT/'outputs/grid93_trigger/code_freeze.json');data=read(ROOT/'outputs/grid93_trigger/data_freeze.json')
    for rel,h in codes.items():assert sha(ROOT/rel)==h,rel
    for rel,h in data.items():assert sha(ROOT/rel)==h,rel
    for c in cases:
        for p in (ROOT/c['reference']).iterdir():
            if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in W.glob('*.py'):codes[p.relative_to(ROOT).as_posix()]=sha(p)
    save('preflight.json',dict(passed=True,cases=len(cases),qualification_runs=2,single_omission_runs=4,
        exact_same_action_control=True,original_global_query_replays=len(cases),new_fits=0,new_physical_steps=0))
    for n in ['design.json','preflight.json']:codes[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
    save('data_freeze.json',data);codes[(OUT/'data_freeze.json').relative_to(ROOT).as_posix()]=sha(OUT/'data_freeze.json')
    save('code_freeze.json',codes);save('manifest.json',dict(planned=6,summaries=[]))
    repo=ROOT/'work/GRL-results-integration';dst=repo/'research_archive/power-grid-2026/protocols/GRID95_labels/frozen';dst.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[OUT/n for n in ['design.json','preflight.json','code_freeze.json','data_freeze.json']]:shutil.copy2(p,dst/p.name)
    note='用户暂停学习，先做标签有效性。固定4状态（同动作校验、低ρ正例、过载、成本反向），2条原FRAME资格重现+4条只取消一次补查，之后沿原FRAME。完整前缀重建控制器记忆，核对原动作/观测/成本、数值记忆与外部攻击。主时域剩余整周，1步/1小时/1天仅诊断；输出存活/成本/查询向量和权衡，不合成任意权重标签，不训练。'
    for n in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
        p=repo/n;s=p.read_text(encoding='utf-8');p.write_text(s.rstrip()+'\n\n## 2026-10-20 — GRID95_LABELS_frozen\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=cases,code_items=len(codes),data_items=len(data),planned=6),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
