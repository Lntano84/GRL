"""Prepare same-date fixed-prefix mechanism comparison before physical replay."""
from pathlib import Path
import datetime as dt,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid86';WORK=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();assert read(ROOT/'outputs/grid84/review.json')['passed_saved_data']
    assert read(ROOT/'outputs/grid85_mechanism/verification.json')['passed']
    for n in ['run.py','mechanism_adapter.py','environment.py']:
        src=ROOT/f'work/grid84/{n}';text=src.read_text(encoding='utf-8').replace('grid84','grid86')
        if n=='run.py':text=text.replace('new_fits=3','new_fits=0')
        (WORK/n).write_text(text,encoding='utf-8')
    text=(ROOT/'work/grid84/audit.py').read_text(encoding='utf-8').replace('grid84','grid86')
    text=text.replace('assert len(S)==72','assert len(S)==36').replace('runs=72,','runs=36,').replace('new_fits=3,','new_fits=0,')
    text=text.replace("if s['method'] in ['FRAME_GUARD','META_CONTEXT0','META_CONTEXT1']:","if s['method']=='FRAME_GUARD':")
    text=text.replace("for baseline in [f'META_CONTEXT{i}' for i in range(5)]:","for baseline in ['FIXED_256','FIXED_512']:")
    text=text.replace('COMPLETE:72 balanced runs;', 'COMPLETE:36 balanced runs;')
    text=text.replace('Same six exposed generated scenarios, fixed-guard and five same-configuration ICAPS2022-core adaptation seeds; old0/1 retained, only2/3/4 fitted; no independent confirmation, original calibration limitation retained; no cross-batch timing claim.', 'Same six exposed generated scenarios, final guard and literal fixed256/512 prefixes. No new training, independent confirmation, or cross-batch timing claim.')
    # Independent arithmetic verifies the literal budget and best observed choice.
    marker="            assert seen==[v['pool_id'] for v in c['outcomes']];checks['causal_calls']+=1"
    assert text.count(marker)==1
    addition="            if s['method'].startswith('FIXED_'):\n                cap=int(s['method'].split('_')[1]);assert seen==c['order'][:cap]\n                legal=[v for v in c['outcomes'] if v['strict_admissible']]\n                independent=max(legal,key=lambda v:(float(np.float32(v['reward'])),-v['pool_id']),default=None)\n                assert c['chosen_pool_id']==(None if independent is None else independent['pool_id'])\n"
    text=text.replace(marker,addition+marker);(WORK/'audit.py').write_text(text,encoding='utf-8')
    old=read(ROOT/'outputs/grid82/design.json');base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
    methods=[dict(base,method=n,policy=n) for n in ['FRAME_GUARD','FIXED_256','FIXED_512']]
    weeks=old['weeks'];order=[]
    for repeat in (0,1):
        for w in weeks if repeat==0 else weeks[::-1]:
            j=weeks.index(w);block=methods[j%3:]+methods[:j%3]
            if repeat:block=block[::-1]
            order.extend(dict(m,week=w,repeat=repeat,stage_phase='all') for m in block)
    D=dict(stage='GRID86_CLOSED_LOOP_FIXED_PREFIX_MECHANISM',authorization='2026-10-16 user: 开始吧 after GRID85 fixed256/512 vs FRAME proposal.',
           frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),weeks=weeks,methods=methods,execution_order=order,repeats=2,pilot=[],
           caps=dict(episode_s=1200,wall_s=10800,physical_steps=90000,public_forecasts=2000000,new_fits=0),
           quality_threshold=old['quality_threshold'],timing=old['timing'],**{k:old[k] for k in ['full_reference']},
           fixed_policy='Query first min(cap,M) STATIC_ID representatives regardless of feedback; return best strict-admissible with float32 reward/lowest-ID tie. No local completion, safety-triggered extension or additional query after cap. With none, return null to unchanged base controller.',
           guard_policy='Exact work/grid70/policies.py FRAME_GUARD, same 128/64/256/two-stale/.9 settings and alias completion.',
           fairness='Common grid, base controller, admissibility, candidate order/aliases, public prediction interface, context construction, DC/safeQP/prior metadata caches. All per-decision work included in control time.',
           scope='Six GRID81-generated already exposed dates; development mechanism test of fixed versus feedback query allocation. Timing repeats are not independent dates; no original conference task comparison in this batch.',
           decision='Quality criterion unchanged: no earlier termination than FULL and complete-week cost<=FULL*1.005 where both complete. For EACH fixed baseline, supported equal-quality computational advantage requires guard and baseline both6/6 quality, >=5% date-macro controller reduction, >=4/6 positive dates, both timing repeats positive, and >=5% query reduction on matched nonzero-query horizons. If baseline fails quality report the quality difference and descriptive computing, not same-quality speed. If either fixed baseline passes quality and is within2% of guard or faster on mean date-relative control and uses<=102% queries on all matched active horizons (macro), accept that cheap fixed alternative as competitive. Other cases mixed/uncertain; report both budgets, do not select a new cap, tune rules, add dates or retune models.',
           timing_horizon='Whole-six-date same-quality speed/cheap-competitive claims additionally require all six paired trajectories to have identical length and completion status. If one survives longer, report survival benefit and matched-horizon timing separately, not speed from premature termination.',
           retention='All dates, failures and fixed budgets retained; no post-hoc best checkpoint/budget. Older studies and user manuscript remain unchanged.')
    OUT.mkdir();save(OUT/'design.json',D);shutil.copy2(ROOT/'outputs/grid84/data_freeze.json',OUT/'data_freeze.json')
    frozen=read(ROOT/'outputs/grid82/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(WORK.glob('*.py'))+[OUT/'design.json',OUT/'data_freeze.json']:
        frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    pack=ROOT/'outputs/grid85_mechanism/mechanism_feedback.json.gz';frozen[pack.relative_to(ROOT).as_posix()]=sha(pack)
    save(OUT/'code_freeze.json',frozen)
    (OUT/'RUN_STATE.md').write_text('FROZEN:FRAME_GUARD vs fixed256 and fixed512;36 serial balanced replays; no training. Preflight pending.\n',encoding='utf-8')
    print(json.dumps(dict(runs=36,dates=6,repeats=2,new_fits=0)))
if __name__=='__main__':main()
