"""Freeze only after the six fits and causal learning/interface checks pass."""
import hashlib,json,py_compile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid65'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists()
    assert read(ROOT/'outputs/grid64/learning_qualification.json')['passed']
    t=read(ROOT/'outputs/grid64/training_finished.json');costindex=t['selected']['costindex']
    weeks=read(ROOT/'outputs/grid63/design.json')['weeks']
    methods=[dict(method=p,rule='STATIC_ID',budget=128,checkpoint=None,policy=p,cache_dc=True,cache_safe=True) for p in ['FULL','FRAME','GLOBAL_MATCH']]
    for seed in [0,1]:
        p=ROOT/f'outputs/grid64/models/cost{costindex}_seed{seed}/final.pt'
        methods.append(dict(method=f'META_DQN{seed}',rule='STATIC_ID',budget=128,checkpoint=None,policy=f'META_DQN{seed}',cache_dc=True,cache_safe=True,meta_model=p.relative_to(ROOT).as_posix(),meta_model_sha256=sha(p),meta_cost_index=costindex))
    order=[]
    for repeat in range(2):
      for dateindex,week in enumerate(weeks):
        offset=dateindex%5;sequence=methods[offset:]+methods[:offset]
        if repeat:sequence=sequence[::-1]
        order.extend({**m,'week':week,'repeat':repeat,'stage_phase':'published_stop_baseline'} for m in sequence)
    D={'stage':'GRID65_PUBLISHED_STOPPING_BASELINE_CLOSED_LOOP','weeks':weeks,'methods':methods,'execution_order':order,'repeats':2,'environment_seed':0,'serial_workers':1,
       'source':'ICAPS2022 Bhatia et al. stopping-only adaptation, Python port of official Double DQN core; not full original planner or tuning action reproduction.',
       'baseline_calibration':t['selected'],'baseline_calibration_quality_matched':t['calibration_quality_matched'],
       'timing':'70 serial fresh workers, two repeats with reversed within-date method order. All share original fast controller/DC/safeQP; meta policy cold load and every inference counted in control time. No fitting/install/heavy audit during replays.',
       'scope':'Seven already exposed dates; no new independent confirmation. Both selected baseline seeds reported separately, existing FRAME/global/full policies unchanged. No online fitting, parameter choice, date replacement or extra baselines.',
       'quality_threshold':.005,'material_cost_threshold':.001,
       'primary':'Survival and full-week cost first, then actual query/control time. No earlier failure relative to same-batch FULL and complete-week cost <=FULL*1.005. Incomplete weeks never compared as full-week costs.',
       'decision_rule':'If FRAME fails quality report FRAME_QUALITY_FAILED. If both META seeds pass every FULL quality screen and each has <=102% FRAME mean eligible-date queries AND <=102% FRAME mean matched-horizon controller time, report META_COMPETITIVE. If FRAME passes all and each META seed either fails quality, or FRAME saves >=5% queries plus nonnegative control time with no >=.1% FULL cost harm on complete pairs, report FRAME_ADVANTAGE_OVER_ADAPTED_STOPPING_IN_MATRIX. Otherwise MIXED. If calibration quality did not match, any apparent FRAME advantage is qualified as a limited adapted-baseline result, never SOTA dominance.',
       'statistics':'Average two timing repeats within date, then compute per-date ratios; separate no-search dates; report P95/P99 and all seed/date outcomes. Repetitions are not independent samples.',
       'permission':'At inference only current legal public metadata and already dispatched prefix feedback; no full unqueried outcomes or trajectory cost supplied. Training/calibration excluded all evaluation date families.',
       'caps':{'episode_s':1200,'wall_s':14400,'physical_steps':180000,'public_forecasts':1800000,'new_fits':0}}
    OUT.mkdir();(OUT/'design.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
    freeze=read(ROOT/'outputs/grid64/code_freeze.json')
    for p,h in freeze.items():assert sha(ROOT/p)==h,p
    for p in list(HERE.glob('*.py'))+[ROOT/'work/grid64/preflight_learning.py',ROOT/'work/grid64/qualify.py',ROOT/'outputs/grid64/learning_preflight.json',ROOT/'outputs/grid64/learning_qualification.json',ROOT/'outputs/grid64/training_finished.json',OUT/'design.json']+[ROOT/m['meta_model'] for m in methods if 'meta_model' in m]:
      if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
      freeze[p.relative_to(ROOT).as_posix()]=sha(p)
    (OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    (OUT/'preflight.json').write_text(json.dumps({'passed':True,'runs':70,'baseline_seeds':2,'calibration_quality_matched':t['calibration_quality_matched'],'new_online_fits':0},indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('FROZEN:70serial closed-loop runs, no concurrent fitting.\n',encoding='utf-8');print(json.dumps({'frozen':True,'runs':70,'selected':costindex}))
if __name__=='__main__':main()
