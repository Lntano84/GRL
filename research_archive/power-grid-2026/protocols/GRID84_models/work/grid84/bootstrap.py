"""Create a finite fixed-configuration seed extension, before any new fits."""
from pathlib import Path
import datetime as dt, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid84';WORK=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists()
    assert read(ROOT/'outputs/grid82/review.json')['passed_saved_data']
    selected=read(ROOT/'outputs/grid77/selected.json');assert selected['costindex']==2 and selected['cost_level']==0
    for old in ['grid77','grid82']:
        for p,h in read(ROOT/f'outputs/{old}/code_freeze.json').items():assert sha(ROOT/p)==h,p
    # Copy the evaluated serial runner/worker adapter; only stage paths differ.
    for name in ['run.py','mechanism_adapter.py']:
        s=(ROOT/f'work/grid82/{name}').read_text(encoding='utf-8').replace('grid82','grid84')
        if name=='run.py':
            s=s.replace("save(OUT/'finished.json',dict(passed_engineering=True,summaries=summaries,new_fits=0))", "save(OUT/'finished.json',dict(passed_engineering=True,summaries=summaries,new_fits=3))")
        (WORK/name).write_text(s,encoding='utf-8')
    # Use the exact old learning loop, restricted to the already selected cost.
    s=(ROOT/'work/grid77/train.py').read_text(encoding='utf-8')
    s=s.replace("from metacontrol import QModel,prefixes,utility,mask_q,replay,DIM", "sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'work/grid77'))\nfrom metacontrol import QModel,prefixes,utility,mask_q,replay,DIM")
    s=s.replace("OUT=ROOT/'outputs/grid77'","OUT=ROOT/'outputs/grid84'")
    s=s.replace("assert read(OUT/'learning_preflight.json')['passed']", "assert read(OUT/'training_preflight.json')['passed']")
    s=s.replace("read(OUT/'code_freeze.json')", "read(OUT/'training_code_freeze.json')")
    s=s.replace("records=read(OUT/'records.json');subs=read(OUT/'data_receipts.json')['subids']", "records=read(ROOT/'outputs/grid77/records.json');subs=read(ROOT/'outputs/grid77/data_receipts.json')['subids']")
    s=s.replace("for costindex,cost in enumerate(D['cost_levels']):", "for costindex,cost in [(2,D['cost_levels'][2])]:")
    marker='    options=[]\n';assert s.count(marker)==1
    s=s.split(marker)[0]+'''    save(OUT/'training_finished.json',dict(passed_engineering=True,fits=3,wall_s=time.perf_counter()-begun,results=results,new_teacher_queries=0,costindex=2,cost_level=0.0,reselected=False))
    (OUT/'RUN_STATE.md').write_text('THREE NEW FIXED-COST FITS COMPLETE; preflight and balanced comparison pending.\\n',encoding='utf-8')
    print(json.dumps(dict(fits=3,wall_s=time.perf_counter()-begun)),flush=True)
if __name__=='__main__':main()
'''
    (WORK/'train.py').write_text(s,encoding='utf-8')
    # Extend saved-data audit to all five seeds; old seed trajectories must match.
    s=(ROOT/'work/grid82/audit.py').read_text(encoding='utf-8').replace('grid82','grid84')
    s=s.replace("assert len(S)==36", "assert len(S)==72")
    s=s.replace("for baseline in ['META_CONTEXT0','META_CONTEXT1']:","for baseline in [f'META_CONTEXT{i}' for i in range(5)]:")
    s=s.replace('runs=36,','runs=72,').replace('new_fits=0,','new_fits=3,')
    s=s.replace("if s['method']=='FRAME_GUARD':", "if s['method'] in ['FRAME_GUARD','META_CONTEXT0','META_CONTEXT1']:")
    s=s.replace("ROOT/'outputs/grid81/runs'/f\"r0__{s['week']}__FRAME_GUARD\"", "ROOT/'outputs/grid82/runs'/f\"r0__{s['week']}__{s['method']}\"")
    s=s.replace('COMPLETE:36 balanced runs;', 'COMPLETE:72 balanced runs;')
    s=s.replace('Same six now-exposed generated scenarios, final fixed-guard vs frozen ICAPS2022-core adaptation; no independent confirmation, original calibration limitation retained; no cross-batch timing claim.', 'Same six exposed generated scenarios, fixed-guard and five same-configuration ICAPS2022-core adaptation seeds; old0/1 retained, only2/3/4 fitted; no independent confirmation, original calibration limitation retained; no cross-batch timing claim.')
    (WORK/'audit.py').write_text(s,encoding='utf-8')
    D=read(ROOT/'outputs/grid82/design.json');T=read(ROOT/'outputs/grid77/design.json')
    base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
    methods=[dict(base,method='FRAME_GUARD',policy='FRAME_GUARD')]
    for seed in range(5):
        path=f'outputs/grid77/models/cost2_seed{seed}/final.pt' if seed<2 else f'outputs/grid84/models/cost2_seed{seed}/final.pt'
        methods.append(dict(base,method=f'META_CONTEXT{seed}',policy=f'META_CONTEXT{seed}',meta_model=path))
    order=[];weeks=D['weeks']
    for rep in [0,1]:
        for w in weeks if rep==0 else weeks[::-1]:
            j=weeks.index(w);block=methods[j%6:]+methods[:j%6]
            if rep:block=block[::-1]
            order.extend(dict(m,week=w,repeat=rep,stage_phase='all') for m in block)
    D.update(stage='GRID84_FIXED_COST_FIVE_SEED_COMPARISON',authorization='2026-10-16 user: 开始吧, after fixed seeds2/3/4 proposal.',
             frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),methods=methods,execution_order=order,
             seeds=[2,3,4],all_reported_seeds=[0,1,2,3,4],cost_levels=T['cost_levels'],episodes=T['episodes'],
             min_explore_steps=T['min_explore_steps'],epsilon_anneal_steps=T['epsilon_anneal_steps'],epsilon_final=T['epsilon_final'],
             lr=T['lr'],gamma=T['gamma'],batch_size=T['batch_size'],replay_capacity=T['replay_capacity'],target_retention=T['target_retention'],
             selected_configuration=selected,calibration='Diagnostics on existing22 only; cost2 fixed from GRID77, no reselection or exclusion based on calibration. Retain failures.',
             training='Exactly three seeds2/3/4, original GRID77 loop and 129/22 records, final checkpoints only; old seed0/1 weights unchanged. Zero new teacher simulation.',
             scope='Five-seed variation of current enhanced grid adaptation on six already exposed generated weeks. No new independent scenario confirmation. All seeds reported.',
             decision='Per-seed GRID82 rule unchanged: both6/6 quality, >=5% macro controller saving, >=4/6 positive dates, both timing-repeat macro savings>0. Overall stronger five-seed support only if guard and all five seeds quality6/6, mean of five macro savings>=5%, and at least3/5 per-seed screens pass. Otherwise report competitiveness/mixed results or quality failures, no seed selection or retuning.',
             conference_provenance=dict(D['conference_provenance'],training='Original0/1 plus new2/3/4 at fixed shared costindex2=0; no new cost selection.'),
             caps=dict(episode_s=1200,wall_s=14400,physical_steps=160000,public_forecasts=4000000,new_fits=3,training_wall_s=1800))
    OUT.mkdir();save(OUT/'design.json',D);shutil.copy2(ROOT/'outputs/grid82/data_freeze.json',OUT/'data_freeze.json')
    frozen=read(ROOT/'outputs/grid82/code_freeze.json')
    for n in ['outputs/grid77/records.json','outputs/grid77/data_receipts.json','outputs/grid77/learning_preflight.json',
              'outputs/grid77/selected.json','work/grid77/train.py']:
        frozen[n]=sha(ROOT/n)
    for p in WORK.glob('*.py'):frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    for n in ['design.json','data_freeze.json']:frozen[f'outputs/grid84/{n}']=sha(OUT/n)
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    save(OUT/'training_code_freeze.json',frozen)
    save(OUT/'training_preflight.json',dict(passed=True,source_loop_unchanged=True,fits=3,seeds=[2,3,4],costindex=2,reselect=False,old_learning_preflight=read(ROOT/'outputs/grid77/learning_preflight.json'),new_teacher_queries=0))
    (OUT/'RUN_STATE.md').write_text('FROZEN:three fixed-cost additional seeds, then72 balanced serial runs; old checkpoints retained.\n',encoding='utf-8')
    print(json.dumps(dict(fits=3,runs=72,methods=len(methods),dates=6),ensure_ascii=False))
if __name__=='__main__':main()
