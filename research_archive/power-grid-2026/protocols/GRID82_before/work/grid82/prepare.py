"""Freeze one final-version conference-method comparison; no tuning or refitting."""
from pathlib import Path
import datetime as dt,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid82'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();assert read(ROOT/'outputs/grid81/review.json')['passed_saved_data']
    D=read(ROOT/'outputs/grid81/design.json');weeks=D['weeks'];base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
    old=read(ROOT/'outputs/grid79/design.json')
    methods=[dict(base,method='FRAME_GUARD',policy='FRAME_GUARD')]+[x for x in old['methods'] if x['method'].startswith('META_CONTEXT')]
    assert len(methods)==3
    order=[]
    for repeat in [0,1]:
        for w in weeks if repeat==0 else weeks[::-1]:
            j=weeks.index(w);block=methods[j%3:]+methods[:j%3]
            if repeat:block=block[::-1]
            order.extend(dict(m,week=w,repeat=repeat,stage_phase='all') for m in block)
    D.update(stage='GRID82_FINAL_FIXED_GUARD_CONFERENCE_COMPARISON',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),methods=methods,execution_order=order,repeats=2,pilot=[],
        scope='Six GRID81-generated scenarios already exposed, one frozen nonlearned search-budget rule and two frozen ICAPS2022-core adaptation models. Same-context same-mode final-version comparison, not a new independent confirmation.',
        conference_provenance=dict(source='Bhatia et al., ICAPS2022; DoubleDQN core task adaptation from Metareasoning.jl',source_commit=read(ROOT/'outputs/grid77/design.json')['source_commit'],selected=read(ROOT/'outputs/grid77/selected.json'),training='GRID77 only; new fits0; both seeds and shared costindex2 retained',qualification_limit='Original calibration_quality_matched=false is preserved. Do not call a quality-failing fast competitor a same-quality speed victory.'),
        full_reference='outputs/grid81/finished.json; quality/cost/horizon only, read in audit after all trajectories. No cross-batch FULL timing comparison.',
        decision='For EACH conference seed independently: if guard passes quality6/6 and competitor also passes6/6, require >=5% macro controller reduction, >=4/6 positive dates, and positive macro in both timing repeats for supported same-quality speed advantage. Guard valid with competitor quality failures => report quality difference plus descriptive timing only, NOT all-date same-quality speed superiority. Both valid but mixed/small => no stable speed advantage; guard failure => quality screen failed. No retuning, extra seeds or deleting dates.',
        ablation='Completed GRID81 gate ablation is retained separately; no new gate, random, SoftGNN or PPO fitting/comparison added.',
        caps=dict(episode_s=1200,wall_s=10800,physical_steps=90000,public_forecasts=2000000,new_fits=0))
    OUT.mkdir();save(OUT/'design.json',D);shutil.copy2(ROOT/'outputs/grid81/data_freeze.json',OUT/'data_freeze.json')
    frozen=read(ROOT/'outputs/grid81/code_freeze.json')
    for p,h in read(ROOT/'outputs/grid79/code_freeze.json').items():
        if p in frozen:assert frozen[p]==h,p
        frozen[p]=h
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+[OUT/'design.json',OUT/'data_freeze.json']:
        frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen)
    (OUT/'RUN_STATE.md').write_text('FROZEN:36 final-version matched comparison runs, no new models or dates; preflight pending.\n',encoding='utf-8')
    print(json.dumps(dict(runs=len(order),methods=[m['method'] for m in methods],dates=6,repeats=2)))
if __name__=='__main__':main()
