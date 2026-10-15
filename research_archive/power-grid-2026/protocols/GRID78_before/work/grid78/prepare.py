"""Freeze seven already-exposed dates, two seeds; no further model selection."""
from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid78'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not OUT.exists();assert read(ROOT/'outputs/grid77/fit_audit.json')['passed']
    selected=read(ROOT/'outputs/grid77/selected.json');D=read(ROOT/'outputs/grid76/design.json');weeks=read(ROOT/'outputs/grid65/design.json')['weeks']
    methods=[]
    for seed in [0,1]:
        model=f"outputs/grid77/models/cost{selected['costindex']}_seed{seed}/final.pt"
        methods.append(dict(method=f'META_CONTEXT{seed}',policy=f'META_CONTEXT{seed}',rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,meta_model=model,meta_model_sha256=sha(ROOT/model)))
    D.update(stage='GRID78_SAME_CONTEXT_MODE_CONFERENCE_COMPARISON',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        weeks=weeks,pilot=[],methods=methods,execution_order=[dict(m,week=w,repeat=0,stage_phase='all') for j,w in enumerate(weeks) for m in (methods if j%2==0 else methods[::-1])],
        decision='Retain both model seeds and every date regardless of results. Compare quality first: no earlier termination than FULL; complete-week cost<=FULL+0.5%. Report per-seed query totals and per-date metrics against frozen GRID76 context gate and GRID70 guard. No date/seed selection and no automatic retuning. Lack of calibration matching must be disclosed.',
        scope='Seven exposed dates, one new run/model/date. Same24 legal context plus baseline-only mode bit and exact guard macro. Common control/caches. Development, not independent confirmation or full original conference task reproduction.',
        timing='Only descriptive absolute controller/latency accounting. Old comparison runs are not matched fresh timing pairs; no end-to-end superiority claim.',
        calibration_quality_matched=selected['calibration_quality_matched'],selected_cost=selected['costindex'],
        caps=dict(episode_s=1200,wall_s=10800,physical_steps=60000,public_forecasts=800000,new_fits=0))
    for k in ['repair','method_role','model']:D.pop(k,None)
    OUT.mkdir();save(OUT/'design.json',D)
    frozen=read(ROOT/'outputs/grid77/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+[OUT/'design.json',ROOT/'outputs/grid77/selected.json']+[ROOT/m['meta_model'] for m in methods]:frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen);save(OUT/'preflight.json',dict(passed=True,source_fit_audit_sha256=sha(ROOT/'outputs/grid77/fit_audit.json'),runs=14,models=2,new_fits=0))
    (OUT/'RUN_STATE.md').write_text('FROZEN:14 finite exposed-date runs, same-context/mode baseline.\n',encoding='utf-8');print(json.dumps(dict(runs=14,frozen=len(frozen))))
if __name__=='__main__':main()
