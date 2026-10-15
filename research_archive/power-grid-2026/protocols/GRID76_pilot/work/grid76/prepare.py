from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid76'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir()
    qualification=read(ROOT/'outputs/grid76_preflight/preflight.json');assert qualification['passed']
    save(OUT/'preflight.json',qualification)
    fit=read(ROOT/'outputs/grid75/training_finished.json');assert not fit['structural_qualified']
    assert next(r for r in fit['models'] if r['name']=='plain')['qualified']
    D=read(ROOT/'outputs/grid70/design.json');D.update(stage='GRID76_QUALIFIED_CONTEXT_BASELINE',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Qualified ordinary operating-context cheap baseline after structural candidates failed. Same13 exposed-date regression, not structural innovation success or independent confirmation. Current FRAME_GUARD preserved.',
        repair='Initial safe128: fixed GRID75_plain classifier with actual public graph context can return early. Otherwise delegate exact GRID70 with prefix cached once. Unsafe prefixes always old guard fallback.',
        timing='One fresh serial worker/date; graph-context construction and first JSON load/prediction counted in choose/controller. No matched fresh FULL timing claim.',
        decision='Stress pilot July9/December10/March5 must all meet original relative-FULL quality. Saving not required on stress pilot: useful gating may defer these. If quality passes, finish remaining10 same dates. Main competitiveness only after all13: compare same-seven query totals with current10305/11201 benchmarks; only if genuinely competitive, prepare stronger same-information/action conference adaptation. No DQN retraining until then.',
        method_role='Cheap context baseline, not a selected structural model or claimed new algorithm. GRID75 structural main failed and is not run.',
        methods=[dict(method='CONTEXT_GATE',policy='CONTEXT_GATE',rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)],
        model='outputs/grid75/models/plain.json',caps=dict(episode_s=1200,wall_s=10800,physical_steps=30000,public_forecasts=800000,new_fits=0))
    base=D['methods'][0]
    D['execution_order']=[dict(base,week=w,repeat=0,stage_phase='pilot' if w in D['pilot'] else 'remaining') for w in D['pilot']+[w for w in D['weeks'] if w not in D['pilot']]]
    save(OUT/'design.json',D)
    frozen=read(ROOT/'outputs/grid75/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list((ROOT/'work/grid76').glob('*.py'))+list((ROOT/'work/grid75_review').glob('*.py'))+[ROOT/D['model'],OUT/'design.json']:
        frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen)
    (OUT/'RUN_STATE.md').write_text('FROZEN:qualified cheap-context baseline pilot; no structural success claim.\n',encoding='utf-8')
    print(json.dumps(dict(runs=13,pilot=3,frozen=len(frozen))))
if __name__=='__main__':main()
