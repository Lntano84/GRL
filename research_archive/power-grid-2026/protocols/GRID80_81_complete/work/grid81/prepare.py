"""Freeze priorities 1+2 together: fresh scenarios and calibrated random-gate ablation."""
from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid81'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();D=read(ROOT/'outputs/grid79/design.json');g=read(ROOT/'outputs/grid80/design.json')
    cal=[r for r in read(ROOT/'outputs/grid75/models/plain_review.json')['rows'] if r['split']=='calibration']
    assert len(cal)==14 and sum(not r['outside_support'] for r in cal)==13 and sum(not r['defer'] for r in cal)==4
    # Bootstrap scenario id8 means the official generator starts new scenarios at id9.
    weeks=[d+'_9' for d in g['dates']]
    base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
    methods=[dict(base,method=p,policy=p) for p in ['FULL','CONTEXT_GATE','FRAME_GUARD','RANDOM_GATE0','RANDOM_GATE1']]
    order=[]
    for repeat in [0,1]:
        for w in weeks if repeat==0 else weeks[::-1]:
            j=weeks.index(w);block=methods[j%5:]+methods[:j%5]
            if repeat:block=block[::-1]
            order.extend(dict(m,week=w,repeat=repeat,stage_phase='all') for m in block)
    D.update(stage='GRID81_FRESH_GATE_CONFIRMATION',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),weeks=weeks,
        methods=methods,execution_order=order,repeats=2,pilot=[],
        environment='work/grid80/fresh_env',data_generation_design_sha256=hashlib.sha256((ROOT/'outputs/grid80/design.json').read_bytes()).hexdigest(),
        scope='Six prospectively fixed 2036 generated weeks on same public grid, all five arms, two reversed-order timing repeats. No new training, tuning, date/seed selection or new ranking. Separate from previously exposed 2035 dates.',
        random_gate=dict(safe_calibration=14,in_support_calibration=13,context_skips=4,probability=4/13,seeds=[0,1],ood='always defer, exactly same support as frozen gate',coin='SHA256(seed, first24 float32 visible prefix/context features); no suffix, labels or test-derived rate',matching='Expected calibration skip rate, not matched realized test query totals; report realized exposure separately.'),
        decision='Quality threshold unchanged: no earlier termination than FULL and complete-week cost <= FULL+0.5%. New-data support requires all6 quality, >=50% macro query reduction on equal horizons with active FULL search, >=5% macro full-controller reduction with >=4/6 dates positive and both timing-repeat macros positive. Otherwise limited/mixed evidence; no tuning or extra scenarios.',
        ablation='Compare CTX vs always verify: quality first, saved verification queries and full-controller wall separately. Vs EACH random seed: stronger targeting supported only if CTX passes all quality and random fails >=1 date, or both pass and CTX reduces macro queries >=5% with >=4/6 searching dates positive; otherwise report no established targeting advantage. Two random seeds are not independent scenarios, failures never dropped.',
        caps=dict(episode_s=1200,wall_s=14400,physical_steps=150000,public_forecasts=2500000,new_fits=0))
    OUT.mkdir();save(OUT/'design.json',D)
    frozen=read(ROOT/'outputs/grid79/code_freeze.json')
    for p,h in frozen.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'outputs/grid80/design.json',OUT/'design.json']:
        frozen[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(OUT/'code_freeze.json',frozen)
    (OUT/'RUN_STATE.md').write_text('FROZEN:60 planned fresh-scenario runs, pending official generation/qualification and paid-feedback preflight.\n',encoding='utf-8')
    print(json.dumps(dict(runs=len(order),dates=weeks,random_skip=4/13)))
if __name__=='__main__':main()
