"""Freeze one balanced timing experiment, all previous models unchanged."""
from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid79'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();assert read(ROOT/'outputs/grid78/review.json')['passed_saved_data']
    D=read(ROOT/'outputs/grid78/design.json');weeks=D['weeks'];base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
    methods=[dict(base,method=p,policy=p) for p in ['FULL','CONTEXT_GATE']]+D['methods'];order=[]
    for repeat in [0,1]:
        for w in weeks if repeat==0 else weeks[::-1]:
            j=weeks.index(w);block=methods[j%4:]+methods[:j%4]
            if repeat==1:block=block[::-1]
            order.extend(dict(m,week=w,repeat=repeat,stage_phase='all') for m in block)
    D.update(stage='GRID79_BALANCED_CONTROLLER_TIMING',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        methods=methods,execution_order=order,repeats=2,
        scope='Same seven exposed dates, same four frozen methods. Two fresh workers/method/date, reversed method and date order on repeat1. No training, tuning, new dates or seed selection.',
        timing='Primary sum of full controller decision_s, including feature construction, first model load/inference and all in-controller optimizers/forecasts. Separate loop wall/setup/bookkeeping/physical time; no mixing query savings and time savings.',
        decision='Quality unchanged: no earlier termination than FULL, complete-week cost<=FULL+0.5%; all existing failures retained. Primary speed metric: average repeats within each date, then macro-average seven relative controller-time reductions. Report pooled-seconds separately. Screen stable >=5% advantage vs FULL and each META seed, >=4/7 dates positive and positive macro in each repeat. Otherwise limited or mixed timing evidence, no retuning/repeats added.',
        latency='Per worker report all-decision P50/P95/P99/max and N1-decision P95; aggregate repeated-date metrics and paired distributions without treating time repeats as independent dates.',
        accounting='Disjoint in-controller optimizer wall, all forecast wall, N1 nonforecast source wall, remainder. Raw sum must reconstruct controller wall; queries and public metadata caches shared, not unique learned contributions.',
        caps=dict(episode_s=1200,wall_s=10800,physical_steps=130000,public_forecasts=2000000,new_fits=0))
    OUT.mkdir();save(OUT/'design.json',D)
    frozen=read(ROOT/'outputs/grid78/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+[OUT/'design.json']:frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen)
    (OUT/'RUN_STATE.md').write_text('FROZEN:56 balanced timing runs, four unchanged methods, seven exposed dates, two repeats.\n',encoding='utf-8')
    print(json.dumps(dict(runs=len(order),methods=4,dates=7,repeats=2,frozen=len(frozen))))
if __name__=='__main__':main()
