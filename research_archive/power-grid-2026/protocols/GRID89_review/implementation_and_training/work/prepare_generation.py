"""Freeze independent-seed data generation before seeing its series or outcomes."""
from pathlib import Path
import datetime as dt,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review';W=Path(__file__).parent
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir();E=W/'fresh_env';E.mkdir();src=ROOT/'work/grid03/formal_env_initialized'
    dates=['2036-01-07','2036-03-03','2036-05-05','2036-07-07','2036-09-01','2036-11-03']
    static={}
    for p in src.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.py'):
            shutil.copy2(p,E/p.name);static[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    params=json.loads((E/'scenario_params.json').read_text());params['all_dates']=dates;save(E/'scenario_params.json',params)
    (E/'chronics').mkdir();shutil.copytree(src/'chronics/2035-01-01_8',E/'chronics/2035-01-01_8')
    D=dict(stage='GRID89_REVIEW_EVIDENCE_AND_FINITE_VALIDATION',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      authorization='2026-10-17 用户：开始吧，把这些全做完。',dates=dates,generation_seed=20261017,scenario_id=10,
      environment='work/grid89_review/fresh_env',source_static_sha256=static,
      generation='Exact prior Chronix2Grid official wrapper; same public grid/calendar dates, independently seeded stochastic series; six predeclared scenarios, no replacement or outcome filtering.',
      order=dict(dates='All six exposed GRID81 dates',orders=['STATIC_ID','RANDOM_11','RANDOM_23','RANDOM_37','LOCAL_STRESS'],arms=['FULL','FRAME_GUARD'],repeats=1,randomization='One seeded global candidate priority per run; filter legal representatives each call. IDs and low-ID tie rule unchanged. Random seeds are order replicates, not independent environments.'),
      parameters=dict(dates=['2035-03-05_0','2035-06-25_8','2035-09-03_4','2035-10-01_5'],default=dict(p=128,h=64,qmin=256,patience=2,safe=.9),single_axis=dict(p=[64,256],h=[32,128],qmin=[128,512],patience=[1,3],safe=[.85,.95]),arms=12,repeats=1,purpose='Quality/query sensitivity; one timing repeat descriptive, no parameter selection; includes FULL and default.'),
      meta=dict(cost_levels=[0.,.0025],seeds=[0,1,2,3,4],new_fits=3,training='Reuse cost .0025 seeds0/1; only fit seeds2/3/4 with the exact existing 30000-episode loop. No new teacher data or calibration selection.',exposed_comparison='six GRID81 dates; FRAME plus zero-cost and positive-cost five seeds, one timing repeat; report all arms, quality/query primary, time descriptive.'),
      local_and_sealed=dict(dates='six new independent-seed series',arms=['FULL','FRAME_GUARD','GLOBAL_MATCH','FIXED_512'],repeats=3,mechanism='Same-state suffix budget derived from public substation/alias metadata; no hidden outcomes to choose budget. Paired FULL feedback and closed-loop outcomes separately.'),
      metrics=['survival and complete weeks','full-week operational cost on jointly completed dates','queries on matched active horizons','full controller wall time','all-step and search-step P50/P95/P99','disjoint compute accounting','same-state reward regret and signed equal-budget differences','per-date grouping and date-cluster bootstrap descriptive intervals'],
      quality_threshold=.005,decision='Preserve all cases. No general quality guarantee. Efficiency requires comparable quality and matched horizons; report macro and per-date data. The existing 5% and majority-direction screen remains an investment screen, not statistical equivalence. Order robustness reported by each predeclared order, no best order selection. Independent seed validation supports only same-grid generated-series transfer.',
      resource=dict(max_total_hours=12,max_new_model_fits=3,no_parallel_timing=True,max_new_replays=246),
      exclusions='No second grid, new ranking model, GNN/RL controller redesign, parameter grid search or removal of adverse cases. No graph/table authoring requested; deliver text/CSV evidence and a new manuscript version preserving V1.')
    save(OUT/'design.json',D)
    save(OUT/'generation_code_freeze.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'prepare_generation.py',W/'generate.py',OUT/'design.json']})
    (OUT/'RUN_STATE.md').write_text('FROZEN: independent generation plan; experiment implementation/preflight pending. No new control outcomes.\n',encoding='utf-8')
    print(json.dumps(dict(seed=D['generation_seed'],dates=dates,max_replays=D['resource']['max_new_replays'])))
if __name__=='__main__':main()
