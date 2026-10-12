"""Freeze first eight-family confirmation, before extracting any reserved data."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid44'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reservation=ROOT/'outputs/grid_final_holdout/reserved_date_families.json'
assert sha(reservation)==(reservation.parent/'reservation.sha256').read_text().strip()
reserved=read(reservation);assert reserved['status']=='RESERVED_NOT_EXTRACTED_OR_EVALUATED'
families=reserved['date_families'];weeks=[r['primary_episode'] for r in families];assert len(weeks)==8
exposed=set(read(ROOT/'outputs/grid_research_audit/split_exposure.json')['exposed_dates'])
assert all(r['date'] not in exposed for r in families)
train=read(ROOT/'outputs/grid21/design.json')['training_weeks']
assert not set(w[:10] for w in weeks)&set(w[:10] for w in train)
assert read(ROOT/'outputs/grid41/review.json')['passed_saved_data']
assert not read(ROOT/'outputs/grid43/pilot_audit.json')['all_rankers_qualified']
methods=[dict(m) for m in read(ROOT/'outputs/grid41/design.json')['methods']]
methods.append(next(dict(m) for m in read(ROOT/'outputs/grid34/design.json')['methods'] if m['method']=='LOCAL_PPO'))
order=[]
for repeat in [0,1]:
    for n,week in enumerate(weeks):
        offset=(n+3*repeat)%len(methods)
        sequence=methods[offset:]+methods[:offset]
        if repeat:sequence=list(reversed(sequence))
        for m in sequence:order.append({**m,'week':week,'repeat':repeat,'stage_phase':'pilot'})
D={'stage':'GRID44_RESERVED_DATE_CONFIRMATION_V1','reservation_sha256':sha(reservation),
   'weeks':weeks,'families':families,'methods':methods,'execution_order':order,'repeats':2,'serial_workers':1,
   'choice_before_access':'Original GRID21 GNN0/1 and GRID28 local-completion control remain active. GRID38 target-mixture did not establish improvement; GRID43 joint tie completion added original-model queries without changing March cost, and is not adopted. SoftGNN uses same original common stopping rule for ranking-only comparison; its bounded GRID40B adaptation and all losses disclosed.',
   'scope':'8previously metadata-only reserved date families,one preselected primary variant each. Own2seeds,one recent SoftGNN adaptation2seeds,authorPPO,matureFULL =3baseline families. No new model,hyperparameter,scenario selection or training. External PPO exposure unknown.',
   'timing':'96fresh serial single-thread workers; rotate and reverse6arm order; identical original controller/DC/safeQP caches/legal masks/canonical actions/stop. Include feature/prior/model/search/guard/optimization in control wallclock; setup,physicalstep,export separate. Two timing repeats are not independent date families.',
   'primary':'Report completion and steps for every family/seed. Require no earlier failure than FULL. Operating cost primary only on FULL-completed weeks when our method also completes; average normalized cost within family,not total across unequal survival. Speed primary on equal-horizon pairs only: mean paired control-time reduction over date families,repeat averaged,own seeds both reported and equally averaged.',
   'secondary':'Pooled equal-horizon control time; topology source time/queries; per-family p95 decision latency; reported training/teacher costs and graph attribution limits. No causal allocation of shared cache savings to GNN.',
   'investment_screen':'For both own seeds: no earlier failure on all8families; comparable complete-week cost<=0.5%ofFULL per family; mean equal-horizon control reduction>=5%,positive in>=ceil(0.75*n_comparable)families. This is practical screening,not equivalence or acceptance guarantee. Other/mixed outcomes remain visible; no test-based model retuning or family replacement.',
   'statistics':'Fixed10000 bootstrap resamples of date-family summaries withseed20261011;average repeats/seeds within family. Report interval as small-sample empirical uncertainty,not guaranteed coverage. Never use96runs as96independent grids.',
   'sealed_use':'All8families and all their variants remain excluded from subsequent training/tuning. If a new model is later made,this dataset is exposed confirmation and must not be described as independent again.',
   'baseline_note':'SoftGNN official GAT/converter unchanged but IDF2023/909actions/basecontext/148states/64epochs is a resource-matched adaptation,not original WCCI2022 full-paper reproduction. Recent-method success/failure cannot refute published results.',
   'caps':{'episode_s':1500,'wall_s':18000,'physical_steps':200000,'public_forecasts':2000000,'new_fits':0,'reserved_evaluations':96}}
worker=(ROOT/'work/grid28/run.py').read_text(encoding='utf-8')
changes={
 '"""GRID28 single-repeat closed-loop quality and measured-cost development pilot.\nDerived from frozen GRID26 runner; all historical files unchanged.\n"""':'"""GRID44 first reserved-family worker; no reference trajectory enters controls."""',
 'OUT = ROOT / "outputs/grid28"':'OUT = ROOT / "outputs/grid44"',
 '    historical = ROOT / "outputs/grid23/runs" / (week + "__FULL")\n    with gzip.open(historical / "steps.jsonl.gz", "rt", encoding="utf-8") as f:\n        reference = [json.loads(line) for line in f]':'    reference = None  # Compare delivered trajectories only after all arms finish.',
 'row["full_reference_match"] = step <= len(reference) and all(row[k] == reference[step-1][k] for k in comparison_keys)':'row["full_reference_match"] = None',
 'if item["policy"] == "PROTECTED":':'if reference is not None and item["policy"] == "PROTECTED":',
 '"exact_full_trajectory": len(rows) == len(reference) and all(r["full_reference_match"] for r in rows)':'"exact_full_trajectory": None',
 '"first_full_divergence": next((r["step"] for r in rows if not r["full_reference_match"]), None)':'"first_full_divergence": None'}
for a,b in changes.items():
    count=worker.count(a);assert count==(2 if a=='if item["policy"] == "PROTECTED":' else 1),(a,count)
    worker=worker.replace(a,b)
(HERE/'worker.py').write_text(worker,encoding='utf-8')
run=(ROOT/'work/grid41/run.py').read_text(encoding='utf-8')
changes={'"""GRID41 matched official-core SoftGNN and original GNN development runs."""':'"""GRID44 first reserved-family confirmation;serial checkpointed workers."""',
         "OUT=ROOT/'outputs/grid41'":"OUT=ROOT/'outputs/grid44'",
         "ROOT/'work/grid28/run.py'":"ROOT/'work/grid44/worker.py'",
         "if item['rule']=='FULL':assert summary['exact_full_trajectory']":"assert summary['exact_full_trajectory'] is None",
         "'new_fits':0,'reserved_evaluations':0":"'new_fits':0,'reserved_evaluations':len(summaries)",
         'all5arms shareDC+safe cache':'all6arms shareDC+safe cache;first reserved confirmation'}
for a,b in changes.items():assert run.count(a)==1,(a,run.count(a));run=run.replace(a,b)
(HERE/'run.py').write_text(run,encoding='utf-8')
OUT.mkdir(exist_ok=False)
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),HERE/'worker.py',HERE/'run.py',HERE/'extract.py',HERE/'audit.py',HERE/'preflight.py',reservation,ROOT/'outputs/grid43/pilot_audit.json']
files += [ROOT/p for p in read(ROOT/'outputs/grid41/code_freeze.json')]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in dict.fromkeys(files)},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN BEFORE EXTRACTION:96runs/8reserved families,3baseline families,all2model seeds kept. No new fits.\n',encoding='utf-8')
(ROOT/'outputs/grid43/closed_after_pilot.json').write_text(json.dumps({'status':'CLOSED_AFTER_6_PILOTS','remaining18_started':False,
    'reason':'Frozen joint quality gate not met; original-model cost unchanged while queries increased. Preserve as bounded common-policy diagnostic; active original guard unchanged.'},indent=2),encoding='utf-8')
(ROOT/'outputs/grid43/RUN_STATE.md').write_text('CLOSED AFTER6PILOTS:original models remain active;unrun18not claimed completed. No pending job.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':96,'families':8,'new_fits':0}))
