"""Freeze GRID04 before running the main full-week comparison."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid04'
OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'design_freeze.json').exists()
split=json.loads((ROOT/'outputs/grid03/scenario_split_frozen.json').read_text(encoding='utf-8'))
selected=[]
for month in [1,4,7,10]:
    names=[n for n in split['extracted_subsets']['development'] if int(n[5:7])==month]
    selected.append(min(names,key=lambda n:hashlib.sha256(('GRID04-seasons-v1:'+n).encode()).hexdigest()))
policies=['FULL','NN20','NN352','LOCAL20','RANDOM20']
plan=[]
for index,scenario in enumerate(selected):
    # Rotate a frozen execution order to avoid always warming one policy first.
    order=policies[index:]+policies[:index]
    plan.extend({'policy':p,'scenario':scenario} for p in order)
design={'stage':'GRID04','environment_seed':0,'shortlist_hash_seed':'GRID04-cheap-v1',
    'data_root':'work/grid03/formal_env_initialized','selected_development_weeks':selected,
    'selection_rule':'Months 1,4,7,10; smallest SHA256(GRID04-seasons-v1:name) within frozen development subset; no outcomes.',
    'plan':plan,'policies':policies,'top_k':20,'same_pool_policies':policies[1:],
    'full_pool_size':421,'shared_pool_size':352,'score_version':'Installed Grid2Op 1.12.5 unnormalized components; no normalized composite.',
    'metrics_order':['native episode completion and survived steps','raw operational cost only for jointly complete episodes','physical losses/redispatch/curtailment/storage; terminal renewable/assistant values','act and end-to-end time; simulation calls; continuous solver status'],
    'local_rule':'For each action, find affected substations. Descending max incident observed rho, then descending sum(max(rho-0.9,0)^2) on incident lines; break ties by SHA256(seed:row). No extra simulation, forecast, cooldown prefilter or future data.',
    'random_rule':'Uniform hash permutation of all 352 rows using seed:current_step:row; first 20. One seed only, no significance claim.',
    'caps':{'physical_steps':40400,'main_runs':20,'per_run_after_import_s':900,'total_runner_s':3600,'preflight_physical_steps':0,'output_bytes':3*1024**3},
    'scope':'Development baseline screen. Published model training IDs unknown. No training/tuning, no validation/test execution; no competition score certification.',
    'decision_rules':[
        'If NN20 and NN352 have identical actions and states throughout all four weeks, stop treating top20 misses within this pool as an evidenced training target.',
        'An existing NN advantage over cheap shortlists requires no extra premature failures, all four paired episodes complete, mean operational reduction >=2% against each cheap arm, and >=3/4 weeks positive against each. This confirms an existing baseline, not a new contribution.',
        'If a cheap arm has no extra failures and complete-episode mean cost <=1.02 times NN20, accept that cheap comparator as a serious baseline; this is a screen, not equivalence.',
        'Incomplete matrix, mixed survival, unqualified metering or directions outside the above are inconclusive. No outcome-selected extra scenes/seeds/budget or automatic training.'
    ]}
freeze=OUT/'design_freeze.json'
freeze.write_text(json.dumps(design,indent=2),encoding='utf-8')
(OUT/'design_freeze.sha256').write_text(hashlib.sha256(freeze.read_bytes()).hexdigest(),encoding='utf-8')
reused=json.loads((ROOT/'outputs/grid03/reused_assets_frozen.json').read_text())
for name in ['work/grid01/trace_agent.py','work/grid03/score_adapter.py']:
    reused[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(OUT/'reused_assets_frozen.json').write_text(json.dumps(reused,indent=2),encoding='utf-8')
(OUT/'GRID04_protocol.md').write_text('''# GRID04 — formal development-week baselines

Frozen before formal main trajectories. See design_freeze.json for exact scenarios, ordering, metrics and resource limits.

Question: do released neural shortlists retain the same-pool exhaustive policy's behavior on full weeks, and do simple legal shortlists attain similar control quality? FULL uses a different 421-item asset and is a separate mature reference; differences versus NN cannot be attributed solely to ranking. NN20, NN352, LOCAL20, RANDOM20 share the 352-item asset, LJNAgentTopoNN, N-1 search, recovery and continuous optimizer. Cheap rules replace only _get_tested_action. All return original action objects in original row semantics; no cold-start or budget handicap for FULL.

No artificial per-decision deployment deadline. Four complete native weeks (2017 steps if unchanged), same environment seed 0, native outages/attacks/rules unchanged. Each run is independent. Validation/test remain sealed. One random shortlist seed is exploratory, not a population estimate. No learning, fitting, new dependency, candidate expansion, Top-k tuning, or normalized competition-score claim.

Scoring is passive. Record official raw operational cost plus independently reconstructed public-observation costs. The installed curtailment fee is the change in total curtailment; also retain physical curtailed energy. An error-terminal fallback from the reward is not a verified physical cost and is explicitly separated. Compare whole-episode cost only when both agents complete; never call a blackout's small sum a gain. Record native terminal renewable/assistant components, not an uncertified weighted composite.

Reuse source telemetry and fresh action field vectorization, with compressed per-step vectors and append-only logs. Timing includes instrumentation and is not production latency. Record construction/import/end-to-end time separately. Preflight has zero physical steps. Main physical ceiling 40,400; 20 runs; 900 s after-import per run; 3,600 s total runner, serial single-thread execution; output ceiling 3 GiB. Stop and preserve partial results on failure; no automatic rerun or substitution. Source/code snapshots precede execution. Matrix results are a development screen and cannot demonstrate a new method or general learning advantage.
''',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('# GRID04\n\nDESIGN FROZEN. No main trajectories yet. No training/automation.\n',encoding='utf-8')
print(json.dumps({'scenarios':selected,'runs':len(plan),'physical_ceiling':design['caps']['physical_steps']},indent=2))
