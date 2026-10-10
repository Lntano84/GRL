"""Combine audited V0 baselines and preserved model versions without test selection."""
from pathlib import Path
import json,hashlib,gzip
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
D=json.loads((OUT/'design.json').read_text());records=[];finished=[]
for version in ['v0','v1','v2']:
    for phase in ['train','evaluate']:
        for kind in ['gnn','mlp']:
            assert (OUT/version/phase/kind/'finished.json').exists(),f'Unfinished phase: {version}/{phase}/{kind}'
assert (OUT/'v0/evaluate/baselines/finished.json').exists(),'Unfinished baseline phase'
for version in ['v0','v1','v2']:
    audit=OUT/version/'audit.json'
    if not audit.exists():continue
    assert json.loads(audit.read_text())['passed']
    for file in sorted((OUT/version/'evaluate').rglob('summary.json')):
        row=json.loads(file.read_text());row['version']=version;records.append(row)
    for file in (OUT/version).rglob('finished.json'):
        finished.append({'path':str(file.relative_to(ROOT)),**json.loads(file.read_text())})
baseline={(r['scenario'],r['policy']):r for r in records if r['version']=='v0' and r['policy'] in D['baselines']}
comparisons=[]
for version in ['v0','v1','v2']:
    for kind in ['gnn','mlp']:
        models={r['scenario']:r for r in records if r['version']==version and r['policy']==kind}
        for name in D['baselines']:
            pairs=[]
            for week in D['evaluation_weeks']:
                if week not in models or (week,name) not in baseline:continue
                a,b=models[week],baseline[week,name]
                same_actions=False;same_observations=False
                pa,pb=ROOT/a['path'],ROOT/b['path']
                with gzip.open(pa/'steps.jsonl.gz','rt',encoding='utf-8') as f:ar=[json.loads(l) for l in f]
                with gzip.open(pb/'steps.jsonl.gz','rt',encoding='utf-8') as f:br=[json.loads(l) for l in f]
                same_actions=len(ar)==len(br) and all(x['action_hash']==y['action_hash'] for x,y in zip(ar,br))
                same_observations=len(ar)==len(br) and all(x['after_hash']==y['after_hash'] for x,y in zip(ar,br))
                pairs.append({'week':week,'model_complete':a['complete'],'baseline_complete':b['complete'],
                    'model_steps':a['steps'],'baseline_steps':b['steps'],
                    'relative_cost_reduction':(b['raw_cost']-a['raw_cost'])/abs(b['raw_cost']) if a['complete'] and b['complete'] else None,
                    'identical_actions':same_actions,'identical_observations':same_observations})
            gains=[p['relative_cost_reduction'] for p in pairs if p['relative_cost_reduction'] is not None]
            comparisons.append({'version':version,'kind':kind,'baseline':name,'pairs':pairs,
                'jointly_complete':len(gains),'mean_relative_cost_reduction':float(np.mean(gains)) if gains else None,
                'positive_pairs':sum(g>1e-9 for g in gains),'negative_pairs':sum(g < -1e-9 for g in gains),
                'identical_trajectory_pairs':sum(p['identical_actions'] and p['identical_observations'] for p in pairs)})
failure_steps=[]
for folder in [OUT/'v0_LOGGING_FAILED',OUT/'v1_MASK_FAILED']:
    if folder.exists():
        for file in folder.rglob('failure.json'):
            r=json.loads(file.read_text());failure_steps.append({'path':str(file.relative_to(ROOT)),'physical_steps':r['physical_steps']})
preflight=json.loads((OUT/'preflight.json').read_text())
resource={'successful_physical_steps':sum(r['physical_steps'] for r in finished),
    'failed_attempt_physical_steps':sum(r['physical_steps'] for r in failure_steps),
    'failed_attempts':failure_steps,'preflight_scope':preflight,
    'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
    'note':'Preflight physical steps are reported in its retained qualification record; counts above are run phases only. Simulated forecast power-flow calls are additional and logged per episode. Training/evaluation phase wall times include concurrent contention; they are not intrinsic speed comparisons.'}
known_preflight_steps=384
diagnostic_files=[OUT/'ramp_diagnostic.json',OUT/'ramp_diagnostic_FAILED_API.json']
resource['diagnostic_physical_steps']=sum(json.loads(p.read_text())['physical_steps'] for p in diagnostic_files if p.exists())
guard_finish=OUT/'guard_qualification/finished.json'
assert guard_finish.exists(),'Training-only guard qualification is unfinished'
assert json.loads((OUT/'guard_qualification/audit.json').read_text())['passed']
resource['guard_qualification_physical_steps']=json.loads(guard_finish.read_text())['physical_steps']
resume=json.loads((OUT/'resume_audit.json').read_text())
assert resume['passed']
resource['interrupted_extra_physical_steps_lower']=resume['extra_physical_steps_lower']
resource['interrupted_extra_physical_steps_upper']=resume['extra_physical_steps_upper']
resource['cumulative_physical_steps_including_qualification']=resource['successful_physical_steps']+resource['failed_attempt_physical_steps']+known_preflight_steps+resource['diagnostic_physical_steps']+resource['guard_qualification_physical_steps']+resume['extra_physical_steps_upper']
assert resource['cumulative_physical_steps_including_qualification']<=220000
assert resource['output_bytes']<=1073741824
original=json.loads((OUT/'original_assets.json').read_text())
for p,h in original.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
report={'records':records,'comparisons':comparisons,'resources':resource,'original_assets_unchanged':True,
    'final_test_used':False,'limits':['Single training seed, four validation weeks; no population or publication guarantee.',
        'Author pretraining identities are unknown. Validation is disjoint from our PPO weeks, not proven disjoint from all author training.',
        'NN20 is instrumented with common proposal generation; wall time is not pure original-agent deployment latency.',
        'Independent audit checks saved cost vectors and accounting, not independent electrical replays or retraining.',
        'One-step proposal screens cannot guarantee later survival.',
        'V1 and V2 are disclosed development adaptations, not independent confirmations.']}
report['limits'].append('Three processes were externally interrupted without a failure marker. Whole-episode checkpoint resumption retained model/Adam/Torch RNG; interrupted costs are counted with a conservative three-step uncertainty bound. Prior process and resumed wall times are not a single uninterrupted cap.')
(OUT/'delivery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# GRID08 training and adaptation delivery','',
    'Auxiliary binary GNN/MLP PPO chooses safe-continuous restoration over an immutable NN20 agent. This is actual training and complete-week replay, not a claim that GNN + RL itself is novel.','',
    '| Version | Model | Reference | Complete pairs | Mean cost reduction | Positive / negative | Identical trajectories |',
    '|---|---|---|---:|---:|---:|---:|']
for c in comparisons:
    g='NA' if c['mean_relative_cost_reduction'] is None else f"{100*c['mean_relative_cost_reduction']:.4f}%"
    lines.append(f"| {c['version']} | {c['kind']} | {c['baseline']} | {c['jointly_complete']} | {g} | {c['positive_pairs']} / {c['negative_pairs']} | {c['identical_trajectory_pairs']} |")
lines+=['','## Interpretation','',
    'An improvement over NN20 may come from supplying the restoration action. Improvement over ALWAYS_RESTORE and IMMEDIATE_COST is the relevant evidence for added decision value from learning. Exact trajectory equality identifies a reproduced fixed rule; it is stronger than a nonsignificant comparison. Lower cumulative cost after earlier failure is never counted as quality gain.','',
    'V0 retains the original shared actor/critic and all rollout-step policy normalization. V1 isolates actor and critic, normalizes and optimizes the actor only on offered decisions, and makes critic bootstrap proposal-invariant. V2 shifts the V1 policy near the qualified fixed restoration rule and performs four additional training passes at half the learning rate, preserving critic and input-dependent logit differences. These are development adaptations; all checkpoints and failed engineering attempts are retained.','',
    '## Resources','',json.dumps({k:v for k,v in resource.items() if k not in ['preflight_scope','failed_attempts']},indent=2),'',
    '## Limits','']
lines+=['- '+x for x in report['limits']]
(OUT/'GRID08_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'resources':{k:v for k,v in resource.items() if k not in ['preflight_scope','failed_attempts']}},indent=2))
