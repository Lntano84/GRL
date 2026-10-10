"""Classify archived terminal feedback; no causal attribution or physics runs."""
import json,gzip,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08';records=[]
for version in ['v0','v1','v2']:
    for phase in ['train','evaluate']:
        for summary in (OUT/version/phase).rglob('summary.json'):
            row=json.loads(summary.read_text())
            if row['complete']:continue
            steps=[json.loads(x) for x in gzip.open(summary.parent/'steps.jsonl.gz','rt',encoding='utf-8')]
            last=steps[-1];text=' '.join(last['exceptions'])
            category='redispatch_feasibility' if 'ImpossibleRedispatching' in text else ('disconnected_grid_or_powerflow' if 'non connected grid' in text else 'other')
            records.append({'version':version,'phase':phase,'scenario':row['scenario'],'policy':row['policy'],
                'episode':summary.parent.name,'steps':row['steps'],'category':category,
                'terminal_exceptions':last['exceptions'],'terminal_choice':last['choice'],
                'terminal_offered':last['proposal']['offered'],'terminal_proposal_reason':last['proposal']['reason']})
report={'scope':'Descriptive classification of actual terminal feedback in completed archived runs. Repeated episodes are not independent cases. Terminal no-choice does not exclude effects of earlier decisions.',
        'records':records,'by_category':dict(collections.Counter(x['category'] for x in records))}
(OUT/'failure_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'counts':report['by_category'],'records':len(records)},indent=2))
