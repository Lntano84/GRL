"""Freeze FA03B design and inherited inputs before any new trajectory."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

root=Path(__file__).resolve().parents[2]
out=root/'outputs/fa03b_design'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
relpaths=[
    'outputs/fa03b_design/FA03B_next_step.md',
    'outputs/fa00/FA00_data.npz',
    'outputs/fa00/FA00_protocol.json',
    'outputs/fa00/FA00_preflight.json',
    'outputs/fa02_end/FA02_protocol.json',
    'outputs/fa02_end/FA02_analysis.json',
    'outputs/fa02_end/FA02_audit.json',
    'work/fa00/core.py', 'work/fa00/run.py', 'work/fa01/policy.py',
    'work/fa02_end/run.py',
    'work/fa03_design/source/Algorithm_Selection_Lexical_Ordering.py',
]
for base, arm in [('fa00','FIXED-100'),('fa01','RANDOM-100'),('fa01','CHEAP-100')]:
    for seed in [7,42,99]:
        for name in ['actions.jsonl.gz','history.json']:
            relpaths.append(f'outputs/{base}/runs/{arm}_s{seed}/{name}')
files={p:sha(root/p) for p in relpaths}
analysis=json.loads((root/'outputs/fa02_end/FA02_analysis.json').read_text())
record=dict(stage='FA03B',frozen_utc=datetime.now(timezone.utc).isoformat(),
    purpose='ASP development strong-baseline completion; not FA03 cross-scenario confirmation',
    dataset='ASP-POTASSCO',native_T=600,cap=100,budget_s=86460,
    train=1048,validation=117,test=129,algorithms=11,
    query_pair=566,query_row=11,initial_rows=20,seeds=[7,42,99],
    new_arms=['LEX-PC-U-PD','RANDOM-ROW'],new_trajectories=6,
    inherited_means=analysis['means']['ALL'],terminal_round=0,
    completion_boundary='ok and runtime <= actual_cap',
    continue_rule='ACTIVE >=5% mean improvement vs all four controls and >=2/3 signs each',
    close_rule='any new control mean <= 1.02*ACTIVE mean',
    new_trajectories_started=0,new_models_fitted=0,files=files)
(out/'FA03B_design_freeze.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps({k:record[k] for k in ['stage','new_trajectories','new_trajectories_started','new_models_fitted','query_pair','query_row','inherited_means']},indent=2))

