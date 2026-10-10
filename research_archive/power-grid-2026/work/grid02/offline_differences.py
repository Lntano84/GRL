"""Attribute only what saved artifacts identify. No policy, simulation or hidden future."""
import json
from pathlib import Path
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'
def read(folder,name):
    return [json.loads(x) for x in (folder/name).read_text(encoding='utf-8').splitlines() if x.strip()]
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
with np.load(OUT/'preflight_vectors.npz',allow_pickle=False) as data:
    mapping={digest(v):i for i,v in enumerate(data['mapping_vectors'])}
result=[]
for scenario in [0,1]:
    full_folder=OUT/f'FULL_scenario{scenario}_full'
    nn_folder=OUT/f'NN20_scenario{scenario}_full'
    full=read(full_folder,'steps.jsonl')
    nn=read(nn_folder,'steps.jsonl')
    different=[i for i,(a,b) in enumerate(zip(full,nn),1) if a['action_hash']!=b['action_hash']]
    first=different[0] if different else None
    item={'scenario':scenario,'first_FULL_NN20_action_difference':first,
          'different_action_steps_in_common_prefix':len(different),
          'qualifier':'After first difference, comparisons involve policy-induced different states.'}
    if first is not None:
        a=full[first-1]
        b=nn[first-1]
        events_full=[e for e in read(full_folder,'events.jsonl') if e['step']==first]
        events_nn=[e for e in read(nn_folder,'events.jsonl') if e['step']==first]
        modules_full=[e for e in events_full if e['kind']=='module']
        modules_nn=[e for e in events_nn if e['kind']=='module']
        topology_full=[e for e in modules_full if e['module']=='topo_12_unsafe' and not e['returned_none']]
        item.update({'before_observation_identical':a['before_observation_hash']==b['before_observation_hash'],
                     'FULL_modules':modules_full,'NN20_modules':modules_nn,
                     'FULL_topology_return_in_NN_mapping':[{'hash':e['returned_action_hash'],'NN_id':mapping.get(e['returned_action_hash'])} for e in topology_full],
                     'FULL_next_rho_max':a['after_rho_max'],'NN20_next_rho_max':b['after_rho_max']})
    result.append(item)
(OUT/'FULL_NN_offline_differences.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in item.items() if k not in ['FULL_modules','NN20_modules']} for item in result],indent=2))
