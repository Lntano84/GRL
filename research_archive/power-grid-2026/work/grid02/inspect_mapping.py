"""No physical steps. Inspect mapping semantics without changing action caches."""
import json
from pathlib import Path
import sys
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/grid01'))
from trace_agent import action_vector,digest
env=grid2op.make(str(Path(grid2op.__file__).parent/'data/l2rpn_idf_2023'),test=True,backend=LightSimBackend())
obs=env.reset()
data=np.load(ROOT/'work/grid02/original_nn/assets/nn_act_space/action_12_unsafe_nn.npz',allow_pickle=True)
actions=data['g2op_id_actions']
vectors=data['vect_actions']
current=np.stack([action_vector(x) for x in actions])
vhashes={digest(v):i for i,v in enumerate(vectors)}
matches=[vhashes.get(digest(a)) for a in current]
layout=[]
offset=0
for name in type(actions[0]).attr_list_vect:
    size=actions[0]._get_array_from_attr_name(name).size
    a=current[:,offset:offset+size]
    b=vectors[:,offset:offset+size]
    layout.append({'field':name,'size':size,'start':offset,'differing_rows':int(np.count_nonzero(np.any(a!=b,axis=1))),
                   'different_entries':int(np.count_nonzero(a!=b)),'object_unique':np.unique(a).tolist(),'vector_unique':np.unique(b).tolist()})
    offset+=size
first=actions[0]
cached=getattr(first,'_vectorized',None)
result={'object_attr_list_vect':list(type(first).attr_list_vect),'current_attr_list_vect':list(type(env.action_space()).attr_list_vect),
        'same_row_matches':sum(i==m for i,m in enumerate(matches)), 'set_matches':sum(m is not None for m in matches),
        'all_set_match_indices':matches,'fields':layout,
        'first_action_cached_vector_shape':list(cached.shape) if cached is not None else None,
        'first_cached_equal_file_vector':bool(np.array_equal(cached,vectors[0])) if cached is not None else None,
        'first_current_object_vs_cached':bool(np.array_equal(cached,current[0])) if cached is not None else None,
        'first_curriculum_type':str(type(data['curriculum_id_actions'][0])), 'first_curriculum_value':str(data['curriculum_id_actions'][0])[:180],
        'first_object_dict_array_shapes':{k:list(v.shape) for k,v in first.__dict__.items() if isinstance(v,np.ndarray)},
        'actual_steps':0}
(ROOT/'outputs/grid02/mapping_diagnostic.json').write_text(json.dumps(result,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['all_set_match_indices','first_object_dict_array_shapes']},default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))
env.close()
