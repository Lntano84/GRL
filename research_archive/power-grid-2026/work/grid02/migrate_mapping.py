"""Restore legacy object storage using the independent vector map, preserving every row."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'
sys.path.insert(0,str(ROOT/'work/grid01'))
from trace_agent import action_vector,digest
env=grid2op.make(str(Path(grid2op.__file__).parent/'data/l2rpn_idf_2023'),test=True,backend=LightSimBackend())
env.reset()
source=ROOT/'work/grid02/original_nn/assets/nn_act_space/action_12_unsafe_nn.npz'
destination=ROOT/'work/grid02/ljn_nn/assets/nn_act_space/action_12_unsafe_nn.npz'
data=np.load(source,allow_pickle=True)
objects=data['g2op_id_actions']
vectors=data['vect_actions']
rows=[]
restored=[]
for i,(old,v) in enumerate(zip(objects,vectors)):
    # Access the serialized state dictionary, bypassing new descriptors that
    # silently ignore the legacy storage name. This is an independent check
    # against BOTH the publisher's vector map and its stored vector cache.
    raw=np.concatenate([old.__dict__[name].astype(np.float32) for name in type(old).attr_list_vect])
    assert np.array_equal(raw,v,equal_nan=True),(i,'stored fields disagree')
    assert np.array_equal(old.__dict__['_vectorized'],v,equal_nan=True),(i,'cache disagrees')
    assert not old.__dict__['_hazards'].any() and not old.__dict__['_maintenance'].any()
    for field in ['shunt_p','shunt_q']:
        assert np.isnan(old.__dict__[field]).all(),(i,field)
    assert not old.__dict__['shunt_bus'].any()
    new=env.action_space.from_vect(v)
    assert np.array_equal(action_vector(new),raw,equal_nan=True)
    assert not new.is_ambiguous()[0]
    desc=new.as_dict()
    assert len(desc['set_bus_vect']['modif_subs_id'])==1
    restored.append(new)
    rows.append({'index':i,'vector_hash':digest(v),'stored_fields_and_cache_equal_vector':True,
                 'restored_vector_equal':True,'substation':int(desc['set_bus_vect']['modif_subs_id'][0])})
np.savez_compressed(destination,curriculum_id_actions=data['curriculum_id_actions'],
                    g2op_id_actions=np.array(restored,dtype=object),vect_actions=vectors)
with np.load(destination,allow_pickle=True) as rebuilt:
    assert np.array_equal(rebuilt['curriculum_id_actions'],data['curriculum_id_actions'])
    assert np.array_equal(rebuilt['vect_actions'],vectors)
    assert all(np.array_equal(action_vector(a),v) for a,v in zip(rebuilt['g2op_id_actions'],vectors))
result={'status':'PASS_EXACT_LEGACY_STORAGE_MIGRATION','actual_steps':0,'rows':rows,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'adapted_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'note':'All 352 legacy __dict__ fields and cached vectors equal supplied vect_actions row-by-row. Current Grid2Op uses lazy private descriptors; rebuild current objects from exactly those vectors. No action, row order or model changed.'}
(OUT/'mapping_migration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
manifest_path=OUT/'adaptation_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['NN_mapping_storage_migrated_exactly']=True
manifest['changes']+=' Legacy action-object storage migrated using exact publisher vectors, checked against stored object fields/cache; model/action meaning/order unchanged.'
manifest['hashes']['assets/nn_act_space/action_12_unsafe_nn.npz']=result['adapted_sha256']
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(result['status'],'352 rows, no physical steps',flush=True)
env.close()
