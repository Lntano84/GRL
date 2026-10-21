"""Audit one new paid FULL teacher pilot and separate learner inputs/targets."""
from pathlib import Path
import json,gzip,hashlib,shutil,collections
import numpy as np
from labels import collect,key,input_at,FIELDS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid100_data_foundation'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    D=read(OUT/'design.json');item=D['pilot'];assert item['partition']=='train';folder=OUT/'teacher_runtime/runs'/f"r0__{item['scenario']}__FULL_TEACHER";result=read(OUT/'teacher_runtime/worker_result_00.json');assert result==read(folder/'summary.json')
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as f:steps=[json.loads(line) for line in f]
    assert len(steps)==result['steps']
    with np.load(folder/'vectors.npz') as z:
        assert len(z['action'])==len(z['observation'])==len(steps)
        for r,a,o in zip(steps,z['action'],z['observation']):assert digest(a)==r['action_hash'] and digest(o)==r['after_hash']
    for before,after in zip(steps,steps[1:]):assert before['after_hash']==after['before_hash']
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];dataset=OUT/'dataset_pilot';assert not dataset.exists(),'Do not overwrite completed data';dataset.mkdir();manifest=[];excluded=[];counts=collections.Counter()
    for p in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        c=read(p);assert c['policy']=='FULL' and c['public_queries']==len(c['order'])==len(c['outcomes']);truth={r['pool_id']:r for r in c['outcomes']};assert set(truth)==set(c['order'])
        for r in c['outcomes']:
            expected=bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['illegal'] and not r['ambiguous'] and not r['exceptions']);assert expected==r['strict_admissible'];counts['feedback_legality_checks']+=1
        b=max((r for r in c['outcomes'] if r['strict_admissible']),key=key,default=None);assert (None if b is None else b['pool_id'])==c['chosen_pool_id']
        state=p.with_name(p.stem+'_state.npz')
        with np.load(state) as z:assert digest(z['observation'])==c['state_hash']==steps[c['step']-1]['before_hash']
        dest=dataset/'states'/state.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(state,dest)
        rows,xx=collect(c,subs,p.relative_to(ROOT).as_posix());excluded.extend(dict(call=p.stem,**x) for x in xx)
        for row in rows:
            i=row['sample_id'];n=row['visible_count'];visible=truth.keys()&set(c['order'][:n]);input_data=dict(row['input'],observation_file='states/'+state.name,public_action_library='work/grid02/ljn_nn/assets/action_N1_unsafe.npz')
            assert not set(input_data)&{'local_reward_gain','after_id','label','target'}
            assert set(r['pool_id'] for r in input_data['visible'])==visible
            save(dataset/'inputs'/(i+'.json'),input_data);save(dataset/'targets'/(i+'.json'),row['label'])
            manifest.append(dict(sample_id=i,partition='train',calendar_group=item['date'],scenario=item['scenario'],physical_state_group=item['scenario']+'|'+p.stem,source_call=p.relative_to(ROOT).as_posix(),source_sha256=sha(p),state_hash=c['state_hash'],step=c['step'],visible_count=n,input_file='inputs/'+i+'.json',target_file='targets/'+i+'.json'))
            counts['positive_gain']+=row['label']['local_reward_gain']>0;counts['zero_gain']+=row['label']['local_reward_gain']==0
        counts['full_teacher_calls']+=1;counts['paid_candidate_queries']+=len(c['order'])
    save(dataset/'manifest.json',dict(samples=manifest,excluded_prefixes=excluded,partition='train',calendar_groups=1,not_independent_samples='Prefixes share physical state/episode and must stay in one split; new scenario uses FULL teacher path, not FRAME.',teacher_complete=result['complete'],teacher_steps=result['steps']))
    assert counts['full_teacher_calls']==result['n1_calls'] and counts['paid_candidate_queries']==result['n1_forecasts']
    # Read-back separation and counts, not a predictive evaluation.
    for r in manifest:
        x=read(dataset/r['input_file']);y=read(dataset/r['target_file']);assert y['sample_id']==r['sample_id'] and x['known_completion_queries']==y['known_completion_queries']
        assert not set(x)&set(y)-{'known_completion_queries'}
    tests=[x for x in D['scenario_plan'] if x['partition']=='test'];E=ROOT/D['environment'];assert not any((E/'chronics'/x['scenario']).exists() for x in tests)
    receipt=dict(passed=True,scenario=item['scenario'],partition='train',teacher_steps=result['steps'],teacher_complete=result['complete'],teacher_cost=result['cost'],samples=len(manifest),excluded_prefixes=len(excluded),counts=dict(counts),calendar_groups=1,final_test_materialized=False,validation_samples=0,model_fits=0,scope='First new train-only data/label engineering pilot. No performance advantage or long-term value claim.')
    save(OUT/'pilot_collection_finished.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
