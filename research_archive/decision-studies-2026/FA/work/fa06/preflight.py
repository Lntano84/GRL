from common import *
import tempfile
import time

def main():
    begin=time.perf_counter();d=design();OUT.mkdir(parents=True,exist_ok=True)
    checks=[]
    cache=ObservationCache((3,11))
    cache.kind[0,0:2]=[2,1];cache.value[0,0:2]=[20,100]
    cache.kind[1,0:2]=[1,2];cache.value[1,0:2]=[30,10]
    cache.kind[2,0:2]=[2,2];cache.value[2,0:2]=[8,8]
    t=make_labels(cache,[0,1,2])[0]
    assert [(r['ORIGINAL'],r['UNIT'],r['NATIVE-UPPER']) for r in t]==[(980,1,5980),(290,1,5990)]
    assert [r['label'] for r in t]==[0,1]
    checks.append('hand_weights_reverse_direction_and_common_tie_exclusion')
    cache.kind[0,0:2]=[2,1];cache.value[0,0:2]=[20,34.5]
    r=make_labels(cache,[0])[0][0];assert r['ORIGINAL']==325 and r['NATIVE-UPPER']==5980
    cache.value[0,0:2]=[20,20];r=make_labels(cache,[0])[0][0]
    assert r['label']==0 and r['ORIGINAL']>0
    checks.append('non100_tail_and_equal_bound_type_semantics')
    assert hard_vote([[0,0],[1,1],[1,0],[0,1]]).tolist()==[0,0]
    checks.append('original_first_encounter_vote_tie')
    runtime,ok=train_truth(d);x,prep=features(d)
    outside=sorted(set(range(len(d['ids'])))-set(d['train']))
    assert np.all(runtime[outside]==600) and not ok[outside].any()
    assert set(d['train']).isdisjoint(d['test']) and set(d['train']).isdisjoint(d['reserved_unused'])
    checks.append('outcome_numeric_loading_training_only')
    changed_runtime=runtime.copy();changed_ok=ok.copy()
    changed_runtime[d['test']]=17;changed_ok[d['test']]=True
    c1,l1,e1,r1=collect(d,runtime,ok,7)
    c2,l2,e2,r2=collect(d,changed_runtime,changed_ok,7)
    canonical=lambda events:[{k:v for k,v in e.items() if k!='active_wall_s'} for e in events]
    assert canonical(e1)==canonical(e2)
    assert support_hash(l1,x)==support_hash(l2,x)
    assert np.array_equal(c1.kind,c2.kind) and np.array_equal(c1.value,c2.value)
    checks.append('real_full_collector_test_hidden_time_and_status_invariance')
    # Exercise paid-prefix resume with the same real runner, including a cut
    # inside a pending batch; the interrupted file keeps all completed feedback.
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        dest=Path(tmp);kept=e1[:15]
        with (dest/'actions.jsonl').open('w',encoding='utf-8') as f:
            for e in kept:f.write(json.dumps(e)+'\n')
        c3,l3,e3,r3=collect(d,runtime,ok,7,dest)
        assert canonical(e1)==canonical(e3) and support_hash(l3,x)==support_hash(l1,x)
        assert r3['charged_s']==r1['charged_s']
    checks.append('real_paid_prefix_resume_preserves_actions_and_cost')
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        dest=Path(tmp)
        c4,l4,e4,r4=collect(d,runtime,ok,7,dest)
        assert (dest/'actions.jsonl').exists()
        on_disk=[json.loads(line) for line in (dest/'actions.jsonl').open()]
        assert canonical(on_disk)==canonical(e4)==canonical(e1)
        assert support_hash(l4,x)==support_hash(l1,x)
    checks.append('fresh_log_created_and_every_feedback_persisted')
    for pi,table in enumerate(l1):
        for arm in ARMS:
            assert all(r[arm]>0 for r in table)
    # Identity is additionally enforced on actual serialized fit inputs in run/audit.
    checks.append('weights_positive_on_common_support')
    result=dict(status='passed',checks=checks,count=len(checks),wall_s=time.perf_counter()-begin,
        diagnostic_full_collector_runs=4,formal_acquisitions=0,new_model_fits=0,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    save(OUT/'FA06_preflight.json',result);print(json.dumps(result))

if __name__=='__main__':main()
