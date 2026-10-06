from common import *
from policy import lexical_indices, cheap_indices, row_batch
from run import execute
import ast, gzip, math, time
import pandas as pd

def main():
    design=verify_design();p,d=load_data();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'FA03B_execution_freeze.json').exists(),'No silent replacement of execution freeze'
    start=time.perf_counter();checks=[];counts={}
    assert p['algorithms']==sorted(p['algorithms'])
    assert len(p['train'])==1048 and len(p['test'])==129
    assert math.ceil(.01*(len(p['train'])-20)*55)==566
    assert math.ceil(.01*(len(p['train'])-20))==11
    checks.append('algorithm_order_and_public_batch_formula')
    # Import only the author's pure sorting function, not the experiment script.
    author=ROOT/'work/fa03_design/source/Algorithm_Selection_Lexical_Ordering.py'
    tree=ast.parse(author.read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='lexical_sort')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(author),'exec'),ns)
    table=np.array([[0,0,.2,1,4],[0,1,.4,1,1],[1,0,.4,1,3],[1,1,.4,1,3],
                    [2,0,.9,2,8],[2,1,.1,0,0]],dtype=float)
    for seed in SEEDS:
        for round_no in [1,2]:
            canonical=np.lexsort((table[:,1],table[:,0]))
            rng=np.random.default_rng(keyed_seed(seed,'acquire',round_no))
            initial=canonical[rng.permutation(len(table))]
            frame=pd.DataFrame(table[initial],columns=['i','pi','Uncertainty','PredCost','PredDiff'],index=initial)
            expected=ns['lexical_sort'](frame,{'PredCost':'min','Uncertainty':'max','PredDiff':'max'}).index.to_numpy().copy()
            np.random.default_rng(keyed_seed(seed,'batch',round_no)).shuffle(expected)
            assert np.array_equal(expected,lexical_indices(table,len(table),seed,round_no))
    checks.append('official_lexical_function_exact_order_six_tables')
    # Tests deliberately include ties on the first two keys and an exact triple tie.
    ix=lexical_indices(table,len(table),7,1)
    assert set(ix)==set(range(len(table)))
    assert set(lexical_indices(table,2,7,1))=={5,2} or set(lexical_indices(table,2,7,1))=={5,3}
    checks.append('lexical_three_keys_and_exact_tie_randomization')

    # Initial feedback identity: all 660 declared measurements are checked
    # against frozen past logs, including cache values and first label records.
    measurements=0;label_records=0
    for seed in SEEDS:
        old=[]
        with gzip.open(ROOT/f'outputs/fa00/runs/FIXED-100_s{seed}/actions.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                e=json.loads(line)
                if e['phase']!='init':break
                old.append(e)
        events=[];ledger=Ledger(BUDGET,events.append);cache=ObservationCache(d['runtimes'].shape)
        env=ReplayEnvironment(d['runtimes'],d['ok'],ledger)
        s=Selector(d['X'],p['algorithms'],p['train'],cache,seed)
        for i in p['initial_rows'][str(seed)]:
            for a in range(len(p['algorithms'])):env.observe(cache,i,a,100,'train')
        s.synchronize_labels(ledger.event)
        old_obs=[e for e in old if e['type']=='observation']
        new_obs=[e for e in events if e['type']=='observation']
        assert len(old_obs)==len(new_obs)==220
        for a,b in zip(old_obs,new_obs):
            for k in ['i','a','kind','value','before_kind','before_value','charged_s','planned_cap_s','actual_cap_s','cached']:
                assert a[k]==b[k],(seed,k)
        old_labels=[{k:e[k] for k in ['pair','i','a','b','va','vb','ka','kb','label','weight']}
                    for e in old if e['type']=='pair_label']
        new_labels=[{k:e[k] for k in ['pair','i','a','b','va','vb','ka','kb','label','weight']}
                    for e in events if e['type']=='pair_label']
        assert old_labels==new_labels
        measurements+=len(new_obs);label_records+=len(new_labels)
    checks.append('three_initial_feedback_and_first_label_identities')
    counts.update(initial_observations=measurements,initial_label_records=label_records)

    # Both worlds have the same revealed prefix; unrevealed cells differ.
    cache1=ObservationCache((4,3));cache2=ObservationCache((4,3))
    r1=np.ones((4,3))*900;r2=np.ones((4,3))*500
    for r in [r1,r2]:r[0,0]=1
    e1=ReplayEnvironment(r1,np.ones((4,3),bool),Ledger(30))
    e2=ReplayEnvironment(r2,np.ones((4,3),bool),Ledger(30))
    for env,cache in [(e1,cache1),(e2,cache2)]:
        env.observe(cache,0,0,2,'train');env.observe(cache,0,1,2,'train')
    assert cache1.snapshot()==cache2.snapshot()
    pool1,rows1,work1=row_batch(cache1,[0,1,2,3],7,1)
    pool2,rows2,work2=row_batch(cache2,[0,1,2,3],7,1)
    assert np.array_equal(pool1,pool2) and rows1==rows2 and work1==work2
    assert [0,0] not in work1 # already completed
    e1.observe(cache1,0,0,2,'train');assert e1.ledger.train==3
    checks.append('actual_two_world_feedback_row_dispatch_and_cached_zero_cost')
    ledger=Ledger(2.5);cache=ObservationCache((1,2))
    env=ReplayEnvironment([[0,3]],[[True,True]],ledger)
    env.observe(cache,0,0,100,'train');env.observe(cache,0,1,100,'train')
    assert ledger.train==2.5 and cache.kind[0,0]==2 and cache.kind[0,1]==1 and cache.value[0,1]==2.5
    checks.append('zero_time_success_and_final_budget_censor')
    s=Selector(np.zeros((4,2)),['a','b','c'],[0,1,2,3],ObservationCache((4,3)),7)
    s.cache.kind[0,:2]=1;s.cache.value[0,:2]=100;s.synchronize_labels()
    assert all(not z for z in s.label_rows)
    checks.append('double_censor_has_no_pair_label')

    # Exercise actual runner pause/resume, with toy data only. Original 100-tree
    # model parameters are retained; no real-scene extra model is fitted.
    toy=dict(X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),
             runtimes=np.array([[1.,2.,3.],[3.,1.,2.],[2.,3.,1.],[3.,2.,1.]]),
             ok=np.ones((4,3),bool))
    tp=dict(algorithms=['a','b','c'],train=[0,1,2,3],initial_rows={'7':[0,1]})
    base=OUT/'preflight_toy'
    first=execute('RANDOM-ROW',7,tp,toy,base/'continuous',math.inf,budget=17.5)
    partial=execute('RANDOM-ROW',7,tp,toy,base/'resumed',math.inf,budget=17.5,pause_after=2)
    assert partial['status']=='paused'
    second=execute('RANDOM-ROW',7,tp,toy,base/'resumed',math.inf,budget=17.5)
    assert first['status']==second['status']=='complete'
    for k in ['acquisition_s','executions','cache_hits','labels','input_sha256']:
        assert first[k]==second[k],k
    assert np.array_equal(np.load(base/'continuous/predictions.npy'),np.load(base/'resumed/predictions.npy'))
    checks.append('actual_runner_pause_resume_exact_cache_labels_and_predictions')
    save(OUT/'FA03B_preflight.json',dict(passed=True,checks=checks,counts=counts,
        wall_s=time.perf_counter()-start,toy_runner_paths=[str(base/'continuous'),str(base/'resumed')],
        formal_trajectories_started=0,formal_model_fits=0,
        warning='Toy smoke models were fitted; initial feedback guards replay 660 historical observations, not additional formal trajectories.'))
    files={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')}
    for name in ['outputs/fa03b_design/FA03B_design_freeze.json','outputs/fa03b_design/FA03B_next_step.md','outputs/fa03b/FA03B_preflight.json']:
        files[name]=sha(ROOT/name)
    save(OUT/'FA03B_execution_freeze.json',dict(files=files,seeds=SEEDS,arms=ARMS,budget_s=BUDGET,
        test_outcomes_not_scored=True,new_trajectories_started=0))
    print(json.dumps(dict(passed=True,checks=checks,counts=counts,wall_s=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
