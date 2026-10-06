from common import *
from coarse_policy import mean_uncertainty, select_rows, coarse_batch
from run import execute
import copy, gzip, math, time

def main():
    start=time.perf_counter();verify_design();p,d=load_data();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'FA04_execution_freeze.json').exists(), 'Do not overwrite an execution freeze'
    checks=[]
    # Exactly 55 columns; one missing model contributes uncertainty 0.5.
    probs=np.ones((4,55));missing=np.zeros(55,bool);missing[-1]=True;probs[:,-1]=.5
    probs[0,:54]=.9;probs[1,:54]=.6;probs[2,:54]=.6;probs[3,:54]=1.
    u=mean_uncertainty(probs,missing)
    expected=np.array([(54*.1+.5)/55,(54*.4+.5)/55,(54*.4+.5)/55,.5/55])
    assert np.allclose(u,expected,rtol=0,atol=1e-14)
    pool=np.array([10,20,30,40])
    for seed in SEEDS:
        chosen=select_rows(pool,u,seed,1,size=2)
        assert set(chosen)=={20,30}
        permutation=np.random.default_rng(keyed_seed(seed,'row_acquire',1)).permutation(4)
        tied=select_rows(pool,np.zeros(4),seed,1,size=2)
        expected_tie=pool[permutation[:2]].copy()
        np.random.default_rng(keyed_seed(seed,'row_execute',1)).shuffle(expected_tie)
        assert np.array_equal(tied,expected_tie)
    checks.append('55_term_hand_calculation_missing_model_and_seeded_ties')
    # Non-constant table catches accidental mean along the candidate axis.
    assert u.shape==(4,) and u[1]>u[0]>u[3]
    checks.append('candidate_axis_and_full_model_average')
    assert len(p['train'])==1048 and len(p['test'])==129 and len(p['algorithms'])==11
    nobs=nlabels=0
    for seed in SEEDS:
        old=[]
        with gzip.open(ROOT/f'outputs/fa00/runs/FIXED-100_s{seed}/actions.jsonl.gz','rt') as f:
            for line in f:
                e=json.loads(line)
                if e['phase']!='init':break
                old.append(e)
        events=[];ledger=Ledger(BUDGET,events.append);cache=ObservationCache(d['runtimes'].shape)
        env=ReplayEnvironment(d['runtimes'],d['ok'],ledger)
        s=Selector(d['X'],p['algorithms'],p['train'],cache,seed)
        assert len(s.pairs)==55
        for i in p['initial_rows'][str(seed)]:
            for a in range(11):env.observe(cache,i,a,100,'train')
        s.synchronize_labels(ledger.event)
        for typ,fields in [('observation',['i','a','kind','value','before_kind','before_value','charged_s','actual_cap_s','cached']),
                           ('pair_label',['pair','i','a','b','va','vb','ka','kb','label','weight'])]:
            left=[{k:e[k] for k in fields} for e in old if e['type']==typ]
            right=[{k:e[k] for k in fields} for e in events if e['type']==typ]
            assert left==right
            if typ=='observation':nobs+=len(left)
            else:nlabels+=len(left)
    assert nobs==660
    checks.append('all_three_initial_prefixes_and_first_labels_match_history')
    toy=dict(X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),
             runtimes=np.array([[1.,2.,3.],[3.,1.,2.],[2.,3.,1.],[3.,2.,1.]]),ok=np.ones((4,3),bool))
    tp=dict(algorithms=['a','b','c'],train=[0,1,2,3],initial_rows={'7':[0,1]})
    alternate=toy['runtimes'].copy();alternate[2:]*=1000
    twin=[]
    for runtimes in [toy['runtimes'],alternate]:
        cache=ObservationCache((4,3));env=ReplayEnvironment(runtimes,toy['ok'],Ledger(17.5))
        s=Selector(toy['X'],tp['algorithms'],tp['train'],cache,7)
        for i in [0,1]:
            for a in range(3):env.observe(cache,i,a,100,'train')
        s.synchronize_labels()
        twin.append(s)
    assert signature(twin[0])==signature(twin[1])
    # Fit once on the identical visible prefix and share that legal fitted state.
    twin[0].fit(0);twin[1]=copy.deepcopy(twin[0])
    one=coarse_batch(twin[0],7,1);two=coarse_batch(twin[1],7,1)
    for a,b in zip(one[:4],two[:4]):assert np.array_equal(a,b)
    assert one[4:]==two[4:]
    assert all(i in [2,3] for i,a in one[-1]) and len(one[-1])==6
    checks.append('two_actual_hidden_worlds_same_visible_prefix_scores_order_and_work')
    base=OUT/'preflight_toy'
    first=execute(ARMS[0],7,tp,toy,base/'continuous',math.inf,budget=17.5)
    partial=execute(ARMS[0],7,tp,toy,base/'resumed',math.inf,budget=17.5,pause_after=2)
    assert partial['status']=='paused'
    second=execute(ARMS[0],7,tp,toy,base/'resumed',math.inf,budget=17.5)
    assert first['status']==second['status']=='complete'
    for k in ['acquisition_s','executions','cache_hits','labels','input_sha256']:assert first[k]==second[k]
    assert np.array_equal(np.load(base/'continuous/predictions.npy'),np.load(base/'resumed/predictions.npy'))
    assert first['acquisition_s']==17.5
    c=np.load(base/'continuous/visible_cache.npz');assert np.any((c['kind']==1)&(c['value']<100))
    checks.append('actual_pause_resume_and_partial_row_budget_censor_match')
    cache=ObservationCache((2,3));s=Selector(np.zeros((2,2)),['a','b','c'],[0,1],cache,7)
    cache.kind[0,:2]=1;cache.value[0,:2]=100;s.synchronize_labels()
    assert all(not z for z in s.label_rows)
    checks.append('double_censored_feedback_creates_no_pair_label')
    save(OUT/'FA04_preflight.json',dict(passed=True,checks=checks,initial_observations=nobs,
        initial_label_records=nlabels,wall_s=time.perf_counter()-start,
        formal_trajectories_started=0,formal_model_fits=0,
        warning='Toy 100-tree fits were executed; guards replay historical initial feedback. No additional ASP formal fit.'))
    files={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')}
    for name in ['outputs/fa04_design/FA04_design_freeze.json','outputs/fa04_design/FA04_next_step.md',
                 'outputs/fa04/FA04_preflight.json','outputs/fa00/FA00_protocol.json','outputs/fa00/FA00_data.npz',
                 'work/fa00/core.py']:
        files[name]=sha(ROOT/name)
    save(OUT/'FA04_execution_freeze.json',dict(files=files,seeds=SEEDS,arms=ARMS,budget_s=BUDGET,
        test_outcomes_not_scored=True,new_trajectories_started=0,wall_limit_s=5400))
    print(json.dumps(dict(passed=True,checks=checks,wall_s=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
