"""Six frozen FA03B trajectories. No test scoring in this process."""
from common import *
from policy import LexSelector, lexical_indices, cheap_indices, row_batch
import argparse, math, gzip, time

def execute(arm,seed,p,d,dest,deadline,budget=BUDGET,pause_after=None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/'result.json'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['status']=='complete':return old
    attempt_start=time.perf_counter()
    checkpoint=dest/'checkpoint.pkl'
    logpath=dest/'actions.jsonl'
    resuming=checkpoint.exists()
    if resuming:
        with checkpoint.open('rb') as f:state=pickle.load(f)
        selector=state['selector'];cache=selector.cache
        prior_wall=state['active_wall_s']
        ledger=Ledger(budget)
        for k,v in state['ledger'].items():setattr(ledger,k,v)
        history=state['history'];phase=state['phase'];pending=state['pending']
        round_no=state['round']
        # Replay only committed log records after the last complete checkpoint.
        with logpath.open('rb') as f:
            f.seek(state['offset']);tail=f.read()
        valid_end=state['offset']
        for raw in tail.splitlines(keepends=True):
            if not raw.endswith(b'\n'):break
            e=json.loads(raw);valid_end+=len(raw)
            ledger.event_id=e['event_id']+1;ledger.round=e['round'];ledger.phase=e['phase']
            ledger.train=e['train_cost_s'];ledger.validation=e['validation_cost_s']
            if e['type']=='batch':
                pending=dict(work=e['work'],cursor=0,initial=e['initial'])
                phase='execute';round_no=e['round']
            elif e['type']=='observation':
                cache.kind[e['i'],e['a']]=e['kind'];cache.value[e['i'],e['a']]=e['value']
                pending['cursor']=e['work_index']+1
                if e['cached']:ledger.cache_hits+=1
                else:
                    ledger.executions+=1
                    if e['kind']==2:ledger.completions+=1
                    else:ledger.cancellations+=1
            elif e['type']=='pair_label':
                selector.label_rows[e['pair']][e['i']]=(e['va'],e['vb'],e['label'],e['weight'])
            elif e['type']=='batch_complete':phase='fit'
            elif e['type']=='terminal_start':phase='terminal'
        with logpath.open('r+b') as f:f.truncate(valid_end)
    else:
        assert not logpath.exists(), 'Uncheckpointed existing log; no overwrite'
        cache=ObservationCache(d['runtimes'].shape)
        cls=LexSelector if arm=='LEX-PC-U-PD' else Selector
        selector=cls(d['X'],p['algorithms'],p['train'],cache,seed)
        ledger=Ledger(budget);history=[];phase='init';round_no=0;pending=None;prior_wall=0.
    context={}
    log=logpath.open('ab')
    def emit(e):
        if e['type']=='observation':e['work_index']=context['work_index']
        log.write((json.dumps(e,separators=(',',':'),allow_nan=False)+'\n').encode())
        log.flush();os.fsync(log.fileno())
    ledger.writer=emit
    env=ReplayEnvironment(d['runtimes'],d['ok'],ledger)
    def checkpoint_now():
        log.flush();os.fsync(log.fileno())
        dump(checkpoint,dict(selector=selector,ledger={k:getattr(ledger,k) for k in [
            'train','validation','executions','cache_hits','completions','cancellations',
            'event_id','phase','round']},history=history,phase=phase,pending=pending,
            round=round_no,offset=log.tell(),active_wall_s=prior_wall+time.perf_counter()-attempt_start))
        save(dest/'progress.json',dict(arm=arm,seed=seed,phase=phase,round=round_no,
            spent_s=ledger.train,remaining_s=ledger.remaining,labels=sum(map(len,selector.label_rows)),
            active_wall_s=prior_wall+time.perf_counter()-attempt_start,resuming=resuming))
    status='complete';reason='budget';done_work=0;terminal_fit_s=0.
    try:
        if not resuming:checkpoint_now()
        # A crash between the last row feedback and its label events is repaired
        # before the next feedback, preserving first legal acquisition values.
        if resuming and phase=='execute' and not pending['initial'] and arm=='RANDOM-ROW':
            selector.synchronize_labels(ledger.event)
        while phase!='done':
            if time.perf_counter()>=deadline:raise WallLimit
            if phase=='init':
                pending=dict(work=[[int(i),a] for i in p['initial_rows'][str(seed)]
                                   for a in range(len(p['algorithms']))],cursor=0,initial=True)
                ledger.event(dict(type='batch',initial=True,work=pending['work'],selected_rows=p['initial_rows'][str(seed)]))
                phase='execute'
            elif phase=='plan':
                if ledger.remaining<=1e-8:phase='terminal';continue
                round_no+=1;ledger.round=round_no;ledger.phase='train'
                start=time.perf_counter()
                if arm=='LEX-PC-U-PD':
                    c=selector.candidates(CAP)
                    if not len(c):reason='candidate_pool_exhausted';phase='terminal';continue
                    ix=lexical_indices(c,566,seed,round_no);cheap=cheap_indices(c,566,seed,round_no)
                    batch=c[ix]
                    work=[]
                    for row in batch:
                        i,pi=int(row[0]),int(row[1]);a,b=selector.pairs[pi]
                        if i not in selector.label_rows[pi]:work.extend([[i,a],[i,b]])
                    np.savez_compressed(dest/f'candidates_r{round_no:04d}.npz',scores=c,chosen=ix,cheap=cheap)
                    selected_rows=[]
                    costs,counts=np.unique(c[:,3],return_counts=True)
                    diag=dict(candidate_count=len(c),cost_tied_candidates=int(counts[counts>1].sum()),
                        cheap_set_overlap=len(set(ix)&set(cheap))/len(ix))
                else:
                    pool,selected_rows,work=row_batch(cache,p['train'],seed,round_no)
                    if not len(pool):reason='candidate_pool_exhausted';phase='terminal';continue
                    np.savez_compressed(dest/f'candidates_r{round_no:04d}.npz',pool=pool,selected=selected_rows)
                    diag=dict(candidate_count=len(pool))
                selector.selection_s+=time.perf_counter()-start
                pending=dict(work=work,cursor=0,initial=False)
                ledger.event(dict(type='batch',initial=False,work=work,selected_rows=selected_rows,**diag))
                phase='execute'
            elif phase=='execute':
                while pending['cursor']<len(pending['work']):
                    if time.perf_counter()>=deadline:raise WallLimit
                    index=pending['cursor'];i,a=pending['work'][index]
                    context['work_index']=index
                    try:env.observe(cache,i,a,CAP,'train')
                    except Exhausted:
                        phase='terminal';reason='budget_in_partial_batch';break
                    pending['cursor']+=1;done_work+=1
                    if arm=='RANDOM-ROW' and not pending['initial']:
                        selector.synchronize_labels(ledger.event)
                    if pause_after is not None and done_work>=pause_after:raise WallLimit
                if phase=='execute':
                    selector.synchronize_labels(ledger.event)
                    ledger.event(dict(type='batch_complete'))
                    phase='fit'
            elif phase=='fit':
                start=time.perf_counter();selector.fit(round_no,deadline);pred=selector.predict()
                np.save(dest/f'predictions_r{round_no:04d}.npy',pred)
                rec=dict(round=round_no,spent_s=ledger.train,labels=sum(map(len,selector.label_rows)),
                    observations=int(np.count_nonzero(cache.kind)),input_sha256=signature(selector),
                    fit_wall_s=time.perf_counter()-start)
                history=[h for h in history if h['round']!=round_no]+[rec]
                save(dest/'history.json',history)
                phase='plan';checkpoint_now()
                print(f'{arm} s{seed} r{round_no} paid={ledger.train:.1f} labels={rec["labels"]} wall={prior_wall+time.perf_counter()-attempt_start:.1f}s',flush=True)
            elif phase=='terminal':
                ledger.event(dict(type='terminal_start'))
                selector.synchronize_labels(ledger.event)
                before=selector.fit_s
                selector.fit(0,deadline);terminal_fit_s=selector.fit_s-before
                predictions=selector.predict()
                for pi,m in enumerate(selector.pair_models):
                    if m is not None:assert m.learner.estimator.random_state==keyed_seed(seed,'pair',0,pi)
                for a,m in enumerate(selector.regressors):
                    assert m.model.random_state==keyed_seed(seed,'reg',0,a)
                np.save(dest/'predictions.npy',predictions)
                np.savez_compressed(dest/'visible_cache.npz',kind=cache.kind,value=cache.value)
                save(dest/'labels.json',labels_json(selector));dump(dest/'selector.pkl',selector)
                phase='done';checkpoint_now()
            else:raise ValueError(phase)
    except WallLimit:
        status='paused';reason='active_wall_limit_or_preflight_pause';checkpoint_now()
    finally:
        log.flush();os.fsync(log.fileno());log.close()
    result=dict(arm=arm,seed=seed,status=status,stop_reason=reason,rounds=round_no,
        acquisition_s=ledger.train,validation_s=ledger.validation,budget_s=budget,remaining_s=ledger.remaining,
        executions=ledger.executions,cache_hits=ledger.cache_hits,completions=ledger.completions,
        cancellations=ledger.cancellations,labels=sum(map(len,selector.label_rows)),
        informative_labels=sum(sum(v[3]>0 for v in z.values()) for z in selector.label_rows),
        observations=int(np.count_nonzero(cache.kind)),full_rows=int(np.all(cache.kind[p['train']]>0,axis=1).sum()),
        wall_s=prior_wall+time.perf_counter()-attempt_start,fit_s=selector.fit_s,
        prediction_s=selector.predict_s,selection_s=selector.selection_s,terminal_fit_s=terminal_fit_s,
        input_sha256=signature(selector),resumed=resuming)
    save(result_path,result)
    if status=='complete':
        with logpath.open('rb') as src,gzip.open(dest/'actions.jsonl.gz','wb') as dst:
            for chunk in iter(lambda:src.read(1024*1024),b''):dst.write(chunk)
        save(dest/'artifacts.json',{x.name:sha(x) for x in dest.iterdir()
                                   if x.is_file() and x.name not in ['artifacts.json','checkpoint.pkl','progress.json']})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args();verify_design()
    freeze=json.loads((OUT/'FA03B_execution_freeze.json').read_text())
    for path,value in freeze['files'].items():assert sha(ROOT/path)==value,path
    assert json.loads((OUT/'FA03B_preflight.json').read_text())['passed']
    p,d=load_data()
    selected=[(arm,seed) for seed in SEEDS for arm in ARMS]
    if args.arm:selected=[(args.arm,args.seed)] if args.seed else [(args.arm,s) for s in SEEDS]
    previous=0.
    for arm,seed in [(a,s) for s in SEEDS for a in ARMS]:
        path=OUT/'runs'/f'{arm}_s{seed}'/'progress.json'
        if path.exists():previous+=json.loads(path.read_text())['active_wall_s']
    deadline=time.perf_counter()+max(0,7200-previous)
    for arm,seed in selected:
        if time.perf_counter()>=deadline:break
        execute(arm,seed,p,d,OUT/'runs'/f'{arm}_s{seed}',deadline)
        results=[json.loads(x.read_text()) for x in (OUT/'runs').glob('*/result.json')]
        complete=[r for r in results if r['status']=='complete']
        save(OUT/'FA03B_training_results.json',complete)
        save(OUT/'RUN_STATE.json',dict(status='predictions_complete_pending_audit' if len(complete)==6 else 'running_or_paused',
            completed=len(complete),target=6,test_scored=False,active_jobs=len(complete)<6))
        if any(r['status']=='paused' for r in results):break

if __name__=='__main__':main()
