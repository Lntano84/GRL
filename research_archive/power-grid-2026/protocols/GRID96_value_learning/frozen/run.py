"""Serial physical collection; all factual qualifications precede omissions."""
from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(n,x):
    p=OUT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def steps(p):
    with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def qualify(item,D):
    new=OUT/'runs'/f"r0__{item['week']}__{item['method']}";old=ROOT/next(c['reference'] for c in D['cases'] if c['week']==item['week'])
    a,b=steps(old),steps(new);assert len(a)==len(b)
    keys=['step','before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']
    assert all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b)),item['week']
    aa=sorted(old.glob('call[0-9][0-9][0-9].json'));bb=sorted(new.glob('call[0-9][0-9][0-9].json'));assert len(aa)==len(bb)
    for p,q in zip(aa,bb):
        x,y=read(p),read(q)
        assert all(x[k]==y[k] for k in ['step','state_hash','input_hash','chosen_pool_id','public_queries','order','aliases'])
        keys2=['pool_id','rho','reward','combined_hash']
        assert [[r[k] for k in keys2] for r in x['outcomes']]==[[r[k] for k in keys2] for r in y['outcomes']]
    assert read(new/'metadata.json')['nominal_episode_steps']==D['nominal_episode_steps']
    for cid in item['watch_cases']:assert (new/f'{cid}_witness.json').exists()
    return dict(week=item['week'],steps=len(a),calls=len(aa),exact_historical_trajectory=True)
def main():
    D=read(OUT/'design.json');assert read(OUT/'preflight.json')['passed']
    for name in ['code_freeze.json','data_freeze.json']:
        for rel,h in read(OUT/name).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    assert not (OUT/'finished.json').exists()
    summaries=read(OUT/'manifest.json')['summaries'];assert not summaries,'No implicit resume'
    start=time.perf_counter();qualified=[]
    for i,item in enumerate(D['execution_order']):
        if i==D['factual_replays']:save('qualification.json',dict(passed=True,checks=qualified))
        if item['omit_case']:assert read(OUT/'qualification.json')['passed']
        assert time.perf_counter()-start<D['caps']['wall_s']
        folder=OUT/'runs'/f"r0__{item['week']}__{item['method']}";assert not folder.exists()
        save('pipeline_state.json',dict(phase='factual_qualification' if not item['omit_case'] else 'single_omission',completed=i,planned=len(D['execution_order']),pid=os.getpid(),item=item))
        with (OUT/f'worker_{i:02d}.log').open('x',encoding='utf-8') as f:
            proc=subprocess.Popen([sys.executable,'-u',str(W/'worker.py'),'--index',str(i)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save('active_worker.json',dict(pid=proc.pid,parent=os.getpid(),index=i))
            try:code=proc.wait(timeout=D['caps']['episode_s']+120)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT);proc.wait();raise
        assert code==0,(i,code)
        s=read(folder/'summary.json');assert s==read(OUT/f'worker_result_{i:02d}.json')
        if not item['omit_case']:qualified.append(qualify(item,D))
        summaries.append(s);save('manifest.json',dict(planned=len(D['execution_order']),summaries=summaries))
        assert sum(x['public_forecasts'] for x in summaries)<=D['caps']['public_forecasts']
        assert sum(x['steps'] for x in summaries)<=D['caps']['physical_steps']
        print(json.dumps(dict(completed=len(summaries),planned=len(D['execution_order']),method=item['method'],steps=s['steps'])),flush=True)
    save('finished.json',dict(passed_engineering=True,runs=len(summaries),wall_s=time.perf_counter()-start))
    save('pipeline_state.json',dict(phase='awaiting_label_audit',completed=len(summaries),planned=len(summaries)))
if __name__=='__main__':
    try:main()
    except BaseException:save('pipeline_error.json',dict(traceback=traceback.format_exc()));raise
