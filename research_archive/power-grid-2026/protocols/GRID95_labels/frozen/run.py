"""Serial bounded label study, with factual qualification before any omission."""
from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid95_labels'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def save(n,x):
    p=OUT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)


def steps(p):
    with gzip.open(p/'steps.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(x) for x in f]


def qualification(D):
    checks=[]
    for item in D['execution_order'][:2]:
        new=OUT/'runs'/f"r0__{item['week']}__{item['method']}"
        c=next(c for c in D['cases'] if c['week']==item['week']);old=ROOT/c['reference']
        a,b=steps(old),steps(new);assert len(a)==len(b)
        keys=['step','before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']
        assert all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b)),item['week']
        oldcalls=sorted(old.glob('call[0-9][0-9][0-9].json'));newcalls=sorted(new.glob('call[0-9][0-9][0-9].json'))
        assert len(oldcalls)==len(newcalls)
        for p,q in zip(oldcalls,newcalls):
            x,y=read(p),read(q)
            assert all(x[k]==y[k] for k in ['step','state_hash','input_hash','chosen_pool_id','public_queries','order','aliases'])
            assert [(r['pool_id'],r['rho'],r['reward'],r['combined_hash']) for r in x['outcomes']]==[(r['pool_id'],r['rho'],r['reward'],r['combined_hash']) for r in y['outcomes']]
        for cid in item['watch_cases']:assert (new/f'{cid}_witness.json').exists()
        checks.append(dict(week=item['week'],steps=len(a),calls=len(oldcalls),exact_historical_trajectory=True))
    save('qualification.json',dict(passed=True,checks=checks,physical_steps=sum(x['steps'] for x in checks)))


def main():
    assert read(OUT/'preflight.json')['passed'];D=read(OUT/'design.json')
    for n in ['code_freeze.json','data_freeze.json']:
        for rel,h in read(OUT/n).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    assert not (OUT/'finished.json').exists();start=time.perf_counter()
    summaries=read(OUT/'manifest.json')['summaries'];assert not summaries,'No automatic resume of a partial label study'
    for index,item in enumerate(D['execution_order']):
        if index==2:qualification(D)
        if index>=2:assert read(OUT/'qualification.json')['passed']
        assert time.perf_counter()-start<D['caps']['wall_s']
        folder=OUT/'runs'/f"r0__{item['week']}__{item['method']}";assert not folder.exists()
        save('pipeline_state.json',dict(phase='reconstruction' if index<2 else 'single_omission',completed=len(summaries),planned=len(D['execution_order']),pid=os.getpid(),item=item))
        with (OUT/f'worker_{index:02d}.log').open('x',encoding='utf-8') as f:
            proc=subprocess.Popen([sys.executable,'-u',str(W/'worker.py'),'--index',str(index)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save('active_worker.json',dict(pid=proc.pid,parent=os.getpid(),index=index))
            try:code=proc.wait(timeout=D['caps']['episode_s']+120)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT);proc.wait();raise
        assert code==0,(index,code)
        s=read(folder/'summary.json');assert s==read(OUT/f'worker_result_{index:02d}.json')
        summaries.append(s);save('manifest.json',dict(planned=len(D['execution_order']),summaries=summaries))
        assert sum(x['public_forecasts'] for x in summaries)<=D['caps']['public_forecasts']
        print(json.dumps(dict(completed=len(summaries),planned=len(D['execution_order']),method=item['method'],steps=s['steps'],cost=s['cost'])),flush=True)
    save('finished.json',dict(passed_engineering=True,runs=len(summaries),summaries=summaries,wall_s=time.perf_counter()-start))
    save('pipeline_state.json',dict(phase='replays_finished_awaiting_saved_data_audit',completed=len(summaries),planned=len(D['execution_order'])))


if __name__=='__main__':
    try:main()
    except BaseException:
        save('pipeline_error.json',dict(traceback=traceback.format_exc()));raise
