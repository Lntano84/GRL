"""One bounded serial study; qualification gates interventions, no automatic retry."""
import gzip,json,os,subprocess,sys,time,traceback
from common106 import ROOT,W,OUT,read,save,verify,utc
def rows(folder):
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def qualify():
    d=read(OUT/'design.json');old=ROOT/d['reference'];new=OUT/'runs/r0__2037-03-02_100__FRAME_QUAL'
    a,b=rows(old),rows(new);assert len(a)==len(b)==2017
    keys=['step','before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']
    assert all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b))
    ps=sorted(old.glob('call[0-9][0-9][0-9].json'));qs=sorted(new.glob('call[0-9][0-9][0-9].json'))
    assert len(ps)==len(qs)
    for p,q in zip(ps,qs):
        x,y=read(p),read(q)
        for k in ['step','state_hash','input_hash','order','aliases','chosen_pool_id','public_queries','local_queries','returned_action_hash']:
            assert x[k]==y[k],(k,p.name)
        fields=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible','phase','combined_hash']
        assert [{k:r[k] for k in fields} for r in x['outcomes']]==[{k:r[k] for k in fields} for r in y['outcomes']]
    save(OUT/'qualification.json',dict(passed=True,steps=2017,search_calls=len(ps),exact_historical_physical_trajectory=True,
        actual_paid_feedback_reproduced=True,utc=utc()))
def main():
    verify();assert read(OUT/'preflight.json')['passed'];d=read(OUT/'design.json')
    assert not (OUT/'runs').exists() and not read(OUT/'manifest.json')['summaries'],'No duplicate or automatic resume'
    start=time.perf_counter();summaries=[]
    for i,item in enumerate(d['execution_order']):
        remaining=d['caps']['wall_s']-(time.perf_counter()-start);assert remaining>0
        if i:assert read(OUT/'qualification.json')['passed']
        save(OUT/'pipeline_state.json',dict(phase='running',completed=i,planned=3,item=item,pid=os.getpid(),utc=utc()))
        with (OUT/f'worker_{i:02d}.log').open('x',encoding='utf8') as f:
            proc=subprocess.Popen([sys.executable,'-u',str(W/'worker.py'),'--index',str(i)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save(OUT/'active_worker.json',dict(pid=proc.pid,parent=os.getpid(),index=i))
            try:code=proc.wait(timeout=min(remaining,d['caps']['episode_s']+60))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT);proc.wait();raise
        assert code==0,(i,code)
        p=OUT/'runs'/f"r0__{item['week']}__{item['method']}";s=read(p/'summary.json')
        assert s==read(OUT/f'worker_result_{i:02d}.json') and s['new_model_fits']==0
        summaries.append(s);save(OUT/'manifest.json',dict(planned=3,summaries=summaries))
        assert sum(r['steps'] for r in summaries)<=d['caps']['physical_steps']
        assert sum(r['public_forecasts'] for r in summaries)<=d['caps']['public_forecasts']
        if i==0:qualify()
        print(json.dumps(dict(completed=i+1,planned=3,method=item['method'],steps=s['steps'],cost=s['cost']),ensure_ascii=False),flush=True)
    save(OUT/'finished.json',dict(passed_engineering=True,runs=3,wall_s=time.perf_counter()-start,summaries=summaries))
    save(OUT/'pipeline_state.json',dict(phase='replays_finished_awaiting_audit',completed=3,planned=3,utc=utc()))
if __name__=='__main__':
    try:main()
    except BaseException:
        save(OUT/'pipeline_error.json',dict(utc=utc(),traceback=traceback.format_exc()));raise
