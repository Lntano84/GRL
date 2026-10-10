"""Finish the existing fixed schedule and audits; no new training or tuning."""
import json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
jobs={};handles={};started=time.monotonic()
assert not (OUT/'completion_supervisor_finished.json').exists()
try:
    while len(jobs)<2 or any(p.poll() is None for p in jobs.values()):
        assert time.monotonic()-started<7200,'Completion supervisor time cap'
        for kind in ['gnn','mlp']:
            train=OUT/'v2/train'/kind
            if (train/'failure.json').exists():raise RuntimeError(f'{kind} training failure; preserved and not retried')
            if kind not in jobs and (train/'finished.json').exists():
                assert not (OUT/'v2/evaluate'/kind).exists(),'Refuse an evaluation overwrite'
                handles[kind]=(OUT/f'{kind}_v2_evaluate.log').open('w',encoding='utf-8')
                jobs[kind]=subprocess.Popen([sys.executable,str(ROOT/'work/grid08/pilot_v2.py'),'--phase','evaluate','--kind',kind],
                    cwd=ROOT,stdout=handles[kind],stderr=subprocess.STDOUT)
                print(json.dumps({'evaluation_started':kind,'checkpoint':str((train/'final.pt').relative_to(ROOT))}),flush=True)
            if kind in jobs and jobs[kind].poll() not in [None,0]:raise RuntimeError(f'{kind} evaluation failed')
        time.sleep(2)
    for handle in handles.values():handle.close()
    for script,args in [('training_review.py',['--version','v2']),('audit_report.py',['--version','v2']),('resume_audit.py',[]),('credit_diagnostic.py',[]),('failure_audit.py',[]),('delivery_report.py',[]),('plot_delivery.py',[])]:
        with (OUT/('final_'+Path(script).stem+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'work/grid08'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'completed':script}),flush=True)
    (OUT/'completion_supervisor_finished.json').write_text(json.dumps({'passed':True,'wall_s':time.monotonic()-started},indent=2),encoding='utf-8')
finally:
    for handle in handles.values():handle.close()
