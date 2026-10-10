"""Wait for fixed training, evaluate each final checkpoint once, then audit."""
import json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid10';jobs={};handles={};started=time.monotonic()
try:
    while len(jobs)<2 or any(p.poll() is None for p in jobs.values()):
        assert time.monotonic()-started<7200
        for version in ['raw','normalized']:
            folder=OUT/version/'train/gnn'
            if (folder/'failure.json').exists():raise RuntimeError(f'{version} training failed; no silent retry')
            if version not in jobs and (folder/'finished.json').exists():
                assert not (OUT/version/'evaluate').exists()
                handles[version]=(OUT/f'{version}_evaluate.log').open('w',encoding='utf-8')
                jobs[version]=subprocess.Popen([sys.executable,str(ROOT/'work/grid10/pilot.py'),'--phase','evaluate','--version',version],cwd=ROOT,stdout=handles[version],stderr=subprocess.STDOUT)
                print(json.dumps({'evaluation_started':version}),flush=True)
            if version in jobs and jobs[version].poll() not in [None,0]:raise RuntimeError(f'{version} evaluation failed')
        time.sleep(2)
    for handle in handles.values():handle.close()
    for script,args in [('audit.py',['--version','raw']),('audit.py',['--version','normalized']),('report.py',[])]:
        version=args[-1] if args else 'combined'
        with (OUT/f'{version}_{Path(script).stem}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'work/grid10'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'completed':script,'version':version}),flush=True)
    (OUT/'finished.json').write_text(json.dumps({'passed':True,'wall_s':time.monotonic()-started},indent=2),encoding='utf-8')
finally:
    for handle in handles.values():handle.close()
