"""Recover flushed, complete fit checkpoints after an external session shutdown.

No policy change. Archive incomplete gzip containers; preserve every JSON event.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path
import numpy as np
import policy

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUT=PROJECT/'outputs'/'fa01'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_freeze():
    frozen=read(OUT/'FA01_freeze_manifest.json')
    for section in ['files','inherited_files']:
        for path,expected in frozen[section].items():assert digest(PROJECT/path)==expected,path

def prepare(seed):
    dest=OUT/'runs'/f'CHEAP-100_s{seed}'
    assert not (dest/'result.json').exists()
    assert not (dest/'resume_manifest.json').exists(),'No repeated recovery'
    archive=dest/'actions.INTERRUPTED.jsonl.gz'
    raw=(dest/'actions.jsonl.gz').read_bytes()
    dec=zlib.decompressobj(31);text=dec.decompress(raw)
    assert text.endswith(b'\n'),'No partial JSON event permitted'
    events=[json.loads(line) for line in text.splitlines()]
    history=read(dest/'history.json');state=read(dest/'RUN_STATE.json')
    with (dest/'checkpoint.pkl').open('rb') as f:ck=pickle.load(f)
    assert history[-1]['round']==ck['round']==state['round']
    assert events[-1]['type']=='round_end' and events[-1]['round']==ck['round']
    assert all(e['event_id']==n for n,e in enumerate(events))
    assert events[-1]['train_cost_s']==ck['ledger']['train']==state['train_cost_s']
    kinds=np.zeros(ck['train_cache'].kind.shape,dtype=np.int8)
    values=np.zeros(kinds.shape,float)
    counts=dict(executions=0,cache_hits=0,completions=0,cancellations=0)
    labels=set()
    for e in events:
        if e['type']=='observation':
            assert e['role']=='train'
            kinds[e['i'],e['a']]=e['kind'];values[e['i'],e['a']]=e['value']
            if e['cached']:counts['cache_hits']+=1
            else:
                counts['executions']+=1
                counts['completions' if e['kind']==2 else 'cancellations']+=1
        elif e['type']=='pair_label':labels.add((e['pair'],e['i']))
    assert np.array_equal(kinds,ck['train_cache'].kind)
    assert np.array_equal(values,ck['train_cache'].value)
    assert labels=={(pi,i) for pi,rows in enumerate(ck['selector'].label_rows) for i in rows}
    assert ck['selector'].cache is ck['train_cache']
    assert np.array_equal(ck['predictions'],np.load(dest/history[-1]['predictions_file']))
    manifest=dict(seed=seed,checkpoint_round=ck['round'],events_preserved=len(events),
        gzip_footer_missing=not dec.eof,discarded_events=0,counts=counts,
        checkpoint_sha256=digest(dest/'checkpoint.pkl'),history_sha256=digest(dest/'history.json'),
        interrupted_log_sha256=hashlib.sha256(raw).hexdigest(),
        elapsed_active_s=state['wall_s'],visible_cache_and_labels_match=True,
        initial_checkpoint_predictions_match=True,
        warning='External session shutdown. In-memory unfinished computations were not checkpointed; no new observations persisted beyond completed round. Resume setup separately timed.')
    archive.write_bytes(raw)
    with gzip.open(dest/'actions.jsonl.gz','wb') as f:f.write(text)
    write(dest/'resume_manifest.json',manifest)
    return manifest

def bootstrap(dest,log,ledger):
    meta=read(dest/'resume_manifest.json')
    with (dest/'checkpoint.pkl').open('rb') as f:ck=pickle.load(f)
    assert digest(dest/'checkpoint.pkl')==meta['checkpoint_sha256']
    ledger.train=ck['ledger']['train'];ledger.validation=ck['ledger']['validation']
    ledger.event_id=meta['events_preserved'];ledger.round=ck['round'];ledger.phase='train'
    for key,value in meta['counts'].items():setattr(ledger,key,value)
    return ck,read(dest/'history.json'),meta['elapsed_active_s']

def worker(seed):
    setup=time.perf_counter();verify_freeze()
    source=(ROOT/'runner_base.py').read_text(encoding='utf-8')
    source=source.replace('log=gzip.open(dest/"actions.jsonl.gz","wt",encoding="utf-8")',
                          'log=gzip.open(dest/"actions.jsonl.gz","at",encoding="utf-8")')
    anchor='    last_predictions=None\n'
    addition='''    ck, history, prior_wall = bootstrap(dest,log,ledger)
    selector=ck['selector'];train_cache=ck['train_cache'];val_cache=ck['val_cache']
    round_no=ck['round'];cap=ck['cap'];last_predictions=ck['predictions']
    started=time.perf_counter()-prior_wall
    resume_setup=time.perf_counter()-resume_setup_started
'''
    assert source.count(anchor)==1;source=source.replace(anchor,addition)
    init='''        for i in p["initial_rows"][str(seed)]:
            for a in range(shape[1]): env.observe(train_cache,i,a,100,"train")
        selector.synchronize_labels(ledger.event)
        selector.fit(0,deadline)
        last_predictions=selector.predict();save_round(last_predictions,"initial")
'''
    assert source.count(init)==1;source=source.replace(init,'')
    path=ROOT/'recovery_base.py'
    # Generated once by the manager to avoid concurrent source mutation.
    assert path.read_text(encoding='utf-8')==source
    spec=importlib.util.spec_from_file_location('fa01_recovery_base',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.bootstrap=bootstrap;module.resume_setup_started=setup
    dest=OUT/'runs'/f'CHEAP-100_s{seed}'
    original=module.atomic_json
    def isolated(path,data):
        if path.name=='RUN_STATE.json':path=dest/'RUN_STATE.json'
        original(path,data)
    module.atomic_json=isolated
    protocol=read(OUT/'FA01_protocol.json')
    data=dict(np.load(PROJECT/'outputs'/'fa00'/'FA00_data.npz'))
    elapsed=read(dest/'resume_manifest.json')['elapsed_active_s']
    result=module.execute('CHEAP-100',seed,protocol,data,time.perf_counter()+7200-elapsed)
    meta=read(dest/'resume_manifest.json')
    meta.update(resume_completed=True,resume_including_setup_wall_s=time.perf_counter()-setup,
                final_result_sha256=digest(dest/'result.json'))
    write(dest/'resume_manifest.json',meta)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);args=ap.parse_args()
    if args.worker is not None:return worker(args.worker)
    verify_freeze();started=time.perf_counter()
    # Generate through the exact same transformation used by each worker.
    source=(ROOT/'runner_base.py').read_text(encoding='utf-8')
    source=source.replace('log=gzip.open(dest/"actions.jsonl.gz","wt",encoding="utf-8")','log=gzip.open(dest/"actions.jsonl.gz","at",encoding="utf-8")')
    source=source.replace('    last_predictions=None\n',"    ck, history, prior_wall = bootstrap(dest,log,ledger)\n    selector=ck['selector'];train_cache=ck['train_cache'];val_cache=ck['val_cache']\n    round_no=ck['round'];cap=ck['cap'];last_predictions=ck['predictions']\n    started=time.perf_counter()-prior_wall\n    resume_setup=time.perf_counter()-resume_setup_started\n")
    source=source.replace('        for i in p["initial_rows"][str(seed)]:\n            for a in range(shape[1]): env.observe(train_cache,i,a,100,"train")\n        selector.synchronize_labels(ledger.event)\n        selector.fit(0,deadline)\n        last_predictions=selector.predict();save_round(last_predictions,"initial")\n','')
    (ROOT/'recovery_base.py').write_text(source,encoding='utf-8')
    manifests=[prepare(s) for s in [7,42,99]]
    batch=read(OUT/'FA01_batch.json');assert batch['completed']==3
    def launch(seed):
        d=OUT/'runs'/f'CHEAP-100_s{seed}'
        with (d/'resume_console.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            code=proc.wait(timeout=7200)
        assert code==0,(seed,code)
        result=read(d/'result.json');result['process_exit_code']=code
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(launch,s) for s in [7,42,99]]
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result();batch['results'].append(r);batch['completed']+=1
            batch['parallel_wall_s']=max(m['elapsed_active_s'] for m in manifests)+time.perf_counter()-started
            batch['resumed_after_external_shutdown']=True
            write(OUT/'FA01_batch.json',batch)
            print(r['arm'],r['seed'],r['status'],r['wall_s'],flush=True)
    write(OUT/'FA01_recovery.json',dict(manifests=manifests,resume_parallel_wall_s=time.perf_counter()-started,
        formal_trajectories_unchanged=6,discarded_observations=0,policies_unchanged=True))
    write(OUT/'RUN_STATE.json',dict(status='complete',completed=6,target=6,parallel_wall_s=batch['parallel_wall_s'],
        resumed_after_external_shutdown=True,unfinished_process=False))

if __name__=='__main__':main()
