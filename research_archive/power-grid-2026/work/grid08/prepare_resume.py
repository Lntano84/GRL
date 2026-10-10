"""Prepare separately versioned runners; never alter frozen model/protocol sources."""
import json,hashlib,math,gzip,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
plans={};total_partial_upper=0
def readable_rows(path):
    # A killed gzip writer may lack a trailer. Decode complete JSON lines only.
    raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(path.read_bytes())
    rows=[]
    for line in raw.splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):pass
    assert all(r['step']==i+1 for i,r in enumerate(rows))
    return rows
for kind in ['gnn','mlp']:
    folder=OUT/'v2/train'/kind
    records=json.loads((folder/'manifest.json').read_text())
    count=len(records);e,j=divmod(count-1,4)
    checkpoint=folder/f'epoch{e}_week{j}.pt'
    assert checkpoint.exists() and not (folder/'finished.json').exists()
    complete_paths={ROOT/r['path'] for r in records}
    incomplete=[p for p in folder.iterdir() if p.is_dir() and p not in complete_paths]
    assert len(incomplete)==1
    rows=readable_rows(incomplete[0]/'steps.jsonl.gz')
    upper=len(rows)+1;total_partial_upper+=upper
    # Updates belonging to the incomplete episode are retained in the archive.
    all_updates=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
    complete_updates=sum(math.ceil(r['steps']/256) for r in records)
    assert len(all_updates)>=complete_updates
    plans[kind]={'completed_episodes':count,'checkpoint':str(checkpoint.relative_to(ROOT)),
                 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 'partial_folder':str(incomplete[0].relative_to(ROOT)),
                 'partial_steps_logged':len(rows),'partial_steps_upper_bound':upper,
                 'completed_updates':complete_updates,'completed_physical_steps':sum(r['steps'] for r in records),
                 'completed_optimizer_wall_s':sum(x['wall_s'] for x in all_updates[:complete_updates]),
                 'archive':str((folder/'interrupted_attempt_01').relative_to(ROOT))}
guard=OUT/'guard_qualification';records=json.loads((guard/'manifest.json').read_text())
completed={ROOT/r['path'] for r in records}
partial=[p for p in guard.iterdir() if p.is_dir() and p not in completed]
assert len(partial)==1
rows=readable_rows(partial[0]/'steps.jsonl.gz');total_partial_upper+=len(rows)+1
plans['guard']={'completed_episodes':len(records),'completed_physical_steps':sum(r['steps'] for r in records),
                'partial_folder':str(partial[0].relative_to(ROOT)),
                'partial_steps_logged':len(rows),'partial_steps_upper_bound':len(rows)+1,
                'archive':str((guard/'interrupted_attempt_01').relative_to(ROOT))}
source=(ROOT/'work/grid08/pilot_v2.py').read_text()
source=source.replace("env=make_env(); physical=0; updates=0; opt_wall=0.; summaries=[]", """env=make_env(); physical=0; updates=0; opt_wall=0.; summaries=[]
RESUME=json.loads((OUT/'resume_01.json').read_text())
if args.phase=='train':
    r=RESUME['plans'][args.kind]
    summaries=json.loads((RUN/'manifest.json').read_text())
    assert len(summaries)==r['completed_episodes']
    physical=r['completed_physical_steps'];updates=r['completed_updates'];opt_wall=r['completed_optimizer_wall_s']
    old=[json.loads(x) for x in (ROOT/r['archive']/'updates_original.jsonl').read_text().splitlines()]
    with (RUN/'updates.jsonl').open('w',encoding='utf-8') as f:
        for row in old[:updates]:f.write(json.dumps(row)+'\\n')
""")
old="""        saved=torch.load(OUT/'v1/train'/args.kind/'final.pt',weights_only=False)
        model.load_state_dict(saved['state'])
        with torch.no_grad():model.policy.actor.bias[1]+=D['warm_start']['bias_shifts'][args.kind]
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        write_json(RUN/'warm_start.json',D['warm_start'])
        torch.save({'state':model.state_dict(),'kind':args.kind,'n_nodes':2*env.n_sub,'max_edges':2*env.n_line},RUN/'initial.pt')
        write_json(RUN/'model_metadata.json',{'kind':args.kind,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'device':'cpu'})"""
new="""        r=RESUME['plans'][args.kind]
        checkpoint=ROOT/r['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
        saved=torch.load(checkpoint,weights_only=False)
        assert saved['episode']==len(summaries)
        model.load_state_dict(saved['state'])
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])"""
assert old in source;source=source.replace(old,new)
source=source.replace("                summary=episode(scenario,args.kind,model,optimizer,label=f'e{epoch}_{scenario}')", "                if epoch*len(D['training_weeks'])+j<r['completed_episodes']:continue\n                summary=episode(scenario,args.kind,model,optimizer,label=f'e{epoch}_{scenario}')")
source=source.replace("'wall_s':time.perf_counter()-START,'optimizer_wall_s':opt_wall,'updates':updates}", "'wall_s':time.perf_counter()-START,'wall_scope':'Resumed invocation only; prior completed episode walls remain in manifest; interrupted logs retained.','optimizer_wall_s':opt_wall,'updates':updates}")
runner=ROOT/'work/grid08/pilot_v2_resume.py';runner.write_text(source,encoding='utf-8')
guard_source=(ROOT/'work/grid08/guard_qualification.py').read_text()
begin=guard_source.index("write_json(FOLDER/'design.json',config)")
end=guard_source.index('try:\n',begin)
guard_source=guard_source[:begin]+"""RESUME=json.loads((OUT/'resume_01.json').read_text())['plans']['guard']
records=json.loads((FOLDER/'manifest.json').read_text());physical=sum(r['steps'] for r in records)
assert len(records)==RESUME['completed_episodes']
done={(r['week'],r['rule']) for r in records}
start=time.perf_counter();env=make_env()
"""+guard_source[end:]
guard_source=guard_source.replace("            folder=FOLDER/(week+'__'+rule)", "            if (week,rule) in done:continue\n            folder=FOLDER/(week+'__'+rule)")
guard_source=guard_source.replace("'wall_s':time.perf_counter()-start})", "'wall_s':time.perf_counter()-start,'wall_scope':'Resumed invocation only; interrupted attempt retained.'})")
guard_runner=ROOT/'work/grid08/guard_qualification_resume.py';guard_runner.write_text(guard_source,encoding='utf-8')
record={'cause':'No Python processes remained; tool sessions unknown. No failure marker. External interruption observed; exact cause undetermined.',
        'plans':plans,'partial_steps_logged':sum(x['partial_steps_logged'] for x in plans.values()),
        'partial_steps_upper_bound':total_partial_upper,
        'wall_budget':'Original per-invocation caps retained; interrupted attempts are disclosed in addition. No model or training schedule changes.',
        'rng_scope':'Checkpoint stores model, Adam and Torch RNG. Global NumPy state was not checkpointed; no explicit global NumPy draws in runner. No bitwise continuation claim for external libraries.',
        'freeze':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [runner,guard_runner,Path(__file__)]}}
(OUT/'resume_01.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
