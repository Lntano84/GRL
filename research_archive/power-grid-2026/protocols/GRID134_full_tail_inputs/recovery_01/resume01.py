"""Preserve old64-row pack assertion failure; chunk unchanged encoder for full set."""
from pathlib import Path
import importlib.util,json,hashlib,shutil
import numpy as np
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid134_full_tail_inputs'; W=Path(__file__).parent
spec=importlib.util.spec_from_file_location('grid134_frozen',W/'run.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
m.verify(); assert not (O/'recovery_01.json').exists() and not (O/'complete.json').exists()
attempt=O/'interrupted_attempts/pack_limit_01';attempt.mkdir(parents=True,exist_ok=False)
for name in ['qualification_started.json','qualification.log','train_calibration_public_inputs.jsonl.gz']:
    p=O/name; dest=attempt/name
    assert p.resolve().parent==O.resolve() and dest.resolve().parent==attempt.resolve()
    p.rename(dest)
for p in [W/'run.py',O/'code_freeze.json',O/'source_hashes.json',O/'design.json']:shutil.copy2(p,attempt/p.name)
oldpack=m.pack
def fullpack(c,g,cost,scale):
    # Keep old public descriptor and every candidate. Only concatenate separately
    # encoded <=64 chunks, never pool/truncate or read outcomes from another batch.
    assert set(c)==m.pack_fields and len(c['next_ids'])>0
    chunks=[]
    for start in range(0,len(c['next_ids']),64):
        part=dict(c,next_ids=c['next_ids'][start:start+64],next_plans=c['next_plans'][start:start+64])
        chunks.append(oldpack(part,g,cost,scale))
    out=dict(chunks[0]); out['next_action']=np.concatenate([v['next_action'] for v in chunks])
    out['known_query_cost']=np.array([cost*len(c['next_ids'])/scale],dtype=np.float32)
    if len(chunks)==1:assert all(np.array_equal(v,chunks[0][k]) for k,v in out.items())
    return out
from schema import FIELDS
m.pack_fields=FIELDS; m.pack=fullpack
write(O/'recovery_01.json',dict(reason='Historical GRID112 pack explicitly rejects next sets longerthan64; frozen qualification failed at that assertion, not physical or model failure.',
    frozen_source_unchanged=True,complete_records_before_failure=0,partial_inputs_preserved=True,new_fits=0,
    change='Wrapper uses unchanged old encoder per<=64 chunks then concatenates ALL candidate rows in original order; same public per-action representation, no pooled or lost candidates.',
    recovery_source_sha256=sha(Path(__file__)),original_source_sha256=sha(W/'run.py'),
    preserved={p.name:sha(p) for p in attempt.iterdir() if p.is_file()}))
m.run()
