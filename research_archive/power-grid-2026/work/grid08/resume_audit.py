"""Verify saved checkpoint resumption and accounting; no environment executions."""
import json,hashlib,zlib,gzip,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid08'
spec=json.loads((OUT/'resume_01.json').read_text());results=[]
for freeze in [spec['freeze'],json.loads((OUT/'v2/code_freeze.json').read_text()),json.loads((OUT/'guard_qualification/code_freeze.json').read_text())]:
    for p,h in freeze.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
for kind,item in spec['plans'].items():
    path=ROOT/item['partial_folder'];archive=ROOT/item['archive']/path.name
    raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress((archive/'steps.jsonl.gz').read_bytes())
    rows=[]
    for line in raw.splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):pass
    assert len(rows)==item['partial_steps_logged']
    actual=[json.loads(x) for x in gzip.open(path/'steps.jsonl.gz','rt',encoding='utf-8')]
    assert len(actual)>=len(rows)
    identical=all(a['before_hash']==b['before_hash'] and a['after_hash']==b['after_hash'] and a['action_hash']==b['action_hash'] for a,b in zip(rows,actual))
    assert identical,(kind,'Resumed episode differs from the interrupted prefix')
    if kind!='guard':
        folder=path.parent;finish=json.loads((folder/'finished.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        assert len(manifest)==16 and finish['physical_steps']==sum(r['steps'] for r in manifest)
        updates=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
        old=[json.loads(x) for x in (ROOT/item['archive']/'updates_original.jsonl').read_text().splitlines()]
        assert updates[:item['completed_updates']]==old[:item['completed_updates']]
        assert len(updates)==sum(math.ceil(r['steps']/256) for r in manifest)
        assert [x['update'] for x in updates]==list(range(1,len(updates)+1))
    results.append({'kind':kind,'identical_interrupted_prefix':identical,'prefix_steps':len(rows),
                    'partial_steps_upper_bound':item['partial_steps_upper_bound'],
                    'note':'Prefix observed to match; no general bitwise reproducibility guarantee for external libraries.'})
report={'passed':True,'results':results,'extra_physical_steps_lower':spec['partial_steps_logged'],
        'extra_physical_steps_upper':spec['partial_steps_upper_bound'],
        'bound_reason':'Each serial runner logs after stepping; at kill at most one physical step may have completed without a row.',
        'scope':'Persisted action/observation hash and update-prefix checks, not independently rerun physics.'}
(OUT/'resume_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
