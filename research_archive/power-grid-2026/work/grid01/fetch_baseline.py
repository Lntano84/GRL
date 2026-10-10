import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid01'
SRC=ROOT/'work/grid01/original_ljn'
SHA='ca0637eab9f098be7f206ed0e46a3900cd4deec0'
FILES=['README.MD','LICENSE','AUTHORS.txt','__init__.py','utils.py','make_agent.py','LJNAgent.py',
       'modules/__init__.py','modules/utils.py','modules/rewards.py','modules/base_module.py',
       'modules/topology_heuristic.py','modules/convex_optim.py',
       'assets/action_12_unsafe.npz','assets/action_N1_unsafe.npz']
OUT.mkdir(parents=True,exist_ok=True)
manifest=[]
for name in FILES:
    url=f'https://raw.githubusercontent.com/lajavaness/l2rpn-2023-ljn-agent/{SHA}/{name}'
    dst=SRC/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    # Empty __init__ can be absent; record absence and create only in adapted package.
    try:
        data=urllib.request.urlopen(url,timeout=60).read()
    except urllib.error.HTTPError as e:
        if name.endswith('__init__.py') and e.code==404:
            manifest.append({'path':name,'source':url,'absent':True})
            continue
        raise
    dst.write_bytes(data)
    manifest.append({'path':name,'source':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    print(name,len(data),flush=True)
assert sum(x.get('bytes',0) for x in manifest)<150*1024**2
(OUT/'source_manifest.json').write_text(json.dumps({'commit':SHA,'files':manifest},indent=2),encoding='utf-8')
with urllib.request.urlopen('https://pypi.org/pypi/cvxpy/json',timeout=30) as r:
    meta=json.load(r)
(OUT/'cvxpy_metadata.json').write_text(json.dumps({'version':meta['info']['version'],'requires_python':meta['info']['requires_python'],
    'requires_dist':meta['info']['requires_dist']},indent=2),encoding='utf-8')
print('CVXPY',meta['info']['version'],flush=True)
