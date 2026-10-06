"""Read a public repository tree and scenario descriptions, never runtime tables."""
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib, json

COMMIT='7a5727651a92fd2fa4960dcbbd7f6dab94130028'
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'work/fa03_recovery/source'
OUT.mkdir(parents=True,exist_ok=True)
def get(url):
    with urlopen(Request(url,headers={'User-Agent':'FA03 metadata qualification'}),timeout=30) as r:
        return r.read()
treeurl=f'https://api.github.com/repos/stacs-cp/JAIR2026-FrugalAS/git/trees/{COMMIT}?recursive=1'
treeblob=get(treeurl)
(OUT/'tree.json').write_bytes(treeblob)
tree=json.loads(treeblob)['tree']
paths=sorted(x['path'] for x in tree if x['path'].startswith('DATASETS/') and x['path'].endswith('/description.txt'))
def archive(path):
    url=f'https://raw.githubusercontent.com/stacs-cp/JAIR2026-FrugalAS/{COMMIT}/{path}'
    blob=get(url);dest=OUT/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
    return dict(path=path,url=url,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
with ThreadPoolExecutor(max_workers=4) as pool:
    records=list(pool.map(archive,paths))
(OUT/'manifest.json').write_text(json.dumps(dict(commit=COMMIT,tree_url=treeurl,tree_sha256=hashlib.sha256(treeblob).hexdigest(),files=records),indent=2),encoding='utf8')
print(json.dumps(dict(scenarios=[p.split('/')[1] for p in paths],runtime_tables_downloaded=0),indent=2))
