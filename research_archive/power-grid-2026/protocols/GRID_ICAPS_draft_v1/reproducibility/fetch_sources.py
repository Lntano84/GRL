"""Fetch public writing-source metadata and official formatting kit; never execute it."""
from pathlib import Path
import hashlib, json, urllib.request, zipfile, io
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/grid_icaps_draft/reference_cache'
OUT.mkdir(parents=True, exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'GRL-paper-source-check/1.0'})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read(),r.geturl(),r.headers.get('Content-Type')
records=[]
for name,url in [
 ('writing_repo_readme.md','https://raw.githubusercontent.com/LAMDA-NeSy/Research-Starter-Kit/main/README.md'),
 ('writing_repo_commit.json','https://api.github.com/repos/LAMDA-NeSy/Research-Starter-Kit/commits/main'),
 ('aaai27_author_kit.zip','https://aaai.org/authorkit27/'),
]:
    try:
        data,final,ct=get(url)
        (OUT/name).write_bytes(data)
        rec=dict(name=name,url=url,final_url=final,content_type=ct,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),status='fetched')
        if name.endswith('.zip'):
            z=zipfile.ZipFile(io.BytesIO(data)); rec['members']=z.namelist()
            dest=(OUT/'aaai27').resolve();dest.mkdir(exist_ok=True)
            for m in z.infolist():
                target=(dest/m.filename).resolve()
                if not target.is_relative_to(dest): raise ValueError('Unsafe archive member')
                if not m.is_dir():
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(m))
        records.append(rec)
    except Exception as e:
        records.append(dict(name=name,url=url,status='not_fetched',error=repr(e)))
(OUT/'retrieval.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k!='members'} for r in records],ensure_ascii=False))
for r in records:
    if 'members' in r: print('\n'.join(r['members']))
