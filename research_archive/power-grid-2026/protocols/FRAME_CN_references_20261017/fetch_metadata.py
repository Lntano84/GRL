from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
import json, hashlib
from datetime import datetime, timezone

OUT=Path('D:/Codex/GRL研究/outputs/frame_paper_sections_20261017/07_参考文献/sources')
OUT.mkdir(parents=True,exist_ok=True)
jobs=[
 ('bhatia_dblp.bib','https://dblp.org/rec/conf/aips/BhatiaSNZ22.bib'),
 ('budd_dblp.bib','https://dblp.org/rec/conf/aaai/BuddLH24.bib'),
 ('svegliato_dblp.bib','https://dblp.org/rec/conf/ijcai/SvegliatoWZ18.bib'),
 ('hassouna_dblp.bib','https://dblp.org/rec/conf/pkdd/HassounaHLJVSS25.bib'),
 ('lehna_dblp.bib','https://dblp.org/rec/journals/corr/abs-2304-00765.bib'),
 ('tolpin_dblp_search.json','https://dblp.org/search/publ/api?'+urlencode({'q':'Towards Rational Deployment of Multiple Heuristics','format':'json','h':5})),
 ('liu_dblp_search.json','https://dblp.org/search/publ/api?'+urlencode({'q':'Progressive decision-making framework power system topology control','format':'json','h':5})),
 ('zhou_dblp_search.json','https://dblp.org/search/publ/api?'+urlencode({'q':'Power Network Topology Optimization Power Flow Control Deep Reinforcement Learning','format':'json','h':5})),
 ('lehna_crossref.json','https://api.crossref.org/works/10.1016/j.egyai.2023.100276'),
 ('liu_crossref.json','https://api.crossref.org/works/10.1016/j.eswa.2023.121070'),
 ('hassouna_crossref.json','https://api.crossref.org/works/10.1007/978-3-032-06129-4_8'),
 ('zhou_publisher.html','https://xuebao.sjtu.edu.cn/article/2021/1006-2467/1006-2467-55-S2-7.shtml')]

def fetch(job):
 name,url=job
 result={'name':name,'url':url,'utc':datetime.now(timezone.utc).isoformat()}
 try:
  with urlopen(Request(url,headers={'User-Agent':'FRAME-Bibliography-Verification/1.0'}),timeout=30) as f:
   content=f.read(); result['resolved_url']=f.geturl()
  (OUT/name).write_bytes(content)
  result.update(bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),success=True)
 except Exception as e:result.update(success=False,error=repr(e))
 return result

with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,jobs))
(OUT/'fetch_receipt.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in results:
 print(r['name'],r['success'],r.get('error',''))
 if r['success'] and r['name'].endswith('search.json'):
  x=json.loads((OUT/r['name']).read_bytes())
  print(json.dumps(x.get('result',{}).get('hits',{}),ensure_ascii=False)[:5500])
