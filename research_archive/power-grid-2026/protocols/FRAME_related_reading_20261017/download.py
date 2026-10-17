from pathlib import Path
import concurrent.futures,datetime as dt,hashlib,json,urllib.request
R=Path(__file__).resolve().parents[2];O=R/'outputs/frame_related_reading_20261017';S=O/'sources'
items=[
 dict(id='AAAI2024_StopPlannerTime',title='Stop! Planner Time: Metareasoning for Probabilistic Planning Using Learned Performance Profiles',venue='AAAI2024',url='https://ojs.aaai.org/index.php/AAAI/article/download/29983/31725',landing='https://ojs.aaai.org/index.php/AAAI/article/view/29983'),
 dict(id='IJCAI2018_OnlinePerformance',title='Meta-Level Control of Anytime Algorithms with Online Performance Prediction',venue='IJCAI2018',url='https://www.ijcai.org/proceedings/2018/0208.pdf',landing='https://www.ijcai.org/proceedings/2018/208'),
 dict(id='IJCAI2013_RationalLazy',title='Towards Rational Deployment of Multiple Heuristics in A*',venue='IJCAI2013',url='https://arxiv.org/pdf/1305.5030',landing='https://arxiv.org/abs/1305.5030'),
 dict(id='EnergyAI2023_ManagingGrids',title='Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents',venue='Energy and AI14(2023)100276',url='https://arxiv.org/pdf/2304.00765v2',landing='https://arxiv.org/abs/2304.00765'),
 dict(id='arXiv2026_GibbsPrior',title='Physics Informed Reinforcement Learning with Gibbs Priors for Topology Control in Power Grids',venue='arXiv preprint2604.01830v1;formal venue not verified',url='https://arxiv.org/pdf/2604.01830v1',landing='https://arxiv.org/abs/2604.01830')]
def get(d):
 p=S/(d['id']+'.pdf');d=dict(d)
 try:
  if not p.exists():
   req=urllib.request.Request(d['url'],headers={'User-Agent':'Mozilla/5.0 (academic paper reading)'})
   with urllib.request.urlopen(req,timeout=45) as f:b=f.read()
   assert b[:5]==b'%PDF-',(d['id'],b[:30]);p.write_bytes(b)
  b=p.read_bytes();assert b[:5]==b'%PDF-';d.update(downloaded=True,path=p.relative_to(R).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:d.update(downloaded=False,error=repr(e))
 return d
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(get,items))
result.append(dict(id='ESWA2024_Progressive',title='Progressive decision-making framework for power system topology control',venue='Expert Systems with Applications235(2024)121070',landing='https://doi.org/10.1016/j.eswa.2023.121070',downloaded=False,coverage='Publisher abstract/highlights and metadata;full text not obtained. Do not infer detailed experimental counts or system sizes.'))
(O/'source_receipt.json').write_text(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),papers=result,new_training=0,new_simulation=0),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps([dict(id=x['id'],downloaded=x['downloaded'],bytes=x.get('bytes'),error=x.get('error')) for x in result],ensure_ascii=False))
