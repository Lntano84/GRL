"""Append-only evidence archives for paper robustness work, retain historical manifests."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[66,67,68],required=True);ap.add_argument('--phase',choices=['before','complete'],required=True);a=ap.parse_args();OUT=ROOT/f'outputs/grid{a.stage}';dest=ARCH/'protocols'/f'GRID{a.stage}_{a.phase}';assert not dest.exists()
 if a.stage==66:assert read(OUT/'metrics_audit.json')['passed']
 elif a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
 else:assert read(OUT/'review.json')['passed_saved_data']
 old={}
 for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
  c=read(m)
  for e in c if isinstance(c,list) else c.get('files',[]):
   p=m.parent/e['path']
   if not p.exists():p=ARCH/e['path']
   assert p.exists() and sha(p)==e['sha256'],str(p);old[str(p)]=e['sha256']
 files=list((ROOT/f'work/grid{66 if a.stage==66 else 67}').glob('*.py'))+[Path(__file__).resolve()]
 files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.csv')]
 if a.stage!=66 and a.phase=='complete':
  for f in (OUT/'runs').iterdir():files += [f/'metadata.json',f/'summary.json',f/'components.json']
  files += [f for f in (ROOT/'work/grid67_review').glob('*') if f.is_file() and f.suffix in ('.py','.md','.json')]
 entries=[]
 for p in sorted(set(files)):
  q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q);entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
 (dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'scope':'New user-authorized robustness and scene extension, no training, preserve old protocols. Raw physical vectors/step logs remain on D; saved-data audit is not independent AC re-solve.'},ensure_ascii=False,indent=2),encoding='utf-8')
 if a.stage==66:note='GRID66离线整理GRID65的70条逐日期成本、计时离散度、全步骤与搜索步骤P95/P99、互斥计算分解。核查旧路径/训练及结果使用记录后确认6个日期家族仅数据检查或未入选候选池，未有本地控制/教师结果；GRID62曾保守地把所有已解压日期也排除。本次明确更正可用池，不改旧裁决；外部作者先验训练覆盖仍未知。'
 elif a.phase=='before':note=f"GRID{a.stage}冻结"+('24条四旧日期单参数敏感性：前缀64/128/256，停滞1/2/3，默认不变，不做交叉网格。' if a.stage==67 else '36条六个本地新日期、FULL/FRAME/GLOBAL_MATCH、两次反序计时。日期不用于参数选择。')+'原控制/排序/缓存/物理环境不变，无训练。'
 else:
  r=read(OUT/'review.json');note=f"GRID{a.stage}完成{r['runs']}条，保存数据审计通过，裁决{r['verdict']}；"+'；'.join(f"{s['method']}质量{s['quality_dates']}/{r['dates']}、完成{s['complete_dates']}/{r['dates']}、查询少{(s['macro_query_reduction'] or 0)*100:.3f}%、控制少{(s['macro_time_reduction'] or 0)*100:.3f}%" for s in r['summaries'])+'。所有不利日期与配置保留；默认不择优替换，无独立AC重求解。'
 for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID{a.stage} {a.phase}\n\n'+note+'\n',encoding='utf-8')
 p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text(t.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+t.split('\n',1)[1],encoding='utf-8')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n工作区 D:/Codex/GRL研究。\n\n1. 按已冻结顺序完成 GRID67/68，计时期间不并行训练或重型审计。\n2. 全部数据核验后更新中文新稿V5与矢量图表；用户修改稿不覆盖。\n3. 以质量、查询和真实控制时间说明框架贡献；等预算全局对照与不利参数/日期保留，不把局部结构优势当已证明。\n',encoding='utf-8')
 p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID{a.stage} {a.phase}\n\n'+note+'\n',encoding='utf-8')
 for p,h in old.items():assert sha(Path(p))==h
 print(json.dumps({'stage':a.stage,'phase':a.phase,'files':len(entries),'preserved':len(old)}),flush=True)
if __name__=='__main__':main()
