"""Preserve baseline training, prospective protocol and completed comparison without raw dataset duplication."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['64','65'],required=True);p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args()
 out=ROOT/('outputs/grid'+a.stage);dest=ARCH/'protocols'/('GRID'+a.stage+'_'+a.phase);assert not dest.exists()
 if a.stage=='64':assert a.phase=='complete' and read(out/'learning_qualification.json')['passed']
 elif a.phase=='before':assert read(out/'preflight.json')['passed'] and not (out/'manifest.json').exists()
 else:assert read(out/'review.json')['passed_saved_data']
 old={}
 for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
  content=read(m)
  for e in (content if isinstance(content,list) else content.get('files',[])):
   f=m.parent/e['path']
   if not f.exists():f=ARCH/e['path']
   assert f.exists() and sha(f)==e['sha256'],str(f)
   old[str(f)]=e['sha256']
 files=list((ROOT/('work/grid'+a.stage)).glob('*.py'))+[Path(__file__).resolve()]
 files += [f for f in out.iterdir() if f.is_file() and f.name!='records.json' and f.suffix in ['.json','.csv','.md']]
 if a.stage=='64':files += [f for f in (out/'models').rglob('*') if f.is_file()]
 if a.stage=='65' and a.phase=='complete':
  for folder in (out/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json',folder/'components.json']
  files += list((ROOT/'work/grid65_review').glob('*.py'))
  files += [ROOT/'论文协作资料/自适应停止对照_GRID64/训练与适配说明.md']
 entries=[]
 for f in sorted(set(files)):
  q=dest/f.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q);assert sha(f)==sha(q)
  entries.append({'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size})
 local=[]
 if a.stage=='64':local=[{'path':'outputs/grid64/records.json','sha256':sha(out/'records.json'),'bytes':(out/'records.json').stat().st_size,'reason':'Derived full feedback retained on D; do not duplicate large historical datasets in Git.'}]
 (dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'local_only':local,'scope':'Explicit ICAPS2022 stopping-only domain adaptation; all cost trials/seeds preserved, no full-task reproduction claim. Original third-party repository is pinned but not redistributed.'},ensure_ascii=False,indent=2),encoding='utf-8')
 if a.stage=='64':
  t=read(out/'training_finished.json');q=read(out/'learning_qualification.json')
  note=f"GRID64论文来源自适应停止基线六次固定拟合完成，CPU墙钟{t['training_wall_s']:.2f}s。排除全部七个评价日期家族，14训练/4校准家族、129/22状态；无新教师仿真，旧已付费反馈151状态/132570次源查询复用。ICAPS2022官方Double DQN核心Python移植，停止/继续限定，q为已观测一步N1奖励/2，不是全周价值。三个预定时间价格用校准集共享选择cost{t['selected']['costindex']}，两个种子均满足预定一步校准条件；132项在线/缓存一致与132项隐藏后缀检查通过，全部六份初末权重与较差配置保留。尚无新闭环结果。"
 elif a.phase=='before':note='GRID65冻结70条串行闭环比较：七个已暴露日期、FULL/FRAME/GLOBAL_MATCH/两个META-DQN种子、两次反序计时重复。全部同原控制/排序/缓存；元策略首次加载和每次推理计入控制时间，无在线拟合。先检验成本/存活，再看查询与时间；不改变模型、时间价格、日期或择种子。'
 else:
  t=read(out/'review.json');note='GRID65论文来源自适应停止对照70条与保存数据审计完成，裁决'+t['verdict']+'；'+'；'.join(f"{s['method']}质量{s['quality_dates']}/7、完整周{s['completed_dates']}/7" for s in t['summaries'])+'。FRAME查询少74.092%、控制少13.583%、最坏成本+0.372%；元策略两种子查询少69.277%/63.045%、控制少13.542%/12.079%、最坏成本+2.973%/+3.983%。FRAME的优势主要是质量和查询；相对元策略宏平均控制仅少0.056%/1.743%，汇总秒数对种子0反而多0.634%、对种子1近乎相同，不能宣称稳定整套提速优势。GLOBAL_MATCH质量7/7、查询与FRAME相同、控制少13.787%，同站结构增量仍未建立。35/35物理重复一致、42项老控制质量/查询复现、无电源事件，276冻结哈希与独立汇总复算通过。逐日期、两个种子、两计时重复均保留。新方法是ICAPS2022停止限定适配，不冒充原A*/RRT*或AAAI2024完整方法，不据此宣称SOTA普遍支配；无独立AC重求解。'
 f=REPO/'docs/RESEARCH_STATE.md';text=f.read_text(encoding='utf-8');f.write_text(text.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+text.split('\n',1)[1],encoding='utf-8')
 nxt='1. 完成GRID65冻结70条串行闭环与保存数据审计，计时期间不并行训练或审计。\n2. 对照所有元策略种子的完整周成本、存活、查询及实际控制时间，不按中途结果改模型。\n3. 全部结果完成后再更新论文，旧稿和GRID63反例保留；不新增日期/训练菜单。\n' if a.phase=='before' or a.stage=='64' else '1. 按GRID65实际结果更新论文基线与贡献范围，保留全部较差种子和GRID63等预算对照。\n2. 展示逐日期完整周质量、时间离散度、P95与离线训练成本；不把适配方法表述为完整SOTA复现。\n3. 当前固定比较结束，不自动重训、改时间价格或换日期。\n'
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n工作区D:/Codex/GRL研究。\n\n'+nxt,encoding='utf-8')
 for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
  f=REPO/'docs'/name;f.write_text(f.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID{a.stage} {a.phase}\n\n'+note+'\n',encoding='utf-8')
 f=ARCH/'README.md';f.write_text(f.read_text(encoding='utf-8')+f'\n## GRID{a.stage} {a.phase}\n\n'+note+'\n',encoding='utf-8')
 for f,h in old.items():assert sha(Path(f))==h
 print(json.dumps({'stage':a.stage,'phase':a.phase,'files':len(entries),'historical_preserved':len(old)}),flush=True)
if __name__=='__main__':main()
