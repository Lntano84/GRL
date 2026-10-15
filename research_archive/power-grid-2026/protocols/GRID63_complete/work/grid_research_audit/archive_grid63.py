"""Archive GRID63 lightweight reproducibility receipts, preserving historical files."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026'
OUT=ROOT/'outputs/grid63'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 R=read(OUT/'review.json');assert R['passed_saved_data'] and R['runs']==63
 assert read(OUT/'recovery_finished.json')['passed']
 dest=ARCH/'protocols/GRID63_complete';assert not dest.exists(),'Existing archive must not be repeated.'
 old={}
 for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
  content=read(m)
  for e in (content if isinstance(content,list) else content.get('files',[])):
   p=m.parent/e['path']
   if not p.exists():p=ARCH/e['path']
   assert p.exists() and sha(p)==e['sha256'],str(p)
   old[str(p)]=e['sha256']
 files=list((ROOT/'work/grid63').glob('*.py'))+list((ROOT/'work/grid63_recovery').glob('*.py'))+[Path(__file__).resolve()]
 files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
 for folder in (OUT/'runs').iterdir():
  files += [folder/'summary.json',folder/'metadata.json',folder/'components.json']
 files += [p for p in (OUT/'interruptions').rglob('*') if p.is_file()]
 entries=[]
 for p in sorted(set(files)):
  q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
  assert sha(p)==sha(q)
  entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
 (dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'scope':'63 serial closed-loop trials on seven exposed dates; equal per-state extra budget; no new training. Full vectors remain local. Interrupted child and recovery retained.'},ensure_ascii=False,indent=2),encoding='utf-8')
 note='GRID63同状态等额补查闭环比较完成：7个已有日期、FULL/FRAME/GLOBAL_MATCH、3次平衡计时，共63条；裁决'+R['verdict']+'。'
 note += '；'.join(f"{s['method']}质量{s['quality_dates']}/7、完整周{s['completed_dates']}/7、控制减少{100*s['macro_time_reduction_full']:.3f}%、查询减少{100*s['macro_query_reduction_full']:.3f}%" for s in R['summaries'])+'。'
 note += 'FRAME相对GLOBAL仅三月+0.04677% FULL、四月+0.00782% FULL，其余完整日期成本相同；没有日期达到0.1%尺度。不能把旧NO_LOCAL消融单独解释为同站结构选择的实质优势。'
 note += '同状态配额在新增反馈前确定，轨迹分叉后全周查询总量不强行等量。三次重复用于计时，不增加独立日期。44条后外部会话中断，未完成片段封存；仅续跑未完成条目。保存数据审计，不是独立AC重求解，无新训练/RL。'
 post=read(OUT/'postrun_reconciliation.json')
 if post['successful_run_timing_qualified']:note+='全区间原审计时间资格false保留；逐worker复核6条电源事件均位于中断空档，未重叠63条成功回放，不扣减任何原时间。'
 elif not R['timing_qualified']:note+='运行区间存在电源事件，计时资格有待按日志界定，不作无条件提速主张。'
 p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — budget-matched framework comparison\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
 nxt='1. 依据GRID63等预算补查结果更新论文的结构贡献主张，保留旧V4和全部反例。\n2. 用逐日期成本、查询与三次计时离散度/P95呈现结果；不把63条当63个独立场景。\n3. 当前没有授权新训练、日期扩张或参数搜索；先向作者汇报本轮核心判读。\n'
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n工作区D:/Codex/GRL研究。\n\n'+nxt,encoding='utf-8')
 for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-15 — GRID63 budget-matched completion\n\n'+note+'\n',encoding='utf-8')
 p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID63 complete\n\n'+note+'\n',encoding='utf-8')
 for p,h in old.items():assert sha(Path(p))==h
 print(json.dumps({'files':len(entries),'historical_preserved':len(old),'verdict':R['verdict']},ensure_ascii=False))
if __name__=='__main__':main()
