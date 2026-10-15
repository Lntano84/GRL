"""Same-seven-date offline comparison; do not conflate original/new FRAME."""
from pathlib import Path
import csv,hashlib,json,shutil,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid70_sota_review';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
old=read(ROOT/'outputs/grid65/finished.json')['summaries'];review=read(ROOT/'outputs/grid65/review.json')
new=list(csv.DictReader((ROOT/'outputs/grid70/complete_comparison.csv').open(encoding='utf-8-sig')))
weeks={s['week'] for s in old};rows=[r for r in new if r['date'] in weeks and int(r['full_queries'])>0]
assert len(weeks)==7 and len(rows)==6
guard=sum(int(r['new_queries']) for r in rows)
result=dict(same_dates=7,query_dates=6,new_physical_runs=0,new_fits=0,
    guard_queries=guard,guard_quality_dates=sum(r['new_quality']=='True' for r in new if r['date'] in weeks),
    guard_macro_query_reduction_full=statistics.mean(float(r['new_query_saving']) for r in rows),
    original_comparison=review['seed_comparison'],meta=[],
    scope='Same original seven exposed dates, six with actual search. Query totals count one deterministic physical trajectory per date. New FRAME_GUARD has no paired fresh timing result; original timings cannot be transferred. Literature stopping-only power-grid adaptation, not a full original-task SOTA reproduction.')
for method in ['META_DQN0','META_DQN1']:
    q={s['week']:s['n1_forecasts'] for s in old if s['method']==method and s['repeat']==0}
    total=sum(q[r['date']] for r in rows)
    result['meta'].append(dict(method=method,queries=total,guard_pooled_query_change=guard/total-1,
        guard_macro_query_saving=statistics.mean(1-int(r['new_queries'])/q[r['date']] for r in rows)))
(OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 修复版与论文来源自适应停止对照：离线补充解释','',
    '原GRID65同批比较：原版FRAME相对FULL查询宏平均少74.09%、整套控制时间少13.58%；META-DQN两个种子分别查询少69.28%/63.05%、控制少13.54%/12.08%。原版FRAME直接相对两个种子控制时间宏平均仅少0.0555%/1.7432%，汇总秒数对种子0还略多。因此不支持稳定的整套速度优势。',
    '', '最新修复版FRAME_GUARD在同样七日期中质量7/7；对照分别6/7、5/7。六个实际执行搜索的日期，修复版查询总数11618，对照10305/11201，修复版分别多12.741%/3.723%。修复版相对FULL的查询宏平均减少63.458%。宏平均比例和总查询比例分别计算，不相互替代。',
    '', '论文核心卖点可以放在减少昂贵预测计算，但目前实证支持的是相对FULL在当前质量尺度下压缩计算。尚不能写成修复版比成熟自适应停止方法计算更少。与该对照的区别主要体现为当前质量条件是否满足；方法创新性需具体说明反馈核查、退出与结构补全如何分配计算，不用相对FULL的节省量代替新颖性论证。',
    '', '新版本没有同批平衡计时；13日期62.0%的查询减少与上述七日期不是同一分母，不能直接混比。全部为已暴露日期开发结果。本轮无新仿真、训练或调参。']
(OUT/'比较说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
REPO=ROOT/'work/GRL-results-integration';dest=REPO/'research_archive/power-grid-2026/protocols/GRID70_sota_review'
assert not dest.exists();entries=[]
for p in [Path(__file__).resolve(),OUT/'comparison.json',OUT/'比较说明.md']:
    rel=p.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    entries.append(dict(path=rel.as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(dest/'manifest.json').write_text(json.dumps(dict(files=entries,scope=result['scope']),ensure_ascii=False,indent=2),encoding='utf-8')
note='离线同日期复核：原版FRAME与META-DQN整套控制时间接近（直接宏平均少0.056%/1.743%）；修复版FRAME_GUARD查询总数11618，高于META两个种子10305/11201（多12.741%/3.723%），质量7/7对6/7、5/7。修复版62%查询节省的参照是FULL，不能表述为对SOTA计算优势；无新运行/训练，尚无修复版同批计时。'
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-15 — GRID70 same-date SOTA interpretation\n\n'+note+'\n',encoding='utf-8')
p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新解释：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
