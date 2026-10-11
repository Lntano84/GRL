"""Quality-first review; timings are single-repeat development estimates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT / "outputs/grid28"
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','all'],default='pilot');phase=parser.parse_args().phase
F=json.loads((OUT / ('pilot_finished.json' if phase=='pilot' else 'finished.json')).read_text())
A=json.loads((OUT / ('pilot_audit.json' if phase=='pilot' else 'audit.json')).read_text());assert A['passed']
D=json.loads((OUT / 'runtime_design.json').read_text())
groups=F['summaries'];index={(s['week'],s['method']):s for s in groups}
rows=[]
for s in groups:
    if s['method']=='PROTECTED_GNN0':continue
    b=index[(s['week'],'PROTECTED_GNN0')]
    same_completion=s['complete']==b['complete']
    earlier_failure=not s['complete'] and (b['complete'] or s['steps']<b['steps'])
    cost_regret=s['cost']/b['cost']-1 if s['complete'] and b['complete'] else None
    matched_duration=s['steps']==b['steps'] and same_completion
    rows.append({'week':s['week'],'method':s['method'],'complete':s['complete'],'steps':s['steps'],
                 'reference_complete':b['complete'],'reference_steps':b['steps'],'earlier_failure':earlier_failure,
                 'cost_regret':cost_regret,'cost':s['cost'],'reference_cost':b['cost'],
                 'exact_full_trajectory':s['exact_full_trajectory'],'first_divergence':s['first_full_divergence'],
                 'n1_queries':s['n1_forecasts'],'reference_n1_queries':b['n1_forecasts'],
                 'query_reduction':1-s['n1_forecasts']/b['n1_forecasts'],
                 'controller_s':s['controller_s'],'reference_controller_s':b['controller_s'],
                 'search_s':s['n1_source_s'],'reference_search_s':b['n1_source_s'],
                 'controller_reduction':1-s['controller_s']/b['controller_s'] if matched_duration else None,
                 'search_reduction':1-s['n1_source_s']/b['n1_source_s'] if matched_duration else None,
                 'illegal':s['illegal'],'ambiguous':s['ambiguous'],'exception_steps':s['exception_steps'],
                 'qualified':not earlier_failure and same_completion and (cost_regret is None or cost_regret<=.005)})
assert len(rows)==(2 if phase=='pilot' else 8)
result={'phase':phase,'rows':rows,'quality_gate_passed':all(r['qualified'] for r in rows),
        'warning':'Same exposed date families. Cost threshold is an investment screen, not equivalence. One repeat and changed paths cannot establish stable timing or a causal GNN increment.',
        'new_fits':0,'reserved_evaluations':0,'complete_runs':len(groups)}
(OUT / (phase+'_comparison.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
with (OUT / (phase+'_comparison.csv')).open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=[f'# GRID28 {phase} 闭环开发结果','',result['warning'],'',
       '| 方法/日期 | 完成/步数 | 成本差 | 查询减少 | 搜索时间减少 | 总控制减少 | 质量筛查 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    fmt=lambda x:'不比较' if x is None else f'{100*x:+.2f}%'
    lines.append(f"| {r['method']}/{r['week']} | {r['complete']}/{r['steps']} | {fmt(r['cost_regret'])} | {fmt(r['query_reduction'])} | {fmt(r['search_reduction'])} | {fmt(r['controller_reduction'])} | {r['qualified']} |")
lines += ['',f"冻结质量筛查整体通过：{result['quality_gate_passed']}。共同失败周不拿较低累计成本作收益。",'',
          '补全规则是所有排序器共有的非学习方法；闭环查询差和单次计时不等于已经证明图模型独有贡献。旧受保护模型/全池守卫仍完整保留。',
          '新拟合/RL/8个封存日期评价均为0。保存数据验解包括掩码、别名、排序复算、付费反馈、独立补查/补全逻辑、连续控制交付和成本账目；未另做AC重放或独立重新训练。']
(OUT / ('GRID28_'+phase+'_report.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT / 'analysis_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT / 'work/grid28/review.py',ROOT / 'work/grid28/audit.py']},indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
