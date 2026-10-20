from pathlib import Path
from collections import Counter
import csv,datetime,hashlib,json,shutil
r=Path('D:/Codex/GRL研究');b=r/'outputs/frame_label_cleanup_20261020';src=r/'outputs/grid96_value_learning';repo=r/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):
    with (b/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
a=rows('01_轨迹不变_39例.csv');d=rows('02_轨迹改变_17例.csv');m=rows('03_权衡明细_11例.csv');x=rows('04_补查前特征_56例.csv');mdata=rows('05_标签与分组_56例.csv')
assert (len(a),len(d),len(m),len(x),len(mdata))==(39,17,11,56,56)
ids=lambda rr:{v['case_id'] for v in rr}
assert ids(a).isdisjoint(ids(d)) and ids(a)|ids(d)==ids(x)==ids(mdata)
assert ids(m)<=ids(d) and all(v['original_label']=='MIXED' for v in m)
features=read(src/'design.json')['learning']['features'];assert list(x[0])==['case_id']+features
original={v['case_id']:v for v in read(src/'labels.json')}
for v in x:assert all(float(v[f])==original[v['case_id']]['features'][f] for f in features)
for v in mdata:assert v['original_label']==original[v['case_id']]['label']
trade=Counter()
for v in m:
    c=v['cost_preference'];q=v['all_queries_preference']
    if {c,q}=={'取消更好','保留更好'}:trade['cost_query_tradeoff']+=1
    elif c==q=='保留更好':trade['keep_cost_queries_better_but_some_risk_worse']+=1
    elif c==q=='取消更好':trade['skip_cost_queries_better_but_some_risk_worse']+=1
assert list(sorted(trade.values()))==[1,3,7]
receipt=read(b/'cleanup_review.json');assert all(h(src/n)==v for n,v in receipt['source_sha256'].items());assert all(h(b/n)==v for n,v in receipt['output_sha256'].items())
proof=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),row_counts=[39,17,11,56,56],mixed_breakdown=dict(trade),features_match_original=True,historical_labels_unchanged=True,historical_predictions_unchanged=True,new_fits=0,new_replays=0,new_predictions=0,model_installations=0,weight_downloads=0,
 local_model_sources={n:h(r/n) for n in ['work/grid02/ljn_nn/make_agent.py','work/grid02/ljn_nn/modules/topology_nn_policy.py','work/grid02/ljn_nn/README.MD']})
(b/'final_review.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
index={p.name:h(p) for p in b.iterdir() if p.is_file()};(b/'artifact_manifest.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
dst=repo/'research_archive/power-grid-2026/paper/FRAME_label_cleanup_20261020';coll=r/'论文协作资料/FRAME_补查标签清理_20261020'
for target in [dst,coll]:
    target.mkdir(parents=True,exist_ok=False)
    for p in b.iterdir():
        if p.is_file():shutil.copy2(p,target/p.name)
shutil.copy2(r/'work/frame_label_cleanup_20261020/prepare.py',dst/'prepare.py')
# Preserve the former current-work list before replacing it.
p=repo/'docs/NEXT_STEPS.md';backup=repo/'docs/history/NEXT_STEPS_before_label_cleanup_20261020.md';assert not backup.exists();shutil.copy2(p,backup)
p.write_text('''# Next steps — FRAME补查机制与标签

1. **当前已完成：** 56例补查记录已清理为30例整周同轨迹、9例同轨迹共同提前结束、5例取消占优、1例保留存活优势、11例指标权衡。特征和事后标签单独保存，原标签/预测/成绩不变。入口：research_archive/power-grid-2026/paper/FRAME_label_cleanup_20261020/README_中文.md。
2. **学习后续：** 用户允许后续考虑神经网络和预训练模型，但当前只要求先清理、慢慢研究学习。本轮无新拟合/预测/回放，不自动扩大模型、下载权重或部署门控。预训练候选优先保留TabPFNv2，TabICL为备选；均须我们的标注上下文，不是现成的电网补查判定器。
3. **先澄清的目标：** 冗余查询识别和改变轨迹后的代价判断分开；11例权衡不强制二分。唯一KEEP是旧试点，不能用新标签漏KEEP为0证明模型已学会保护重要补查。
4. **其余后续：** 第二电网、比例预算、CCMA及稿件修订维持待确认，不因清理或候选模型调研而自动启动。旧实验及全部不利结果保留。

前一版任务表保存在docs/history/NEXT_STEPS_before_label_cleanup_20261020.md。GRID96固定模型诊断详见protocols/GRID96_value_learning/complete/学习结果解读_中文.md。
''',encoding='utf-8')
note='用户要求先清理，学习慢慢研究并可考虑预训练网络。56例全部保留，按轨迹/完整性/指标偏好分开，输入特征与未来答案隔离；无原标签或预测更改。11个MIXED中7个成本与查询冲突、3个保留省成本和查询但部分过载指标变差、1个取消省成本和查询但累计过载更高。核查TabPFNv2、TabICL与本地LJN用途；仅形成资料候选，无安装/权重下载/拟合/预测/回放。清理包见paper/FRAME_label_cleanup_20261020。'
for name in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
    p=repo/name;p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n\n## 2026-10-20 — FRAME_LABEL_CLEANUP_complete\n\n'+note+'\n',encoding='utf-8')
p=repo/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');head,tail=s.split('\n',1);p.write_text(head+'\n\n- **FRAME_LABEL_CLEANUP_complete（当前）：** '+note+'\n'+tail,encoding='utf-8')
p=repo/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n\n- [补查标签清理与预训练候选](paper/FRAME_label_cleanup_20261020/README_中文.md)：56例全保留，30/9/5/1/11分组；无新训练、仿真或模型运行。\n',encoding='utf-8')
print(json.dumps(proof,ensure_ascii=False,indent=2))
