"""Publish feature matrices and definitions, no statistical target selection."""
from pathlib import Path
import csv,json,hashlib,shutil,statistics
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid98_feature_enrichment';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvsave(n,rows,fields):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    review=read(OUT/'review.json');assert review['passed'];d=read(OUT/'design.json');rows=read(OUT/'features.json')
    csvsave('features_original15_unchanged.csv',[dict(case_id=r['case_id'],**r['original15']) for r in rows],['case_id']+d['original_features'])
    csvsave('features_enriched35.csv',[dict(case_id=r['case_id'],**r['all35']) for r in rows],['case_id']+d['original_features']+list(d['new_features']))
    stats={}
    for k in rows[0]['all35']:
        values=[r['all35'][k] for r in rows];stats[k]=dict(unique=len(set(values)),minimum=min(values),maximum=max(values),constant=len(set(values))==1,added=k in d['new_features'])
    save('feature_profile_no_labels.json',stats)
    constants=[k for k,s in stats.items() if s['constant']];newconstant=[k for k in constants if k in d['new_features']]
    times=read(OUT/'extraction_timing.json');ms=[r['pure_feature_seconds']*1000 for r in times]
    save('summary.json',dict(states=56,original=15,added=20,total=35,derived_not_new_information=['incumbent_clearance_to_limit','incumbent_clearance_to_safe'],constant_features=constants,new_constant_features=newconstant,pure_feature_mean_ms=statistics.mean(ms),pure_feature_max_ms=max(ms),new_fits=0,new_predictions=0,new_replays=0))
    lines=['# GRID98：补查前特征补充','',
        '56例已全部补齐并核验，原15维保持逐值不变，新增20项，共35项。新增包括9项候选结构、5项同站已查反馈及6项当前状态/余量；其中2项余量只是原rho特征的确定组合，不是新增观测。本轮只准备输入，没有拟合模型、生成预测或重跑电网。','',
        '## 这次补了什么','',
        '原输入主要概括搜索进度。这次把问题落到当前最佳站：剩余动作与当前最佳动作的连接分区相差多大，与已查过的同站动作有多大区别，是否还有未覆盖的分区；同时补充同站已查奖励的分散程度、第一/第二名差距、合法反馈比例，以及该站在当前过载中的位置。','',
        '分区使用设备两两是否接在同一母线的签名，不输入站号或候选编号；交换母线1/2编号后特征不变。这里衡量的是当前可见动作命令所表达的意向连接关系，不是候选仿真得到的未来拓扑或潮流。无显式母线命令的位置沿用当前拓扑，显式断线按断开处理，因此不能把它当作所有重连行为的精确物理预测。','',
        '## 新增特征定义','']
    for name,(group,definition,source) in d['new_features'].items():lines.append(f'- **{name}**（{group}）：{definition}。输入：{source}。')
    lines += ['','## 没有偷看补查答案','',
        '先按原NO_LOCAL策略仅从已经支付的前缀/分批反馈重建补查前状态，再提取特征。每例单独保存contexts/Sxxx.json，提取函数只接收该可见前缀、已知候选命令和当前观测，不接收标签、补查后选择、整周成本或未来反馈。','',
        '56例均通过：删除全部未查反馈、把未查奖励改成9999并把rho改成0.001、母线编号整体交换、拒绝额外注入事后选择字段。删除或篡改未查反馈后，原15维与新增20维均保持完全相同。909个原动作及各例候选端点屏蔽/别名合并重新核对，已查动作哈希逐条一致。20项新增数值还用独立循环实现复算，共1120个数值核验。','',
        '仅构造一次已有Grid2Op向量解码模板，没有调用reset、step或simulate；候选预测计数前后均为0，没有优化器或控制器运行。这里未进行独立AC重求解。','',
        '## 输入质量与边界','',
        '全56例恒定的列：'+('、'.join(constants) if constants else '无')+'。这些列已标注，但没有按标签筛选或悄悄删除；原15维文件仍与上一轮一致。新增列的唯一值数和范围见feature_profile_no_labels.json，该检查没有使用标签。','',
        f'新增特征纯数值函数在这56例上平均耗时{statistics.mean(ms):.3f}ms、最大{max(ms):.3f}ms；这不是完整在线开销，未包含动作重建和输入组装，原始逐例耗时全部保存。没有用这些读数宣称净提速。','',
        '56例及原物理/经济标签、日期划分全部保留。11个MIXED仍不强制二分，39个同轨迹例中9个共同提前结束的记录仍单列。特征是看到GRID97结果后按机制假设补充的探索性设计，不能称为独立验证。35项也不意味着小样本拟合时应该全部直接使用；后续如比较特征组，缩放、去常量或特征选择必须限制在训练日期内部，不按测试成绩挑选。','',
        '**本轮结论是输入已经补齐且可在补查前计算，尚无证据证明预测变好。** 继续学习时应保留原15维对照，使用相同日期划分及固定模型，检验候选结构和同站反馈是否带来增量；本轮没有自动开始这个评价。','',
        '## 文件','',
        '- features_original15_unchanged.csv：原输入，逐值不变。',
        '- features_enriched35.csv / features.json：新输入；case_id只用于关联，不作为模型输入。',
        '- labels_and_groups_unchanged.json：原答案与分组，独立保存。',
        '- original_folds_unchanged.json：原18个日期评价划分。',
        '- contexts/、provenance.json、review.json：可见输入、来源及核验记录。',
        '- design.json、freeze.json：提取前封存的定义、源码与输入哈希。','']
    (OUT/'GRID98_report.md').write_text('\n'.join(lines),encoding='utf8')
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID98_feature_enrichment/complete';archive.mkdir(parents=True,exist_ok=False)
    collab=ROOT/'论文协作资料/FRAME_补查特征补充_GRID98_20261020';collab.mkdir(parents=True,exist_ok=True)
    for p in OUT.rglob('*'):
        if p.is_file() and p.name!='complete.json':dest=archive/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
    for n in ['GRID98_report.md','features_enriched35.csv','features_original15_unchanged.csv','summary.json','review.json','labels_and_groups_unchanged.json','original_folds_unchanged.json']:shutil.copy2(OUT/n,collab/n)
    note='GRID98按用户要求补齐56例补查前输入：原15维逐值不变，新增20项（9候选结构、5同站反馈、6当前状态/余量，其中2为确定组合），共35维。56隐藏反馈删除/扰动及母线交换、1120独立数值核验通过，原标签和18折不变；无新预测、拟合、回放或部署。尚不声称预测改善。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n## 2026-10-20 — GRID98 feature enrichment completed\n\n'+note+'\n')
    shutil.copy2(REPO/'docs/NEXT_STEPS.md',REPO/'docs/history/NEXT_STEPS_before_grid98_20261020.md')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID98特征已补齐\n\n1. 先检查GRID98新增候选结构、同站反馈与当前状态特征定义及输入质量。\n2. 后续若学习，保留原15维对照、相同日期划分与固定模型；任何缩放或去常量只使用训练日期。当前没有自动拟合。\n3. 冗余识别与经济价值分开，不部署门控或增加回放，不恢复旧自动化。\n4. 第二电网、比例预算、CCMA及稿件修改仍待授权。\n',encoding='utf8')
    for p in [REPO/'research_archive/power-grid-2026/README.md',REPO/'research_archive/decision-studies-2026/README.md']:
        with p.open('a',encoding='utf8') as f:f.write('\n- GRID98：56例补查前特征由15维保留扩为35维，候选结构/同站反馈/当前状态；隐藏反馈隔离与1120独立复算通过，无新拟合或回放。见GRID98_feature_enrichment/complete/GRID98_report.md。\n')
    manifest={p.relative_to(archive).as_posix():sha(p) for p in archive.rglob('*') if p.is_file()}
    save('complete.json',dict(passed=True,archive=str(archive),collaboration=str(collab),archive_sha256=manifest,report_sha256=sha(OUT/'GRID98_report.md'),no_new_fits=True,no_new_predictions=True,no_new_replays=True,publication_receipt='final_verification.json recorded locally after push'))
    shutil.copy2(OUT/'complete.json',archive/'complete.json');shutil.copy2(OUT/'complete.json',collab/'complete.json')
    print(note);print('constant features',constants);print('new constants',newconstant)
if __name__=='__main__':main()
