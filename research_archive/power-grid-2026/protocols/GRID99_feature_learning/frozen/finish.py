from common import *
import csv,shutil
NAMES={'LOGISTIC':'逻辑回归','FOREST64':'浅层森林','TABPFN_V2':'TabPFNv2'}
def fmt(v):return '—' if v is None else f'{v:.3f}'
def main():
    assert read(OUT/'review.json')['passed'];verify_freeze();d=read(OUT/'design.json');rs=read(OUT/'results.json');changes=read(OUT/'paired_changes.json');pp=read(OUT/'predictions.json')
    def get(arm,name,scheme='calendar_loo'):return next(r for r in rs if r['arm']==arm and r['model']==name and r['scheme']==scheme)
    comparisons=[]
    for name in d['models']:
        a=get('ORIGINAL15',name)['primary_nonpilot'];b=get('ENRICHED35',name)['primary_nonpilot']
        comparisons.append(dict(model=name,original_auc=a['auc'],enriched_auc=b['auc'],original_ba=a['balanced_accuracy'],enriched_ba=b['balanced_accuracy'],original95=a['thresholds']['0.95'],enriched95=b['thresholds']['0.95']))
    save('decision.json',dict(status='fixed_comparison_completed',utc=utc(),comparisons=comparisons,descriptive_only=True,independent_test=False,feature_set_selected_for_deployment=False,learned_gate_deployment=False,new_physical_replays=0,extra_hyperparameter_fits=0))
    fields=['arm','model','scheme','fold','case_id','calendar_date','cohort','pilot_known','y','p_same_path','original_label','both_complete']
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(pp)
    lookup={r['case_id']:r for r in read(OUT/'dataset.json')};errors=[]
    for p in pp:
        if p['arm']=='REUSED_BASELINE':continue
        for threshold in [.5,.95]:
            if p['y']==0 and p['p_same_path']>=threshold:errors.append(dict(**p,threshold=threshold,original_outcome=lookup[p['case_id']]['original_outcome']))
    save('false_omissions.json',errors)
    text='# GRID99：补充特征后的固定学习比较\n\n已完成原15维与补充35维的逻辑回归、浅层森林、TabPFNv2比较，并通过保存结果与概率复核。所有模型及阈值预先固定；没有新电网回放或门控部署。\n\n'
    text+='## 主要结果：53个非试点，按日期留出\n\n| 模型 | 原15维AUC | 补充35维AUC | 原15维平衡准确率 | 补充35维平衡准确率 |\n|---|---:|---:|---:|---:|\n'
    for c in comparisons:text+=f"| {NAMES[c['model']]} | {fmt(c['original_auc'])} | {fmt(c['enriched_auc'])} | {fmt(c['original_ba'])} | {fmt(c['enriched_ba'])} |\n"
    text+='\n正类是“取消这一次补查后完整物理轨迹相同”，不是“补查经济价值为负”。AUC的0.5是无区分能力参照，平衡准确率同时考虑轨迹相同/改变两类。对照结果逐项保留，不根据结果选新特征、模型或门槛。\n\n'
    for c in comparisons:
        z=c['enriched95'];text+=f"- {NAMES[c['model']]}：0.95固定门槛下，补充特征后建议取消{z['proposed_omissions']}例，其中{z['correct_redundancy']}例确实轨迹相同、{z['changed_path_misclassified']}例轨迹改变。\n"
    text+='\n## 实验条件\n\n同一56例、相同15项原输入逐值保留、相同18折（17个日历日期留出及old13训练/G1+G2评价），G1/G2同日历日期不跨训练/测试。39例轨迹相同、17例轨迹改变；原3个已知试点仍进入其他日期训练，只从主要测试成绩扣除，全部56例成绩与试点单列。9例同轨迹共同提前结束仍保留。输入补充发生在已暴露数据和GRID97结果之后，因此是开发诊断，不是独立确认。\n\n'
    text+='逻辑回归C=1、lbfgs、最多2000次迭代；64棵浅树深度4、叶节点至少3例；TabPFN2.2.1使用同一v2默认本地权重、4集成视图、CPU单线程和同一随机种子20261020。三模型均等例权重，逻辑回归只用训练折标准化；全部原常量、重复列保留，不额外搜索特征子集。35列包含2项原特征确定组合及已标注的常量/重复列，不代表35种独立信息。TabPFN没有梯度更新，fit表示训练上下文和缓存准备。\n\n'
    text+='18折×2输入组×3模型，每个上下文只拟合/准备一次，共108个。审计用相同支持集和冻结检查点再建36个TabPFN上下文验证概率，不作为第二次实验或挑选结果。逻辑回归和森林导出系数/树后独立复算；原15维全部折外概率需重现GRID97。每个测试例同时检查单例与批预测一致。没有下载或安装新依赖。\n\n'
    text+='## 全部56例及迁移诊断\n\n| 输入 | 模型 | 全56例AUC | 全56例平衡准确率 | old13到生成22例AUC | old13到生成22例平衡准确率 |\n|---|---|---:|---:|---:|---:|\n'
    for name in d['models']:
        for arm in d['feature_sets']:
            a=get(arm,name)['all56_or_transfer'];b=get(arm,name,'old_to_generated')['primary_nonpilot'];text+=f"| {arm} | {NAMES[name]} | {fmt(a['auc'])} | {fmt(a['balanced_accuracy'])} | {fmt(b['auc'])} | {fmt(b['balanced_accuracy'])} |\n"
    text+='\n试点、来源、逐日期、固定0.5/0.95阈值、准确率、AP、Brier及混淆矩阵全部见results.json；逐状态概率见predictions.csv，所有误判及原成本/查询/存活/过载记录见false_omissions.json。原KEEP只有一个旧试点，不能用主要53例里漏KEEP为0证明模型已经保护了关键补查。不同折训练比例不同，训练比例基线合并后的AUC不必为0.5，不以它的低AUC证明方法优势。\n\n'
    text+='## 同批单例推理时间\n\n| 模型 | 原15维平均ms | 补充35维平均ms | 补充35维P95 ms |\n|---|---:|---:|---:|\n'
    for name in d['models']:
        a=get('ORIGINAL15',name)['timing'];b=get('ENRICHED35',name)['timing'];text+=f"| {NAMES[name]} | {fmt(a['single_row_mean_ms'])} | {fmt(b['single_row_mean_ms'])} | {fmt(b['single_row_p95_ms'])} |\n"
    text+='\n折内输入组顺序交替，模型顺序轮转；全部CPU单线程。每上下文首次批预测、3次热批预测、每测试例一次单例预测均记录。表中不含特征提取成本；GRID98的补充特征纯数值函数平均约13.142ms，且未包含完整动作重建/输入组装，不能直接从分类精度声称闭环净提速。旧GRID97时间只用于历史说明，表内均为本轮实测。\n\n'
    text+='## 解释边界\n\n本轮衡量的是新增输入是否带来可观察的离线增量。轨迹改变可能受益，也可能吃亏；11个MIXED没有被重新二分。即使指标改善，也不能据此保证长期补查价值、跨网泛化或多次取消后的闭环收益；即使不改善，也不能排除其他特征、样本或学习方法的可能性。没有自动进入新模型、特征筛选、回放或部署。\n'
    (OUT/'GRID99_report.md').write_text(text,encoding='utf8')
    save('pipeline_state.json',dict(stage='audited_complete',utc=utc(),completed_contexts=108,new_physical_replays=0))
    repo=ROOT/'work/GRL-results-integration';a=repo/'research_archive/power-grid-2026/protocols/GRID99_feature_learning/complete';a.mkdir(parents=True,exist_ok=False);c=ROOT/'论文协作资料/FRAME_补充特征学习_GRID99_20261020';c.mkdir(parents=True,exist_ok=True)
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix!='.tmp':dest=a/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in W.glob('*.py'):shutil.copy2(p,a/p.name)
    for n in ['GRID99_report.md','results.json','paired_changes.json','predictions.csv','false_omissions.json','decision.json','review.json']:shutil.copy2(OUT/n,c/n)
    summary='；'.join(f"{NAMES[z['model']]} AUC {z['original_auc']:.3f}->{z['enriched_auc']:.3f}, BA {z['original_ba']:.3f}->{z['enriched_ba']:.3f}" for z in comparisons)
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (repo/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n## 2026-10-20 — GRID99 feature learning completed\n\n'+summary+'。53非试点日期留出，108固定上下文，原15维重现，全部不利结果保留；无新物理回放或部署，轨迹影响不是经济价值。\n')
    shutil.copy2(repo/'docs/NEXT_STEPS.md',repo/'docs/history/NEXT_STEPS_before_grid99_20261020.md');(repo/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID99固定学习比较已完成\n\n1. 用户查看原15维与补充35维在相同日期隔离评价中的增量、误判与推理时间。\n2. 不根据开发结果自动选择特征、门槛或模型，不自动部署或新增回放。\n3. 轨迹影响与经济价值保持分开；第二电网、比例预算、CCMA和稿件修改仍待确认。\n',encoding='utf8')
    for p in [repo/'research_archive/power-grid-2026/README.md',repo/'research_archive/decision-studies-2026/README.md']:
        with p.open('a',encoding='utf8') as f:f.write('\n- GRID99：原15维/补充35维、固定LR/浅树/TabPFNv2日期隔离比较，108上下文、原15维重现及误判完整保留，无新物理回放或部署。见GRID99_feature_learning/complete/GRID99_report.md。\n')
    save('complete.json',dict(passed=True,archive=str(a),collaboration=str(c),archive_sha256={p.relative_to(a).as_posix():sha(p) for p in a.rglob('*') if p.is_file()},report_sha256=sha(OUT/'GRID99_report.md'),new_physical_replays=0,publication_receipt='final_verification.json saved locally after remote SHA verification'))
    shutil.copy2(OUT/'complete.json',a/'complete.json');shutil.copy2(OUT/'complete.json',c/'complete.json');print(text)
if __name__=='__main__':main()
