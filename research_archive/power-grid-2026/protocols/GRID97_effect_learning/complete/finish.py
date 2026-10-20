"""Read audited fixed predictions, publish compact results and retained false cases."""
from common import *
import csv,shutil
NAMES={'LOGISTIC':'逻辑回归','FOREST64':'浅层随机森林','TABPFN_V2':'TabPFNv2','ALWAYS_RETAIN':'始终保留补查','ALWAYS_OMIT':'始终视为冗余','TRAIN_PRIOR':'训练集类别比例'}
def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def result_table(results,scheme,field):
    lines=['| 方法 | 例数 | AUC | 平衡准确率 | Brier | 0.5：建议取消/误判轨迹改变 | 0.95：建议取消/误判轨迹改变 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if r['scheme']!=scheme:continue
        m=r[field];a=m['thresholds']['0.5'];b=m['thresholds']['0.95']
        lines.append(f"| {NAMES[r['model']]} | {m['n']} | {fmt(m['auc'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['brier'])} | {a['proposed_omissions']}/{a['changed_path_misclassified']} | {b['proposed_omissions']}/{b['changed_path_misclassified']} |")
    return '\n'.join(lines)
def main():
    assert read(OUT/'review.json')['passed'];verify_freeze()
    d=read(OUT/'design.json');rr=read(OUT/'dataset.json');results=read(OUT/'results.json');pred=read(OUT/'predictions.json');lookup={r['case_id']:r for r in rr}
    cols=['model','scheme','fold','case_id','calendar_date','cohort','pilot_known','y','p_same_path','original_label','both_complete']
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(pred)
    errors=[]
    for p in pred:
        if p['model'] not in d['models']:continue
        for threshold in [.5,.95]:
            if p['y']==0 and p['p_same_path']>=threshold:
                z=lookup[p['case_id']];errors.append(dict(**p,threshold=threshold,original_outcome=z['original_outcome']))
    save('false_omissions.json',errors)
    main=[r for r in results if r['scheme']=='calendar_loo' and r['model'] in d['models']]
    best=max(main,key=lambda r:r['primary_nonpilot']['auc'])
    # This readout selects no new model, threshold, or physical experiment.
    decision=dict(utc=utc(),status='offline_comparison_completed',target='trajectory_effect_not_economic_value',n=56,primary_nonpilot=53,new_physical_replays=0,gradient_training_tabpfn=False,
        best_observed_auc_descriptive=dict(model=best['model'],auc=best['primary_nonpilot']['auc']),
        learned_gate_deployment=False,closed_loop_savings_established=False,thresholds_selected_from_results=False,
        conclusion='本轮未观察到日期留出下可靠的冗余识别能力；三方法主要AUC约为0.500/0.498/0.381，均无可用取消门控证据。TabPFN没有改善预测且单例计算更重。该结论只针对现有56例、15特征与冻结配置，不能外推所有学习方法或补查价值不可预测。')
    save('decision.json',decision)
    text='# GRID97：冗余补查能否从补查前状态识别？\n\n'
    text+='本轮已完成逻辑回归、浅层随机森林与本地预训练TabPFNv2的固定离线比较。主要53例的AUC分别为0.500、0.498、0.381，未观察到可靠的冗余识别能力。TabPFN没有改善预测且推理更重。未增加物理回放，未部署学习门控，未改变FRAME。\n\n'
    text+='## 1. 研究问题与样本\n\n判断“取消这一次同站补查后，完整物理轨迹是否保持不变”。56例中39例轨迹相同、17例轨迹改变；前者包括30例双方完成整周及9例双方在相同位置提前结束。轨迹改变的17例仍保留原来5 SKIP、1 KEEP、11 MIXED的经济/风险记录，没有把它们全当成坏补查。唯一KEEP是旧试点。\n\n'
    text+='这些是已暴露的21个场景条目、17个日历日期中的机制样本，并非新的独立测试集。沿用原18折：17个日历日期留出，G1/G2相同日期一起留出；另有old13训练、G1/G2评价的迁移诊断。全部56例参与相应训练，主要成绩扣除3个已知试点，只对53个非试点的留出预测计分；全部56例与试点成绩另外公开。不能将同一周多个状态当作独立场景。\n\n'
    text+='## 2. 固定模型与公平条件\n\n三者只用相同15个补查前特征及相同训练/测试ID；不输入日期、场景编号、试点身份、动作标签或事后结果。每个训练状态等权，无类重采样；日期宏平均另外报告。为统一TabPFN无sample_weight接口的条件，本轮LR/RF也采用等例权重，与GRID96的日期加权三分类任务不同，不能将两轮成绩直接归因于模型变化。\n\n'
    text+='逻辑回归C=1、lbfgs、最多2000次迭代，仅用训练折均值/标准差标准化；随机森林64棵树、深度4、叶节点至少3例；TabPFN2.2.1使用官方v2默认分类权重、4个集成视图、fit_with_cache，CPU单线程，随机种子20261020。三者参数、权重SHA、样本和划分均在正式预测前封存并推送。TabPFN fit为已标注训练上下文及缓存准备，没有梯度更新或电网微调。\n\n'
    text+='预训练权重来源：[官方模型仓库](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/tree/f851f2a3c941544733b712d8c0f96dfae9b28862)，[固定版本软件包](https://pypi.org/project/tabpfn/2.2.1/)。权重SHA256为`'+d['checkpoint']['sha256']+'`，许可文本随归档保存。遥测关闭、离线本地推理，未上传特征或标签。\n\n'
    text+='## 3. 主要成绩：53个非试点\n\n'+result_table(results,'calendar_loo','primary_nonpilot')+'\n\n'
    text+='正类为轨迹相同，AUC衡量排序，平衡准确率同时考虑两类，Brier越低越好。“误判轨迹改变”指建议取消但实际轨迹改变，不能据此断言一定造成经济损害。0.5与0.95是预先固定的两个读数；0.95不是经过校准的安全置信保证。始终视为冗余提供多数类参照，不能仅因普通准确率高就认为学到了机制。\n\n'
    for r in main:
        m=r['primary_nonpilot'];a=m['thresholds']['0.95'];text+=f"- {NAMES[r['model']]}：AUC {m['auc']:.3f}，平衡准确率 {m['balanced_accuracy']:.3f}；0.95下建议取消{a['proposed_omissions']}例，其中{a['changed_path_misclassified']}例实际改变轨迹。日期宏平均准确率{r['date_macro_accuracy']:.3f}。\n"
    text+='\n逻辑回归0.95下的5个建议只有2例真正轨迹不变，另3例为S021、S010、S043，原经济标签分别为MIXED、MIXED、SKIP。因此“误判轨迹改变”也不能直接当成3次经济损害。浅树和TabPFN在这个阈值都不提出取消，覆盖为零，不能叫作已经学会可靠取消。全部取消的多数类准确率为38/53=71.7%，高于三模型的60.4%、62.3%、66.0%，再次说明普通准确率不足以证明有效。\n'
    text+='\n## 4. 全部56例及旧场景到生成场景诊断\n\n全部56例：\n\n'+result_table(results,'calendar_loo','all56_or_transfer')+'\n\nold13训练、G1/G2非试点评价：\n\n'+result_table(results,'old_to_generated','primary_nonpilot')+'\n\n'
    text+='旧试点、三个来源、逐日期、9例共同提前结束的分层结果保存在results.json。false_omissions.json逐例保留所有误判及原始成本、查询、完成和过载结果，包括唯一KEEP的结果。不得以新53例没有KEEP或某阈值漏KEEP为0，声称模型已学会保护关键补查。\n\n'
    text+='## 5. 推理开销\n\n| 方法 | 17折拟合/上下文总秒数 | 单例平均ms | 单例P95 ms | 热批处理折内每例中位ms |\n|---|---:|---:|---:|---:|\n'
    for r in main:
        t=r['timing'];text+=f"| {NAMES[r['model']]} | {fmt(t['fit_context_total_s'])} | {fmt(t['single_row_mean_ms'])} | {fmt(t['single_row_p95_ms'])} | {fmt(t['warm_batch_per_row_median_ms'])} |\n"
    text+='\n全部运行采用CPU单线程；模型顺序按折轮转。拟合/上下文成本包含检查点加载、预处理和缓存准备；下载、安装、Python导入不进入预测计时。每折首次批预测、3次热批预测及每个测试例1次单例预测全部记录。单例与批概率一致性已检查；单例时延更接近在线逐状态使用条件。未计入控制器提取15维特征的额外开销，未将离线批处理吞吐量当在线延迟，也没有从这些读数推断闭环净提速。\n\n'
    text+='## 6. 审计与贡献边界\n\n核对清理清单及GRID96原始标签/特征哈希、18个原划分、108个模型/参照上下文。逻辑回归由导出的标准化参数和系数独立复算，随机森林按导出的树逐节点重走；TabPFN每折用同一冻结权重和训练ID重建上下文复算，未做额外模型选择。全部概率、单例/批处理一致性与指标已复核，旧文件保持不变。这个审计没有重解AC电网。\n\n'
    text+='版本兼容告警全部保留在日志：SciPy对旧sklearn传入的iprint选项给出未识别告警（仅控制输出），逻辑回归全部未达到迭代上限且导出概率独立复算一致。部分诊断日期只有一类，sklearn给出单类指标告警；results.json中的date_macro_balanced_accuracy沿用sklearn的存在类别平均定义，不能作为统一两类识别能力的证据，正文采用整体两类平衡准确率和日期宏平均普通准确率。\n\n'
    text+='本轮回答的是“哪些补查可能不改变轨迹”，不能代替“哪些轨迹改变值得付出计算”。即使识别冗余有效，也只是单次删补查的离线线索：多次取消会改变后续状态分布，必须另行评价才能主张部署收益。17例改变轨迹的数据较少，经济价值仍有11例真实权衡；不通过继续换模型、阈值或补样救结果。\n'
    text+='\n下一步更值得检查的是现有15维特征是否包含“未查同站候选仍可能改善当前动作”的信息。当前结果不足以区分信息缺失、日期分布变化与模型能力限制，不应直接得出学习没有用或继续加大模型就能解决的结论。本轮到此收口，任何新特征、额外标签或闭环学习策略另行确定。\n'
    (OUT/'GRID97_report.md').write_text(text,encoding='utf8')
    save('pipeline_state.json',dict(stage='audited_ready_to_archive',utc=utc(),new_physical_replays=0))
    repo=ROOT/'work/GRL-results-integration';archive=repo/'research_archive/power-grid-2026/protocols/GRID97_effect_learning/complete';archive.mkdir(parents=True,exist_ok=True)
    collab=ROOT/'论文协作资料/FRAME_补查轨迹影响学习_GRID97_20261020';collab.mkdir(parents=True,exist_ok=True)
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix not in ['.log','.tmp']:
            dest=archive/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
    for n in ['GRID97_report.md','results.json','predictions.csv','false_omissions.json','review.json','decision.json']:shutil.copy2(OUT/n,collab/n)
    for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        p=repo/'docs'/name
        with p.open('a',encoding='utf8') as f:
            f.write('\n\n## 2026-10-20 — GRID97 offline completed\n\n')
            for r in main:
                m=r['primary_nonpilot'];z=m['thresholds']['0.95'];f.write(f"{r['model']}: nonpilot53 AUC={m['auc']:.4f}, BA={m['balanced_accuracy']:.4f}; P>=.95 omission {z['proposed_omissions']}, changed-path errors {z['changed_path_misclassified']}. ")
            f.write('全部离线、未增加回放或部署；路径影响不等于经济价值。完整报告 protocols/GRID97_effect_learning/complete/GRID97_report.md。\n')
    (repo/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID97已完成\n\n1. 用户查看GRID97相同样本/日期折的LR、浅树和TabPFNv2比较及误判案例。\n2. 冗余识别与改变轨迹后的经济价值保持分开；不以单次删补查离线成绩声称闭环收益，不自动部署或继续换模型。\n3. 第二电网、比例预算、CCMA和稿件修订仍待授权；不恢复旧自动化。\n',encoding='utf8')
    for p in [repo/'research_archive/power-grid-2026/README.md',repo/'research_archive/decision-studies-2026/README.md']:
        with p.open('a',encoding='utf8') as f:f.write('\n- GRID97补查轨迹影响：相同56例/15特征/日期隔离的LR、浅树、TabPFNv2离线比较，53非试点主成绩，全部误判保留；无新物理回放或部署。见power-grid-2026/protocols/GRID97_effect_learning/complete/GRID97_report.md。\n')
    manifest={p.relative_to(archive).as_posix():sha(p) for p in archive.rglob('*') if p.is_file()}
    save('complete.json',dict(utc=utc(),passed=True,archive=str(archive),collaboration=str(collab),archive_sha256=manifest,review_sha256=sha(OUT/'review.json'),report_sha256=sha(OUT/'GRID97_report.md'),pending='remote push verification'))
    shutil.copy2(OUT/'complete.json',archive/'complete.json')
    print(text)
if __name__=='__main__':main()
