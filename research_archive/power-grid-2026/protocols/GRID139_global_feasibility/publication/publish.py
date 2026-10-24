"""Report feasible learning subtask without converting search labels into business claims."""
from run import *
def main():
    verify();assert read(O/'review.json')['passed'];assert read(O/'compression_task_summary.json')['passed'];assert not (O/'publication_receipt.json').exists()
    review=read(O/'review.json');sit=read(O/'decision_situations.json');op=read(O/'equivalent_completion_opportunity.json');gate=read(O/'compression_task_summary.json');bs=read(O/'compression_baseline_summary.json');progress=read(O/'matched_progress.json');coverage=read(O/'compression_label_coverage.json')
    lines=['# GRID139：全局分批核查的学习资格','',
        '本轮读取GRID100已生成的40个训练/验证场景、20个日历日期、305次FULL搜索及268356条已支付候选反馈。原243次训练调用/62次验证调用全部保留。没有新训练、模型前向、物理仿真或数据采集，最终8测试场景仍未生成。','',
        '## 1. 需要区分下一批与后面所有批次','',
        '| 原FRAME全局提前停止点 | 停止点数 | 后面仍有更优预测候选 | 下一批就改善 | 更后面才改善 |',
        '|---|---:|---:|---:|---:|']
    for part,name in [('train','训练'),('validation','验证')]:
        r=next(x for x in sit if x['partition']==part and x['scope']=='guard_stops');lines.append(f"| {name} | {r['prefixes']} | {r['remaining_improvement']} | {r['next_improvement']} | {r['delayed_improvement']} |")
    lines += ['',
        '这些是FULL教师物理状态上重执行原搜索的结果，不是实际FRAME轨迹上的新失败。后面预测奖励更高，也不等于整周运行更好。原补查还能恢复部分FULL动作：验证62次调用中，全局阶段与FULL动作一致45次，补查后52次。','',
        '训练全部2999个全局前缀里有585个后续改善，其中374个下一批没有改善；验证789个前缀中132个后续改善，其中85个需要跨过至少一批无改进。只把下一批收益作为停止答案，会忽略这类延迟收益。前缀共享调用和日期，不能按行数当独立场景。','',
        '## 2. 平台期信号有信息，但不能直接当充分条件','',
        '原两个无改进批次信号已由FRAME使用。匹配相同已查询数量、候选总数、入口安全分支和当前可接纳/安全状态后，训练102个、验证27个公共条件单元有两侧记录。以重叠样本量加权，平台期组的后续改善比例分别低20.63、12.07个百分点；但单元方向并不一致，验证27单元中8个负向、11个相同、8个反向。全部单元保留，没有从中选择新门槛。','',
        '这提供了超出简单查询进度的描述性关联，不是因果证明，也不证明模型能超过原启发式。信息已有明确作用，新的学习必须证明还能进一步区分同一粗条件中的不同结果。','',
        '## 3. 更早停止后，还要支付同站补查','',
        '| 事后等价参照 | 原FRAME候选查询 | 计入补查后的查询 | 日期宏平均减少 | 与原FRAME动作一致 |',
        '|---|---:|---:|---:|---:|']
    for r in op['summary']:lines.append(f"| {r['partition']} | {r['original_frame_queries']} | {r['masked_hindsight_queries']} | {100*r['date_macro_hindsight_reduction']:.2f}% | {r['exact_original_frame_actions']}/{r['calls']} |")
    lines += ['',
        '参照用完整反馈找到最早已选到原全局阶段动作的128/64检查点，再按原条件执行同站补查，核对最终动作。更早停可能增加站内待查量；训练1909、验证486个额外补查已经扣除。没有可接纳当前动作的调用保持原搜索，不利用事后知道“查完也找不到动作”来虚报节省。','',
        '这是未来信息可见的事后参照，说明固定状态中有可研究的查询空间，不是模型成绩或真实控制时间提速。它也只保持这些FULL教师状态上的原FRAME动作；FULL与FRAME物理路径不同，不能由此推导新策略整周质量。','',
        '## 4. 具体学习子任务草案','',
        '固定为原FRAME全局搜索的计算压缩：在原128起始前缀、每64一批的检查点，决定是否提前转入原条件同站补查。没有可接纳动作时继续；原FRAME强制停止点仍生效，模型不能在它之后延长搜索。','',
        '输入只使用已查询反馈和公开候选信息。监督分别保留“现在停止并按原规则补查能否返回原FRAME动作”和“相对原FRAME候选查询差”；正停止标签要求动作一致且查询严格更少。它是保持原决策的计算子任务，不把动作差异称为业务失败，也不拿原模型的旧失败改口径算成功。','',
        '| 参与草案的检查点 | 前缀 | 物理调用 | 日期 | 可以等价少查 | 需继续以保持原动作 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in gate['summary']:lines.append(f"| {r['partition']} | {r['eligible_prefixes']} | {r['calls']} | {r['dates']} | {r['positive_stop_labels']} | {r['continue_labels']} |")
    neg=[r for r in coverage if not r['stop_label']]
    lines += ['',f"需继续的训练记录覆盖{neg[0]['calls']}次调用/{neg[0]['dates']}个日期，验证覆盖{neg[1]['calls']}次调用/{neg[1]['dates']}个日期。它们是动作保持标签，不是新增整周成本失败；与GRID137的两个业务失败不是同一概念。当前已有正负答案可做一次受控监督学习资格，尚无模型结果。",'',
        '## 5. 简单对照也必须带相同补查','',
        '以下四个参照都保留没有可接纳动作继续、原FRAME强制停止，以及提前停止后的同站补查；cap表示尽量到达的全局检查点，并非强迫覆盖原停止点。全部参照保留，不按验证结果选一个新预算。','',
        '| 分区 | 提前停止参照 | 与原FRAME动作不同 | 候选查询减少宏平均 |','|---|---:|---:|---:|']
    for r in bs:lines.append(f"| {r['partition']} | {r['cap']} | {r['action_differences']}/{r['calls']} | {100*r['date_macro_query_reduction']:.2f}% |")
    lines += ['',
        '256参照在验证62个固定状态上动作全部一致并少查8.08%，但训练243次仍有10次动作不同。这是必须认真面对的强简单对照；不能把它的验证结果当正式测试或直接部署依据。未来学习需要证明查询/原动作保持的增量，而不是仅报高分类准确率。全部其它参照和逐日期结果也保留。','',
        '先前不带补查的固定128/192/256/512及FULL、原全局阶段、完整FRAME的查询/FULL动作/奖励差均保存在baseline_summary.json，用于搜索层诊断。不同对照组成不能混为一张方法主结果表。','',
        '## 6. 数据与核验边界','',
        '3788个全局前缀比旧局部标签范围更广，包含341个尚无可接纳动作的前缀。缺少当前动作时奖励差定义为空，而不是错误填成零；下一批或未来首次找到可接纳动作另存。原305次搜索均与官方GRID86/70逐查询重执行和独立手写实现一致，268356条可接纳判断复算，3788项隐藏后缀扰动不改变输入。','',
        '输入、答案、分组分别保存在public_inputs.jsonl.gz、targets.jsonl.gz、metadata.jsonl.gz。新草案另存compression_task_targets.jsonl.gz，没有修改GRID100/102、旧模型、费用或任何历史标签。原训练32场景16日期与验证8场景4日期仍然隔离；这些验证记录已经用于开发，不能包装成新的独立测试。','',
        '## 7. 下一步','',
        '可以进入一次小规模监督门控的训练设计，暂不扩大数据或网络。先封存单一模型、训练内部按日期校准、按调用/日期权重、停止阈值与整次搜索评价办法，并在训练前通知用户。主要对照为原FRAME及同补查的简单参照；模型不能读取原FRAME未来动作、未来查询数或FULL终点答案。','',
        '先回答“公开反馈能否在保持原动作的条件下更早停止”；随后需要在实际FRAME状态与闭环中检查质量、真实控制时间、所有日期和种子。逐批奖励预测与整周经济价值的研究仍可保留，但不在本轮同时展开，不因局部指标好就宣布业务成功。']
    (O/'GRID139_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    draft='''# 下一轮训练任务草案：压缩原FRAME全局搜索

## 固定问题

利用当前运行状态和已经获得的搜索反馈，判断能否提前结束原FRAME的全局核查，并转入原条件同站补查；目标是在候选查询更少时保持原FRAME动作。原控制器、候选顺序、合法性、别名、float32比较与原强制停止规则不变。没有可接纳动作时继续原搜索。

这是明确的计算子任务。训练标签不声称原FRAME动作最优，也不把不同动作等同经济损失；最后仍以原整周质量与实际控制时间评价。原学习实验和失败记录不重新改判。

## 现成材料

原训练分区584个可学习前缀，来自222次FULL教师状态/16日期，484个停止正例、100个继续例。验证142个前缀来自56调用/4日期，120个停止正例、22个继续例。重复前缀不是独立样本。继续例训练跨64调用15日期，验证跨16调用4日期。

源记录的下一批预测增益、全部余下预测增益与当前压缩任务标签分别保存。任务输入不含查询后的新反馈、原FRAME将来选择、原运行总查询、日期标识或FULL整周答案。

## 模型与评价设计必须先封存

先做一个简单监督模型的资格，不同时扫多模型、网络规模、预算或费用。模型参数及缺失值处理尚未选择。训练内部拟合/校准按日历日期隔离，沿用已有训练/验证边界；按日期及调用控制重复前缀权重，避免长调用支配学习。

校准与评价必须按整次搜索逐批执行：模型提前停止后，后面的前缀不会再出现。原强制停止规则保留，模型只尝试提前停止。逐行准确率仅作辅助；主诊断看与原FRAME动作不同的调用数、查询变化、逐日期与完整调用结果。阈值及不可接受的动作差异准则要在训练前固定，不能根据验证错例调整。

## 必需参照

原FRAME；带相同条件补查、相同无当前动作继续与原强制停止的便宜参照。128/192/256/512离线结果全部保留，不按验证选预算。256参照验证动作62/62一致且候选查询少8.08%，训练仍差10/243，是需要面对的强对照。

## 阶段边界

本草案没有启动训练、神经网络、RL、采集或新物理回放。训练开始前通知用户并给出确切配置。离线通过只表示值得做下一步实际FRAME状态验证，不授权扩大规模、使用最终8测试或发布论文成功结论。
'''
    (O/'下一轮训练任务草案.md').write_text(draft,encoding='utf8')
    save('decision.json',dict(decision='GLOBAL_COMPRESSION_SUBTASK_QUALIFIED_FOR_TRAINING_DESIGN',ready_for_training_design=True,ready_to_train=False,ready_to_scale=False,adopted=False,
        reason='Nontrivial equivalent-action query opportunity;both label classes across calendar groups;delayed search gains require horizon awareness;strong same-completion cheap references retained.',
        task='Earlier stop only on original global path,retain incumbent mask/original forced stop/conditional completion;supervision preserves originalFRAME action and saves candidate queries.',
        train_prefixes=584,validation_prefixes=142,train_continue_dates=15,validation_continue_dates=4,
        training_started=False,model_configuration_frozen=False,new_collection_required_for_first_offline_fit=False,new_physical=0,final_tests_absent=True))
    verify();save('publication_receipt.json',dict(passed=True,main_review=sha(O/'review.json'),completion_review=sha(O/'equivalent_completion_opportunity.json'),task_review=sha(O/'compression_task_summary.json'),
        originals_unchanged=True,new_fits=0,new_model_forwards=0,new_physical=0,final_tests_absent=True,all_references_retained=True,
        service_interruption='Previous turn automatic approval service hit usage limit before script creation;normal retry after user continue succeeded. No analysis action executed in interrupted attempt.'))
    collab=R/'论文协作资料/FRAME_全局核查学习资格_GRID139';collab.mkdir(parents=True,exist_ok=False)
    names=['GRID139_report.md','下一轮训练任务草案.md','decision_situations.json','baseline_summary.json','compression_baseline_summary.json','compression_task_summary.json','compression_label_coverage.json','review.json','decision.json','publication_receipt.json']
    for n in names:shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    save('collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
    d=A/'publication';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    shutil.copy2(REPO/'docs/NEXT_STEPS.md',d/'NEXT_STEPS_before_GRID139.md')
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
    entry='''

## 2026-10-24 — GRID139全局分批核查学习资格完成

只读原40训练/验证场景、20日期、305FULL调用268356反馈，独立重执行官方搜索305次、3788隐藏后缀核验通过。全局前缀2999训练/789验证，包括341无当前动作，奖励差空值和首次找到可用动作分开。原全局停止后训练72/222、验证17/56仍有更优预测候选；其中42/11不在下一批改善，不能用短视下一批标签替代停止价值。控制查询数量/候选总数与原分支状态后，旧平台信号仍有描述性关联但验证27匹配单元方向8负11同8反，不选新阈值、不宣称学习优势。事后最早同原全局动作并计入额外补查，保留无当前动作继续掩码，原FRAME查询训练92799→66804、验证23988→17434，日期宏平均34.38%/30.60%，全部305固定状态原动作一致；这是未来信息参照，不是模型或整周提速。具体压缩任务草案：原强制停止保留，仅尝试更早转原补查；584训练前缀484停100继续，142验证120停22继续，继续标签训练64调用15日期/验证16调用4日期。标签为原动作保持+查询差，不是新增经济失败；旧标签与模型原判读不变。公平简单对照保留同补查与原强制停止：256参照验证62/62原动作一致少查8.08%，训练差10/243；全部128/192/256/512结果保留，不按验证挑预算。具备一次监督训练设计材料，模型/校准/阈值尚未封存，开始训练前通知，不自动拟合/扩规模/采集。最终8未生成，无新物理/模型前向/论文修改。报告research_archive/power-grid-2026/protocols/GRID139_global_feasibility/publication/GRID139_report.md
'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(old+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 全局搜索压缩的训练设计

1. GRID139完成40既有场景305FULL状态/3788全局前缀的搜索资格，下一批与延迟收益分开。事后同原FRAME动作的查询压缩空间训练34.38%/验证30.60%，不是模型结果、整周质量或真实时间。
2. 已另存训练任务草案：模型仅在原全局搜索路径上提前停止，再按原条件补查；原强制停止、无当前动作继续掩码保留。584训练与142验证可学习前缀分别含100/22继续标签，跨15/4日期；不是新增业务失败，不复制整段价值给下一批。
3. 下一步先封存单一简单模型、按日期拟合/校准、按调用/日期权重、停止阈值与整次搜索评价，在开始训练前通知用户。当前模型/参数未选，未训练。不扩大采集或直接进入RL。
4. 强参照必须带相同补查与停止约束。256参照验证62/62原动作一致少查8.08%，训练10次动作差异；全部128/192/256/512结果保留，不按验证挑预算。学习须证明增量，不能用多数类准确率代替搜索结果。
5. 原FULL教师与实际FRAME路径不同，离线通过后还需原FRAME状态与闭环质量/实际计时；FULL周成本/未来动作/未来查询只作标签，不进在线特征。最终8封存；原104/127/135失败与旧费用/模型/阈值不变，不恢复自动化，不改论文。
''',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID139全局分批学习资格](protocols/GRID139_global_feasibility/publication/GRID139_report.md)：305既有FULL状态、3788前缀；延迟收益与同原动作压缩任务草案，强简单对照齐备，零训练/仿真。\n',encoding='utf8')
    files=0
    for root in A.iterdir():
        if root.is_dir():
            for p,h in read(root/'archive_manifest.json').items():assert sha(root/p)==h;files+=1
    verify();print('GRID139_PUBLICATION_VERIFIED',files,len(names))
if __name__=='__main__':main()
