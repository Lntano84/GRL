"""Close saved-log investigation and state the evidence-based research priority."""
from run import *
from scope_resume01 import v

def main():
    v();assert read(O/'review.json')['passed'];assert read(O/'historical_timing_summary.json')['passed'];assert not (O/'publication_receipt.json').exists()
    timing=read(O/'timing_opportunity.json');history=read(O/'historical_timing_summary.json');public=read(O/'public_information_summary.json');rr=read(O/'public_feature_outcome_ledger.json')
    base=read(R/'outputs/grid104_learning_closedloop/finished.json')['summaries'];lookup={(s['repeat'],s['week'],s['method']):s for s in base}
    stage=[]
    for r in read(O/'timing_runs.json'):
        if r['method']!='FRAME_GUARD':continue
        s=lookup[(r['repeat'],r['week'],r['method'])];global_s=s['n1_forecast_s']-r['local_s'];noncandidate_s=s['all_forecast_s']-s['n1_forecast_s']
        assert global_s>=0 and noncandidate_s>=0
        parts=dict(local=r['local_s'],global_candidate=global_s,noncandidate_forecast=noncandidate_s,optimizer=r['optimizer_s'],search_overhead=r['search_overhead_s'],other=r['other_s'])
        assert abs(math.fsum(parts.values())-r['controller_s'])<1e-7
        stage.append(dict(date=r['date'],repeat=r['repeat'],complete=r['complete'],seconds=parts,shares={k:t/r['controller_s'] for k,t in parts.items()}))
    # Each date has two repetitions,so equal repetition weights equal date macro weights.
    macro={k:statistics.mean(x['shares'][k] for x in stage) for k in stage[0]['shares']};assert abs(sum(macro.values())-1)<1e-12
    save('stage_compute_scope.json',dict(runs=stage,four_date_macro=macro,not_removable_saving=True,global_stage_new_policy_not_evaluated=True))
    lines=['# GRID138：补查计算空间与补查前信息核查','',
        '本轮完成约定的前两项，只读取既有计时和真实分支记录。没有拟合、模型前向、场景生成或物理回放。计时主分析覆盖GRID104全部48运行/366调用；其中原FRAME四日期各两重复作为主要计算分解。另完整覆盖GRID91原25个场景条目的179次调用，防止从四日期推及全部场景。','',
        '## 1. 同站补查在完整控制中占多少','',
        '| 记录范围 | 场景条目/完成 | 同站补查预测占控制时间 | 第一批64之后占比 |',
        '|---|---:|---:|---:|']
    for g in history['groups']:
        m=g['date_entry_macro'];lines.append(f"| {g['cohort']} | {g['runs']}/{g['complete']} | {100*m['local_share_lower']:.3f}%–{100*m['local_share_upper']:.3f}% | {100*m['after64_share_lower']:.3f}%–{100*m['after64_share_upper']:.3f}% |")
    t=timing['all4_date_macro'];tc=timing['completed3_date_macro']
    lines.append(f"| GRID104四开发日期、各两重复 | 4/3 | {100*t['local_share']:.3f}% | {100*t['after64_share']:.3f}% |")
    hg=next(g for g in history['groups'] if g['cohort']=='all25')['complete_entry_macro']
    lines += ['',f"只看完整周时，旧25中的20个完成条目占比为{100*hg['local_share_lower']:.3f}%–{100*hg['local_share_upper']:.3f}%；四开发日期中的三个完成日期为{100*tc['local_share']:.3f}%。提前结束条目始终保留并单列，不能把其时间少称为提速。",'',
        '占比先在原运行内计算，再平均原重复与场景条目。原25包含重复日历日期及已暴露场景，不是25个独立测试。历史绝对时间来自不同批次，本轮只比较各自内部的组成比例，不以跨批总时间判断方法速度。','',
        'G2旧版逐候选计时包含预测后的反馈解析，而汇总预测计时只覆盖simulate。本轮由源码与全部48次G2调用核实这项差额，用总差额给站内纯预测时间建立上下界；未按比例强行分摊开销。其它两组及GRID104内外计时一致。最初把二者强设相等的检查失败与原日志保留；恢复检查中手填调用数44又触发断言，已改为六场景实际调用数之和48，原脚本保留，未改任何历史数值。','',
        '上述是保存路径中的预测耗时组成。它不包含无法独立归属的可省循环开销，也不包含策略改变轨迹后的求解成本或新模型推理成本；不是完整闭环可实现提速的严格上界。即便如此，局部预测只占约1%–2%的规模，说明单靠删站内查询很难支撑较大的完整控制提速。不能再用站内查询节省比例替代总时间优势。','',
        '## 2. 哪个阶段计算规模更大','',
        '| GRID104四开发日期原FRAME | 完整控制时间占比宏平均 |','|---|---:|']
    for name,k in [('同站补查预测','local'),('全局候选预测','global_candidate'),('其它预测','noncandidate_forecast'),('连续优化求解','optimizer'),('搜索开销','search_overhead'),('其余控制开销','other')]:lines.append(f'| {name} | {100*macro[k]:.3f}% |')
    lines += ['',
        '全局候选阶段的已测预测计算规模大于站内补查。这是研究位置的依据，不等于这些预测都能删除；全局反馈核查已有明确作用，若学习介入必须以现有FRAME为强对照，验证剩余搜索的价值与质量风险。其它预测和连续优化属于共同控制底座，当前没有改造授权，也不把它们混作搜索模型可省开销。','',
        '## 3. 17个改变轨迹的状态有什么证据','',
        '17例均保留，16例可比较完整周成本，其中取消后9例成本上升、7例下降；1例提前终止不能比较完整周成本。原15个公开特征全部保留，不依据结果挑特征、阈值或模型。逐例表见《17例补查前信息与结果》。','',
        'S025存活失败是已知试点。排除同日和其它试点后，其补查前最佳候选预测rho高于参照记录的范围；当前记录不足以据此验证跨日期过载风险识别。S028成本失败则全部15个特征都落在其它日期参照范围内，最近的公开特征参照均通过原质量标准。因此简单的逐项越界检查不能发现所有已知风险；也不能由几何邻近推出不可学习。','',
        'S028补查前最佳候选预测rho为0.89612，已经低于原0.9阈值，但取消后整周成本仍超过FULL的100.5%。该rho是已预测候选结果，不是当前真实电网的实际rho。它反驳“只要候选已低于0.9就能取消”的充分性，不提供另一个可直接采用的新阈值。','',
        '全部15项与连续成本差的描述性秩相关已保存，不做显著性宣称或最好特征选择。16行、重复日期、场景尺度差异和已知结果限制了这些系数的解释；没有建立稳定的长期成本预测关系，也没有证明现有输入必然无法学习。查后奖励增量和真实分支结果只作对照，不混入查前输入。','',
        '## 4. 对后续路线的影响','',
        '如果主要目标是完整控制提速，优先核查全局分批阶段的剩余收益与计算机会。同站补查保留原条件质量作用，当前不继续扩大网络或数据，只为追求站内查询节省。若选择研究站内学习，应明确目标是运行质量/价值判断，并单独检验其计算收益，不能沿用一个不相称的总提速预期。','',
        '下一项建议仍是零训练的全局阶段资格分析：从原完整反馈记录检查停止点之后还剩多少有用候选、现行反馈是否提供可区分信息、简单策略能做到什么，按原日期分组保留失败。先确认有实质决策空间，再确定成对采集的动作、连续答案与预算；本轮未启动这个新分析或任何新采集。','',
        '原GRID104固定一批比较、所有学习种子及质量不合格结果均保留；原GRID135模型失败不由本轮占比重新解释为成功。最终8测试仍未生成，未改论文。']
    (O/'GRID138_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    lines=['# 17例：补查前信息与真实分支结果','',
        '成本差=取消−保留；正值表示取消更贵。查询差=保留−取消；正值表示取消后整周查询更少。提前终止的查询差只作记录，不称同质量计算节省。预测rho来自补查前最佳候选；即时奖励增量需补查后才知道，仅作结果对照。','',
        '| ID | 场景 | 步 | 查前预测rho | 待补查询 | 查后奖励增量 | 整周成本差 | 查询差 | 原质量评价 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rr:
        if r['physical_path_identical']:continue
        cost='不可比' if r['cost_delta'] is None else f"{r['cost_delta']:+.3f}"
        lines.append(f"| {r['case_id']} | {r['week']} | {r['step']} | {r['features']['incumbent_rho']:.6f} | {r['completion_budget']} | {r['instantaneous_reward_gain']:.8g} | {cost} | {r['queries_saved']:+d} | {r['quality_evidence_label']} |")
    lines += ['', '## 排除同日期后的公开特征邻近记录','',
        '只按15个原特征计算距离，参照为其它日期的非试点、质量结果已知状态；结果不参与距离计算。每维以参照跨度缩放，常量跨度取1，等权均方距离。不是分类器、部署决策或泛化测试。全部17例及所有距离保存在date_excluded_feature_neighbors.json。','']
    for n in public['failure_neighborhoods']:
        lines += [f"- {n['case_id']}：范围外特征{n['outside_reference_features']}；最近三例"+'、'.join(f"{x['case_id']}（{x['quality']}，取消成本{x['cost_direction']}）" for x in n['nearest3'])+'。']
    lines += ['', '56例中未发现15项特征逐项完全相同的记录。这既不构成矛盾标签证明，也不构成可学习性证明。原15项完整原值保存在public_feature_outcome_ledger.json。']
    (O/'17例补查前信息与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    route='''# GRID138后的学习路线建议

本轮改变的是研究优先级。局部补查预测占完整控制时间约1%–2%，现有FRAME已经把大部分全局候选计算压缩掉，再只优化局部尾部，完整控制可见收益较小。这个发现不否定补查的条件质量作用，也不否定学习。

建议接下来先评估学习介入全局分批核查的资格：是否存在比固定一批更有价值的反馈决策、可减少的计算是否足够、错停是否损失重要动作。先用完整保存反馈做离线诊断，明确它只反映原状态上的搜索价值；真实整周质量仍需后续成对分支与闭环验证。

仍固定“利用当前状态与已揭示搜索反馈估计后续评估价值”的问题，先确认应在哪个阶段决策，再锁定动作和标签。全局方案尚未建立，不直接把站内模型迁移过去、不重训、不改现有策略或门槛。

若继续站内学习，应以连续运行收益与质量风险为主要研究对象；终止、整周成本差、查询和真实时间分别保存。现有56例不能仅用达标二分或原MIXED类别替代这些结果。先前模型在自己的目标上失败依然保留。

新成对采集之前，需要确定采用哪一种决策位置与固定预算；不同时展开两套采集。当前没有新采集/训练，最终8测试未生成。
'''
    (O/'后续学习路线建议.md').write_text(route,encoding='utf8')
    save('decision.json',dict(decision='LOCAL_TIMED_COMPONENT_SMALL_REPRIORITIZE_GLOBAL_STAGE_FEASIBILITY',ready_to_scale=False,ready_to_train=False,adopted=False,
        local_four_date_macro=t['local_share'],local_historical25_interval=[history['groups'][-1]['date_entry_macro'][k] for k in ['local_share_lower','local_share_upper']],
        global_four_date_macro=macro['global_candidate'],stable_public_trigger_established=False,
        next_step='Saved-full-feedback qualification of global batch decisions before collection/modeling;new proposal only,not executed.',
        limits='Measured path components,not closed-loop achievable speed bounds. Small17-case exploratory predecision analysis cannot establish learning impossibility.',
        new_fits=0,new_physical=0,old_quality_and_results_unchanged=True))
    v();assert sha(W/'scope_resume02.py')==read(O/'scope_count_correction.json')['script_sha256']
    save('publication_receipt.json',dict(passed=True,main_review_sha256=sha(O/'review.json'),scope_review_sha256=sha(O/'historical_timing_summary.json'),
        original_data_labels_models_unchanged=True,engineering_failures_retained=['timing_semantics01','timing_count02'],
        analysis_only=True,new_fits=0,new_model_forwards=0,new_physical=0,final_tests_absent=True))
    collab=R/'论文协作资料/FRAME_补查计算空间与路线_GRID138';collab.mkdir(parents=True,exist_ok=False)
    names=['GRID138_report.md','17例补查前信息与结果.md','后续学习路线建议.md','timing_opportunity.json','historical_timing_summary.json','stage_compute_scope.json','public_information_summary.json','review.json','decision.json','publication_receipt.json']
    for n in names:shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    save('collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
    d=A/'publication';d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    shutil.copytree(O/'interrupted_checks',d/'interrupted_checks');shutil.copy2(REPO/'docs/NEXT_STEPS.md',d/'NEXT_STEPS_before_GRID138.md')
    (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
    entry='''

## 2026-10-24 — GRID138补查计算规模与查前信息核查完成

零拟合/前向/物理。复核GRID104全部48运行366调用，原FRAME四日期各两重复中站内预测占控制时间宏平均1.858%，第一批后1.017%，完成三日期为2.478%/1.356%。原三组全部25场景条目179调用补齐，站内预测0.865%–0.870%，完成20条目1.000%–1.005%；范围完整，非独立新场景，不跨批比较绝对速度。G2原计时内层simulate与外层含反馈解析不同，源码核实、48调用差额保留，以区间计算不改历史；原严格相等检查与计数44误填断言完整保存，恢复仅检查口径/计数。四开发日期全局候选预测占10.280%，连续优化62.221%；组成比例不当可实现提速上界。17改变轨迹案例原15查前特征与结果并列，16完整成本9升7降1存活不可比；S028查前候选rho0.896且15项无单项越界，说明简单阈值/越界不足，不证明不可学习。全部相关系数和跨日期邻近记录保留，不选择触发规则。研究优先建议先做全局分批核查的保存反馈资格分析，再决定新成对采集位置/预算；尚未执行新方案，不扩大局部网络以追求微小总时间，原补查条件质量作用保留。最终8封存，原模型/标签/门槛/论文未改。报告：research_archive/power-grid-2026/protocols/GRID138_opportunity/publication/GRID138_report.md
'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(old+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 决策位置与计算收益

1. GRID138完成既有计时/17分支查前分析：站内预测占控制时间旧25约0.87%、四开发日期1.86%，第一批后约0.44%/1.02%。这是原路径计算组成，不是可实现闭环提速上界；完成与未完成记录分别保留。
2. 原FRAME四开发日期全局候选预测占10.28%，规模更大。下一项建议先只读完整反馈核查全局分批决策的剩余收益/计算机会、原信息能否区分，保留强简单对照与所有失败；方案尚未执行，不直接移植/重训局部模型。
3. 17改变轨迹中16可比成本9升7降，1存活失败；S028已低于0.9且全部15特征无单项越界。没有新的可靠触发规则，也未证明输入不可学习，不按这些例子定阈值或挑样。
4. 新成对采集前先固定决策位置、动作、连续答案、日期隔离与总预算，通知用户再扩大。质量达标与计算获益分开，FULL周成本/未来分支不进入在线输入，单次结果不复制给每批或相加为多次策略收益。
5. 最终8测试未生成；原104/127/135学习失败、全部种子/标签/校准/质量标准保留。本轮无拟合/前向/物理/论文修改，不恢复自动化。
''',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID138计算空间与查前信息](protocols/GRID138_opportunity/publication/GRID138_report.md)：完整旧25与四开发日期站内预测占比约0.87%/1.86%；建议先评估全局分批学习资格，保留17分支与历史计时差异，零训练/回放。\n',encoding='utf8')
    files=0
    for root in A.iterdir():
        if not root.is_dir():continue
        for p,h in read(root/'archive_manifest.json').items():assert sha(root/p)==h;files+=1
    v();save('publication_verification.json',dict(passed=True,archive_files_verified=files,collaboration_files_verified=len(names),
        code_and_source_hashes_verified=True,final_tests_absent=True,git_push_pending=True))
    print('GRID138_PUBLICATION_VERIFIED',files,len(names),json.dumps(macro))
if __name__=='__main__':main()
