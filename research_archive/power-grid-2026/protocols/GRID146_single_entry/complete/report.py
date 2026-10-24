"""Lifecycle notes, result report and hash-verified publication forGRID146."""
from pathlib import Path
import json,shutil,sys
import run as G
O,R,REPO,A=G.O,G.R,G.REPO,G.A

def note(stage):
    if stage=='frozen':
        message='GRID146方案冻结：只在128处一次学习决策，继续则完整原FRAME接管；153拟合（118停/35继续）、66校准，原12/4/4日期划分。原18特征/36维，64树/深4/叶20/种子不变，153行按日期重新等权，无类别平衡。旧RF140分数同单入口控制分离结构/重训作用。预检219后缀/486完整搜索通过，一次拟合尚未启动；训练前通知，最终8不生成。'
    elif stage=='trained':
        cut=G.read(O/'cutoff.json');f=G.read(O/'fit_receipt.json')
        message=f"GRID146单次拟合完成，{f['rows']}行，{f['elapsed_s']:.4f}秒。新单入口阈值{cut['primary_threshold']:.17g}、旧分数单入口控制阈值{cut['old140_entry_control_threshold']:.17g}，从0.5按原经验零不合格程序校准。模型/阈值封存，开发验证尚未评分；不重训或改阈值。"
    else:raise ValueError(stage)
    entry=f'\n\n## 2026-10-24 — GRID146 {stage}\n\n{message}\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(f'# Next steps — GRID146 {stage}\n\n1. {message}\n2. 模型/阈值推送且哈希确认后，再评分原已暴露开发验证；保留全部15方法和两读数。\n3. 原动作保持/严格查询节省目标不变，筛查不优于CAP256则不采纳或补训；即使通过也只提出实际FRAME状态覆盖方案。\n4. 无新物理、测试8、RL、论文或扩大。\n',encoding='utf8',newline='\n')
    print('STATE_NOTED',stage)

def complete():
    G.verify_trained();review=G.read(O/'review.json');assert review['passed']
    assert G.read(O/'state.json')['stage']=='audited_pending_report'
    d=G.read(O/'design.json');cut=G.read(O/'cutoff.json');sums=G.read(O/'summary.json');dec=G.read(O/'decision.json')
    metric=lambda part,name:next(s for s in sums if s['evaluation_partition']==part and s['method']==name)
    main=metric('validation',d['primary']);control=metric('validation','OLD140_ENTRY128_CALIBRATED');cap=metric('validation','CAP256')
    rank=G.read(O/'fit_calibration_metrics.json')+[dict(partition='validation',**v) for v in G.read(O/'validation_metrics.json')]
    f=G.read(O/'fit_receipt.json');w=G.weights(G.rowfile(O/'fit_rows.json.gz'));fitrows=G.rowfile(O/'fit_rows.json.gz')
    lines=['# GRID146：128处单次学习决策','',
        '本轮只在原前128个候选评估完成、有可接纳动作且原搜索未强制停止时调用一次学习器。决定停止时执行原条件同站补查；决定继续时，原FRAME完成剩余搜索，后面不再调用学习停止器。候选顺序、控制器、选择规则、标签与补查规则均不改变。','',
        f"仅训练一份浅层随机森林，拟合153条状态（118停、35继续）来自12个原拟合日期，校准66条状态来自4个原校准日期。输入仍为18项公开摘要及18个缺失标记，64棵树、最大深度4、最小叶20、种子20261022均继承RF140。拟合耗时{f['elapsed_s']:.4f}秒，不能当在线推理或控制时间。",'',
        '每个拟合日期总权重153/12=12.75，同日期的入口状态等分权重。没有类别平衡；前一轮多前缀权重没有直接裁剪复用。缺失值中位数只用153条拟合入口重算，预处理规则相同，但数值未要求与旧多深度模型相同。','',
        '这是结构和拟合数据分布的联合改变，不冒称单因素权重实验。专门保留旧RF140模型/分数的单入口版本：结构相同，只复用旧分数，不重训；它与旧多出口RF140区分结构作用，与新RF146区分重新拟合的作用。不能凭一份种子把差异解释成普遍因果结论。','',
        f"原0.5经验阈值规则不变：在完整校准搜索中发现动作或查询不合格，就抬至最高错误停止分数，并重新检查全部搜索。新阈值{cut['primary_threshold']:.17g}，旧分数单入口控制阈值{cut['old140_entry_control_threshold']:.17g}。严格分数大于阈值才停止。二者均在开发验证评分前封存并推送；不是CRC/LTT统计保证。",'',
        '## 开发验证完整搜索结果','',
        '| 方法 | 动作差异 | 不合格学习停止 | 查询 | 查询节省日期宏平均 | 学习提前停止 |','|---|---:|---:|---:|---:|---:|']
    for r in [s for s in sums if s['evaluation_partition']=='validation']:
        lines.append(f"| {r['method']} | {r['action_differences']}/{r['calls']} | {r['joint_failures']} | {r['queries']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
    lines += ['',
        '宏平均沿用原口径：先算每个场景查询减少比例，再同日期副本平均，最后日期平均。查询汇总比例与宏平均不同，均保留。所有CAP参照保留原无可接纳动作继续、原强制停止和同站补查；它们不是无条件截断。非学习CAP的“学习提前停止”列为0，不代表未提前结束。','',
        f"预定主模型为RF146_ENTRY128_CALIBRATED：动作差异{main['action_differences']}/62，不合格学习停止{main['joint_failures']}，查询节省宏平均{100*main['date_macro_query_reduction']:.3f}%；旧分数单入口为{control['action_differences']}/62、{100*control['date_macro_query_reduction']:.3f}%；固定256为{cap['action_differences']}/62、{100*cap['date_macro_query_reduction']:.3f}%。",'',
        '| 日期 | 主模型动作差异 | 主模型查询 | 固定256查询 |','|---|---:|---:|---:|']
    for r in dec['per_date']:lines.append(f"| {r['date']} | {r['action_differences']} | {r['queries']} | {r['cap256_queries']} |")
    lines += ['',
        f"固定开发筛查通过={dec['passes_development_screen']}：全部动作保持、无不合格停止、查询节省宏平均严格超过CAP256，每个日期查询不更多且至少一个日期更少。没有按结果选择固定0.5、新阈值或旧分数控制版本替换主读数。",'',
        '## 同一入口的识别能力','',
        '| 分区 | 读数 | 状态数 | 继续数 | 继续类AP | 继续召回 | 有效停止召回 |','|---|---|---:|---:|---:|---:|---:|']
    for r in rank:lines.append(f"| {r['partition']} | {r.get('reading')} | {r['prefixes']} | {r['continue_count']} | {r['continue_ap']:.4f} | {100*r['continue_recall']:.2f}% | {100*r['valid_stop_recall']:.2f}% |")
    lines += ['',
        '这里的“继续”仍是：此刻停止加原补查不能同时保持原动作并严格少查。没有换成运行失败、一步奖励变差或未来整周成本标签。当前模型输出是森林分数，不是经验证的自然风险概率。辅助分类指标不代替完整搜索结果。','',
        '## 单入口性质与审计','',
        '同一保存状态的停止分数和停止后补查结果固定。提高阈值只能把该次学习停止改成原FRAME，不会暴露第二个学习出口。因此完整搜索动作差异、或动作/非节省联合损失均随阈值非增。这是固定状态上的结构性质，不能证明模型正确、跨日期泛化或整周经济质量。','',
        f"审计核对{review['public_suffix_checks']}条入口公开特征及隐藏后缀扰动、相同数量的独立森林遍历分数，最大偏差{review['max_score_error']:.3g}；{review['checks']['complete_searches']}条完整搜索的已查询集合/补查/最佳动作；{review['checks']['single_entry_shadow_searches']}条新入口搜索与原循环禁用后续学习出口的对照；{review['checks']['old_reference_rows_unchanged']}条旧参照保持不变。经验校准重执行，所有可学习入口分数边界两侧和原搜索回退核对通过。",'',
        '全部错误保存在all_single_entry_failures.json，保留固定0.5与校准两读数，不因结果追加训练、参数或阈值。无新场景、物理仿真、RL/神经网络、论文改动或最终8测试。开发验证已反复暴露，本轮仍是开发证据；FULL教师状态与实际FRAME闭环路径不同，查询减少不等于实测完整控制提速。','',
        '## 判读与下一步','']
    if dec['passes_development_screen']:
        lines += ['单入口通过了原开发搜索筛查，值得进入下一层证据，但不自动采纳或扩大。下一步应先核查实际FRAME保存状态覆盖和额外输入/推理开销，再由新冻结方案评估闭环质量与真实控制时间。']
    else:
        lines += ['单入口没有通过原开发搜索筛查，保持原RF140与固定预算对照，不采纳新模型。它解决了多出口误差不单调这个结构问题，尚未证明查询收益。接下来只读解释其错误/停止机会与旧分数单入口的差异，不自动追加第二次拟合、另一权重或新入口。']
    (O/'GRID146_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    G.save('complete.json',dict(passed_audit=True,new_fits=1,old_fits=0,development_screen_passed=dec['passes_development_screen'],
        adopted=False,ready_to_scale=False,new_physical=0,final8_absent=True))
    G.save('state.json',dict(stage='complete'));G.verify_trained();G.archive('complete')
    c=R/'论文协作资料/FRAME_单入口学习_GRID146';c.mkdir(exist_ok=False)
    names=['GRID146_report.md','design.json','planned_fit_weights.json','fit_receipt.json','cutoff.json','prevalidation_publication.json',
        'summary.json','validation_metrics.json','fit_calibration_metrics.json','calibration_history.json','old_entry_calibration_history.json','review.json','decision.json','complete.json']
    for n in names:shutil.copy2(O/n,c/n)
    msg=f"GRID146单入口一次拟合完成：128处153拟合/66校准，原18特征36维和RF参数，日期重新等权，无类别平衡。旧RF140分数单入口控制不重训。模型与两个阈值评分前发布；新主验证动作差异{main['action_differences']}/62、查询节省宏平均{100*main['date_macro_query_reduction']:.3f}%，旧分数单入口{100*control['date_macro_query_reduction']:.3f}%，CAP256 {100*cap['date_macro_query_reduction']:.3f}%；原筛查={dec['passes_development_screen']}。{review['public_suffix_checks']}公开后缀/独立分数、4575完整搜索及单入口回退核验通过。无新物理/最终8/采纳/扩大，全部读数与不利结果保留。报告research_archive/power-grid-2026/protocols/GRID146_single_entry/complete/GRID146_report.md"
    entry='\n\n## 2026-10-24 — GRID146单入口训练完成\n\n'+msg+'\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    nxt='只提出实际FRAME保存状态覆盖及推理开销核验方案，另行冻结后才回放，不自动扩大。' if dec['passes_development_screen'] else '先只读解释单入口错误、校准停止损失和旧分数单入口对照，不自动第二拟合、权重或入口。'
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID146完成\n\n1. '+msg+'\n2. '+nxt+'\n3. 原目标/基线不改，开发验证不是独立测试；最终8封存，无新物理/RL/论文或扩大。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID146单入口学习完成\n\n[报告](protocols/GRID146_single_entry/complete/GRID146_report.md)；仅128处一次学习停止，原FRAME接管后续。全部结果和旧分数结构控制保留。\n\n'+p.read_text(encoding='utf8'),encoding='utf8',newline='\n')
    count=0
    for phase in A.iterdir():
        for n,h in G.read(phase/'archive_manifest.json').items():assert G.sha(phase/n)==h;count+=1
    for p in c.iterdir():assert G.sha(p)==G.sha(O/p.name)
    G.save('publication_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),
        original_sources_models_cutoffs_unchanged=True,only_one_new_fit=True))
    d=A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json')
    (d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':G.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
    print('GRID146_COMPLETE',json.dumps(dec,ensure_ascii=False))

if __name__=='__main__':
    if sys.argv[1]=='complete':complete()
    else:note(sys.argv[1])
