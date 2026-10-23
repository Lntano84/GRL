"""Repair self-referencing archive manifest and interpret fixed predictions; no fits."""
from common135 import *
import shutil,collections
def main():
    verify();assert read(O/'review.json')['passed'];assert not (O/'publication_recovery_01.json').exists()
    dest=A/'complete';manifest=read(dest/'archive_manifest.json');bad=[n for n,h in manifest.items() if sha(dest/n)!=h];assert bad==['archive_manifest.json']
    history=O/'interrupted_attempts/publication_manifest_01';history.mkdir(parents=True,exist_ok=False);shutil.copy2(dest/'archive_manifest.json',history/'original_archive_manifest.json')
    save('publication_error.json',dict(error='Archive integrity assertion after opening manifest for writing included the empty manifest in its own hash list.',
        affected_keys=bad,numerical_results_or_models_changed=False,original_publication_source_unchanged=True))
    # The manifest attests archive members, never itself. Preserve the failed list.
    repaired={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    (dest/'archive_manifest.json').write_text(json.dumps(repaired,indent=2)+'\n',encoding='utf8')
    sources=[O/n for n in ['fits.json','families.json','validation_results.json','validation_trajectories.json','decision.json','diagnostics.json','thresholds.json','train_calibration_logits.npz','fit_metadata.json','calibration_metadata.json','fit_targets.npz','calibration_targets.npz','review.json','evaluation_freeze.json']]
    frozen={p.name:sha(p) for p in sources};save('interpretation_sources.json',frozen)
    S=read(O/'validation_results.json');F=read(O/'families.json');diagnostics=read(O/'diagnostics.json');th=read(O/'thresholds.json');rows=read(O/'validation_trajectories.json');decision=read(O/'decision.json')
    meta={part:read(O/f'{part}_metadata.json') for part in ['fit','calibration']}
    z=np.load(O/'train_calibration_logits.npz');witness=np.load(O/'calibration_targets.npz')['witness'];prob=np.load(O/'fit_targets.npz')['probability']
    groupmass=collections.defaultdict(float)
    for m,p in zip(meta['fit'],prob):groupmass[m['call']]+=float(p)
    badrows=[r for r in rows if r['expensive'] and r['action_disagreement']];bycall=collections.defaultdict(list)
    for r in badrows:bycall[r['call']].append(r)
    detail={}
    for call,rr in bycall.items():
        unique={r['sample_id']:r for r in rr};detail[call]=dict(unique_prefixes=len(unique),methods=sorted({r['method'] for r in rr}),date=rr[0]['date'],
            examples=[dict(sample_id=r['sample_id'],gain_lost=r['gain_lost'],skip_id=r['chosen_id'],full_id=r['baseline_id']) for r in unique.values()],
            score_minus_cutoff_range=[min(r['no_gain_score']-r['cutoff'] for r in rr),max(r['no_gain_score']-r['cutoff'] for r in rr)])
    anchors={}
    for name,cut in th.items():
        ss=no_gain(z[name+'__calibration']);assert cut==float(ss[witness].max());ii=np.flatnonzero(witness&(ss==cut))
        anchors[name]=dict(cutoff=cut,witness_prefixes=len(ii),witness_calls=sorted({meta['calibration'][i]['call'] for i in ii}),
            sample_ids=[meta['calibration'][i]['sample_id'] for i in ii],full_gains=[meta['calibration'][i]['full_baseline_gain'] for i in ii])
    stats={}
    for arm in ARMS:
        members=[f'{arm}_seed{s}' for s in range(3)]
        stats[arm]=dict(AUC_mean={part:float(np.mean([diagnostics[n][part]['auc'] for n in members])) for part in ['fit','calibration','validation']},
            query_saving_percent_by_seed=[100*S[n]['macro']['query_saving'] for n in members],
            expensive_disagreements_by_seed=[S[n]['expensive_disagreements'] for n in members],
            query_saving_minus_GROUP_pp_perdate={n:{d:100*(v['query_saving']-S['GROUP']['per_date'][d]['query_saving']) for d,v in S[n]['per_date'].items()} for n in members})
    analysis=dict(family_diagnostics=stats,new_expensive_miss_calls=detail,calibration_anchors=anchors,
        training_calls=len(groupmass),max_training_call_sampling_mass=max(groupmass.values()),sampling_mass_concentration_equivalent_calls=1/sum(p*p for p in groupmass.values()),
        concentration_caveat='Inverse squared sampling mass is a descriptive concentration statistic,not a count of independent observations or inference degrees of freedom.',
        total_binding_gate_changes=sum(d['binding_shift']['calibrated_decision_changes'] for d in diagnostics.values()),
        joint_binding_probability_change_range=[min(d['binding_shift']['max_abs_probability_change'] for n,d in diagnostics.items() if 'JOINT' in n),max(d['binding_shift']['max_abs_probability_change'] for n,d in diagnostics.items() if 'JOINT' in n)],
        not_a_new_readout=True,new_fits=0,new_model_forwards=0,thresholds_changed=False)
    save('interpretation.json',analysis)
    lines=['# GRID135：为什么完整尾部和配对仍未带来稳定改进','',
        '本页只解释已经冻结的12份模型和预测，没有重新训练、计算新模型前向、重选门槛或增加数据。','',
        '## 当前有用的结论','',
        '补齐首批以外候选没有带来稳定改善。主对照FULL_JOINT相对FIRST_JOINT的查询节省差，三个种子分别为−4.56、+0.63、0.00个百分点；同样保留配对时，看全尾并未整体更省。',
        '保留动作—反馈配对也没有超过汇总反馈。完整尾部中JOINT相对FACTOR分别少省1.89、0.15、0.00个百分点；不能以网络名中有关系模块就声称它学会了补查机制。','',
        '## 不同种子揭示的限制','',
        '种子0在四组都增加7条大补查误停，种子1无新增大组动作差异并小幅节省更多，种子2大组全部回退。所有种子保留，不能选中间一个作为方法结果。种子1在部分日期查询仍多于GROUP，三种子和逐日期条件均没有稳定超过旧控制。','',
        '| 方法 | 拟合AUC均值 | 校准AUC均值 | 验证AUC均值 | 三种子同站查询节省 | 新大组动作差异 |','|---|---:|---:|---:|---|---|']
    for arm,v in stats.items():
        a=v['AUC_mean'];lines.append(f"| {arm} | {a['fit']:.4f} | {a['calibration']:.4f} | {a['validation']:.4f} | {[round(x,2) for x in v['query_saving_percent_by_seed']]}% | {v['expensive_disagreements_by_seed']} |")
    lines+=['',
        '拟合排序已很强，换到校准/验证日期后下降。验证AUC约0.7说明存在一定排序信息，但无法直接推出保护所有有用补查时仍能省计算；校准所需的分数末端可靠性与平均排序是两个问题。此结果不证明所有模型都无法学习，也不证明只要增加独立场景就一定解决。','',
        '## 新误停究竟是什么','']
    for call,v in detail.items():
        lines.append(f"- {call}：{v['unique_prefixes']}条相关前缀，来自同一次物理搜索；四个种子0模型都误停。")
        for r in v['examples']:lines.append(f"  {r['sample_id']}：跳过候选{r['skip_id']}，完整补查选{r['full_id']}，预测奖励增量损失{r['gain_lost']:.12f}。")
    lines+=['',
        '这不是7个独立失败场景，更不是整周失败。旧小补查分支的7条差异另计，四组都保持不变。新增这一调用的误停表明当前模型保护收益的能力仍随种子变化。','',
        '## 配对信息有没有被使用','',
        f"固定循环错位诊断改变了JOINT概率，最大变化在{analysis['joint_binding_probability_change_range'][0]:.6f}至{analysis['joint_binding_probability_change_range'][1]:.6f}之间；全部12份模型的验证门控变化却是{analysis['total_binding_gate_changes']}。", 
        '所以不能说配对完全没被网络读取；能说的是，此诊断中它尚未形成门控决策上的可见作用。FACTOR约1e−7的概率变化属于浮点求和顺序量级。错位诊断是非物理输入干预，不能单独证明配对信息没有价值。','',
        '## 现有训练覆盖','',
        f"445条拟合前缀来自{analysis['training_calls']}次物理搜索、12日期，193条校准前缀来自24次搜索、4日期。单个调用占抽样概率最多{100*analysis['max_training_call_sampling_mass']:.2f}%；相关前缀重复训练不等于得到更多独立运行状态。", 
        '按原权重限制到大组是本轮匹配条件，不自动视作最优训练分布。当前不能把问题继续简单归为网络不够大，也不能据这一次实验断言结构信息不够。','',
        '## 下一步取舍','',
        '现有证据不支持继续把“更全的候选＋关系网络”当作主要增益来源。先保留原FRAME与GROUP，把跨调用/日期的误差、校准见证覆盖和训练样本重复性查清；需要新拟合时先提出与当前问题对应的固定对照，不直接增加网络或RL轮次。',
        '这不是恢复旧预算/调参或按已知误停设计触发规则的理由。最终8测试仍封存，尚不扩大数据或做新的物理闭环。','',
        '## 归档工程修复','',
        '首次发布在归档清单自校验处停止：清单写入时把尚为空的清单自身列为哈希成员。原错误清单已保留，恢复只将清单自身排除，模型、分数、动作、数值结果和冻结源码全部未改。原始错误版本也保留在Git历史。']
    (O/'结果解释与后续取舍.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for n,h in frozen.items():assert sha(O/n)==h
    receipt=dict(passed=True,archive_self_reference_only=True,all_numerical_source_hashes_unchanged=True,no_new_fit_or_threshold=True,
        preserved_bad_manifest_sha256=sha(history/'original_archive_manifest.json'),recovery_source_sha256=sha(Path(__file__)))
    save('publication_recovery_01.json',receipt)
    collab=R/'论文协作资料/FRAME_完整尾部匹配学习_GRID135'
    for n in ['结果解释与后续取舍.md','interpretation.json','publication_recovery_01.json']:
        shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    post=A/'interpretation';post.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),history/'original_archive_manifest.json',*[O/n for n in ['publication_error.json','publication_recovery_01.json','interpretation_sources.json','interpretation.json','结果解释与后续取舍.md']]]:shutil.copy2(p,post/p.name)
    (post/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in post.iterdir()},indent=2)+'\n',encoding='utf8')
    for phase in ['frozen','complete','interpretation']:
        root=A/phase
        for n,h in read(root/'archive_manifest.json').items():assert sha(root/n)==h,(phase,n)
    family_text='; '.join(f"{arm}:增量{100*v['macro']['gain_capture']:.5f}%/省查询{100*v['macro']['query_saving']:.3f}%/新增大组差异{v['expensive_disagreements_by_seed']}" for arm,v in F.items())
    entry='\n\n## 2026-10-23 — GRID135十二份匹配学习与解释完成\n\n'+family_text+'。各组同5938参数/901步/445拟合193校准/原12+4日期/三种子同初始化和抽样，完整Q及代理费信息全组相同，首批与全尾仅计划范围不同。全部权重和校准门槛先冻结；范围与配对主机制信号均False。种子0四组共同误停2037-08-03_101|call008的7相关前缀(候选35对576，每条G=.001331567764)，非7独立场景；种子1无新增大组差异但有日期查询不及GROUP，种子2全部大组回退。拟合AUC高、保留日期下降，已购反馈错位改变JOINT概率却没有门控变化；不称模型完全没读取配对或信息不可学。独立3156 NumPy前向最大误差2.61e-6、7080决策/查询/增量、12份901步优化器及旧便宜分支重现通过。出版自包含清单哈希工程错误已保留，仅排除清单自身，数值/模型/冻结源码未改，归档全通过。无新物理/新场景/最终测试，不采用/扩大/择种子/追加训练。报告：research_archive/power-grid-2026/protocols/GRID135_matched_tail/complete/GRID135_report.md；解释：research_archive/power-grid-2026/protocols/GRID135_matched_tail/interpretation/结果解释与后续取舍.md\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 聚焦跨调用可靠性

1. GRID13512份匹配分类完成：完整尾部与配对均无稳定机制信号；全部三种子和日期保留，不选择种子1、扫门槛或追加网络/训练。
2. 固定预测显示拟合强、保留日期弱，错误集中同一调用7相关前缀；451/445等前缀数量不能当独立数据规模。先核查原51训练调用/24校准调用的覆盖、抽样重复与校准见证，再决定下一固定对照。
3. JOINT概率会受反馈错位影响，但验证门控没有变化，不概括为配对未被读取；旧统计/关系失败与新的负结果一起保留。
4. 保留原FRAME/GROUP；新增大组差异与旧便宜分支7差异分开。局部增量/查询不等于整周质量或速度，最终8测试封存，不自动扩大/物理回放/改论文。
'''.replace('451/445等前缀数量','445条前缀'),encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID135完整尾部/配对匹配分类](protocols/GRID135_matched_tail/complete/GRID135_report.md)：四组各三种子同容量预算，范围与配对未形成稳定优势；[结果解释](protocols/GRID135_matched_tail/interpretation/结果解释与后续取舍.md)保留单调用新误停、种子波动及归档清单修复。\n',encoding='utf8')
    verify();print(json.dumps(dict(publication_recovery=receipt,analysis=analysis),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
