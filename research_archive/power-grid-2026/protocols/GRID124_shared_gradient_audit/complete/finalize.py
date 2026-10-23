"""Audit all descriptive addenda, then publish immutable read-only diagnosis."""
import diagnose as d
import gzip
np=d.np

def main():
    assert not (d.O/'complete.json').exists();d.verify();review=d.read(d.O/'review.json');assert review['passed'];analysis=d.read(d.O/'analysis.json');cells=d.read(d.O/'action_cell_accounting.json');tails=d.read(d.O/'trajectory_accounting.json');heads=d.read(d.O/'head_consistency.json')
    for fn,script in [('posthoc_partition_plan.json','partition.py'),('posthoc_trajectory_plan.json','divergence.py'),('posthoc_head_plan.json','head_consistency.py')]:assert d.read(d.O/fn)['script_sha256']==d.sha(d.W/script)
    assert d.read(d.O/'engineering_recovery_01.json')['correction_sha256']==d.sha(d.W/'recovery01.py')
    metrics=analysis['state_metrics']
    for row in cells['rows']:
        n=row['name'];arm='AUX_GAIN' if '__AUX_GAIN_' in n else 'AUX_NETSIGN';b=n.replace(arm,'NET_ZERO');part='validation' if row['domain']=='validation' else 'held';src=metrics[n][part];ref=metrics[b][part]
        assert abs(row['huber_difference']-(src['main_huber']-ref['main_huber']))<1e-10
        assert abs(row['sign_regret_difference']-(src['sign_regret']-ref['sign_regret']))<1e-8
        assert abs(sum(v['mass'] for v in row['cells'].values())-1)<1e-12
    trajs={dom:{(v['method'],v['sample_id']):v for v in d.read(d.S/fn)} for dom,fn in [('validation','full_validation_trajectories.json'),('oof','oof_trajectories.json')]}
    cases=d.src.pickle.loads((d.R/'outputs/grid102_search_rl/cases.pkl').read_bytes());by={v['sample_id']:v for v in cases};price=d.read(d.S/'design.json');div={};counts={dom:d.collections.Counter(v['date'] for v in tt.values() if v['method']=='NET_ZERO_seed0') for dom,tt in trajs.items()};grouped={};checked=0
    with gzip.open(d.O/'first_divergences.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=d.json.loads(line);dom=r['domain'];sid=r['sample_id'];seed=r['seed'];arm=r['arm'];a=trajs[dom][f'{arm}_seed{seed}',sid];b=trajs[dom][f'NET_ZERO_seed{seed}',sid];c=by[sid]
            first=next((i for i,(u,v) in enumerate(zip(a['actions'],b['actions'])) if u!=v),None);assert first==r['first_divergence_position']
            weight=1/(3*len(counts[dom])*counts[dom][c['date']]);assert abs(weight-r['date_case_seed_weight'])<1e-15
            if first is None:assert a['actions']==b['actions'] and r['group']=='same_actions'
            else:
                net=(c['gains'][first]-price['cost_per_query']*c['queries'][first])/price['gain_scale'];g=('baseCONT_auxSTOP' if b['actions'][first] else 'baseSTOP_auxCONT')+('__nextPositive' if net>0 else '__nextNonpositive');assert g==r['group'] and abs(net-r['next_net'])<1e-12
            assert abs(r['trajectory_delta']-(a['net_utility']-b['net_utility']))<1e-12
            key=(dom,arm,r['group']);grouped[key]=grouped.get(key,0)+weight*r['trajectory_delta'];div[dom,arm,seed,sid]=r;checked+=1
    assert checked==17808 and len(div)==checked
    for (dom,arm,group),v in grouped.items():assert abs(v-tails['summary'][dom][arm]['groups'][group]['utility_contribution'])<1e-12
    Q=np.load(d.S/'predictions.npz');mm={part:d.read(d.S/f'{part}_metadata.json') for part in ['train','validation']};lookup={part:{(v['sample_id'],v['position']):i for i,v in enumerate(vv)} for part,vv in mm.items()};folds=d.read(d.S/'design.json')['folds'];hc=d.collections.Counter()
    for r in heads['records']:
        dom=r['domain'];arm=r['arm'];s=r['seed'];sid=r['sample_id'];t=div[dom,arm,s,sid];assert t['group']==r['group'] and t['first_divergence_position']==r['position'];c=by[sid];f=next((ff['fold'] for ff in folds if c['date'] in ff['held_dates']),None);split='full' if dom=='validation' else 'f'+str(f);part='validation' if dom=='validation' else 'train';q=Q[f'{split}__{arm}_seed{s}__{part}'][lookup[part][sid,r['position']]];assert float(q[0])==r['aux_logit'] and float(q[1])==r['predicted_net'];label=c['gains'][r['position']]>0 if arm=='AUX_GAIN' else t['next_net']>0;assert bool(label)==r['own_aux_label'] and bool((q[0]>0)==label)==r['aux_correct'];hc['|'.join([dom,arm,r['group']])]+=int(r['aux_correct'])
    assert len(heads['records'])==sum(r['group']!='same_actions' for r in div.values())==6665
    for key,v in heads['groups'].items():assert v['aux_correct']==hc[key]
    d.save(d.O/'review_addenda.json',dict(passed=True,all30_action_cell_differences_checked=True,all17808_first_divergences_checked=True,all6665_head_records_checked=True,all_differences_conserved=True,source_hashes_unchanged=True,posthoc_descriptive_only=True,new_fits=0,weight_updates=0,new_physical_calls=0,final_test_materialized=False))
    summary={}
    for scope in ['full','folds']:
        summary[scope]={}
        for arm in d.ARMS:
            names=[n for n in metrics if '__'+arm+'_seed' in n and ((n.startswith('full__'))==(scope=='full'))];ss={}
            for part in ['fit','validation' if scope=='full' else 'held']:
                ss[part]={k:float(np.mean([metrics[n][part][k] for n in names])) for k in metrics[names[0]][part] if k!='states'}
            if arm!='NET_ZERO':
                ss['fit_main_huber_worse_pairs']=sum(metrics[n]['fit']['main_huber']>metrics[n.replace(arm,'NET_ZERO')]['fit']['main_huber'] for n in names)
                ss['fit_pairs']=len(names)
            summary[scope][arm]=ss
    grads=analysis['gradient_records'];final=[v for v in grads if v['stage']=='final'];init=[v for v in grads if v['stage']=='initial']
    gs=dict(final_pairs=len(final),final_raw_negative_cosines=sum(v['cosine']<0 for v in final),initial_raw_negative_cosines=sum(v['cosine']<0 for v in init),final_negative_combined_main_dot=sum(v['combined_main_directional_dot']<0 for v in final),final_negative_adam_v_metric_combined_main_dot=sum(v['adam_v_metric_combined_main_dot']<0 for v in final))
    d.save(d.O/'summary.json',dict(state_metric_means=summary,gradient_geometry=gs,trajectory_groups=tails['summary'],new_fits=0,new_physical_calls=0))
    report='''# GRID124：辅助学习为什么没有稳定改善补查

本轮只分析已有30份辅助模型、15份原回归对照及保存的训练记录，没有重新训练、调系数或电网回放。结论是：**辅助任务确实学到了训练标签，但没有稳定改善有代价的补查选择；也不能把问题简单归为两个梯度相互抵消。**

## 1. 辅助分类学到了什么

“出现改进”辅助分类的加权AUC，在完整训练侧为0.9943，原验证侧为0.8092；四折训练侧平均为0.9978，留出日期侧为0.7710。它不是完全没有学到信息，但训练侧与留出侧存在差距。

概率误差则更不理想：原验证侧BCE为0.4694，高于只预测训练侧正例比例的常量对照0.3147；留出日期侧平均为0.8424，常量对照为0.3614。Brier误差也更大。说明其排序包含信息，概率预测却没有稳定优于简单常量，不能把AUC高直接解释成可靠的补查价值。辅助概率在GRID123中没有直接控制动作，这些只是诊断指标。

“净收益为正”辅助分类同样出现训练侧AUC很高、验证/留出侧概率误差不及常量对照的情况。因此，问题不只发生在原始改进标签上。原始改进事件与正净收益也不是同一个目标：原训练侧438个改进状态中，仅211个扣查询费后为正，不能把识别到改进直接视为值得补查。

## 2. 主回归在训练侧已经发生取舍

沿用原抽样权重、仅比较净收益主输出的Huber误差：原回归完整训练侧为0.01345，加改进辅助任务后为0.02918，三个种子全部增大；四折训练侧平均从0.01154增至0.02449，12个配对模型全部增大。净收益正负辅助的15个配对模型也全部增大。

这说明辅助任务替换原停止头损失后，对训练目标产生了可测的取舍，并非只有换到新日期才出现问题。但训练误差增加本身不等于泛化必然变差，下一节保留相反方向的指标。

## 3. 平均回归误差改善，为何补查选择仍退步

改进辅助模型在原验证侧的主回归Huber误差从0.08230降至0.06925；与此同时，单状态决策遗憾从0.01783升至0.02642，完整补查搜索净效用从0.16292473降至0.14956484。

按原模型与新模型是否继续补查，把全部状态分成四类后发现：误差的改善主要来自“两者都会继续”的状态，这一类贡献Huber变化−0.01380075；它改善了数值预测，却没有改变这一状态上的继续/停止选择。“原模型继续、新模型停止”一类的决策遗憾则增加0.01041130，超过“原模型停止、新模型继续”一类带来的0.00182043改善。这里是全部状态上的诊断，不能与实际访问的完整搜索路径混为一谈。

因此，我们又按双方实际共同搜索前缀的首次动作分歧，核对全部已保存的补查轨迹。原验证侧改进辅助模型相对原回归的净效用变化为：

- 下一批无正净收益时，原模型继续、新模型停止：300条前缀—种子记录，组内完整搜索尾部差值合计+0.00821384。
- 下一批无正净收益时，原模型停止、新模型继续：337条记录，贡献−0.00313834。
- 下一批有正净收益时，原模型继续、新模型停止：13条记录，来自7次实际调用，贡献−0.02278094。
- 下一批有正净收益时，原模型停止、新模型继续：10条记录，贡献+0.00434556。
- 其余轨迹的动作完全相同，贡献为0。

四组差值严格相加为−0.01335989，与GRID123原结果一致。**少量漏掉的高收益机会抵消了较多节省查询的收益。** 记录共享前缀和训练种子，不是13个独立场景；调用也可能跨组重叠。每组的下一批标签只是分组依据，报告的效用贡献包含从首次分歧到各自搜索结束的全部已保存尾部，不能归因于一次查询，更不代表整周物理成本。

上述13条误停记录中，6条的辅助头实际上已经给出正改进判断，但净收益头仍选择停止；另外7条的辅助头也未识别到改进。折外对应64条误停记录中，16条辅助头判断为正、48条判断为负。这是两个不同的失误来源；事件正确也不自动保证收益估计正确，不能靠一个梯度解释涵盖全部情况。两个头表达不同条件量，其输出差异本身也不是数学上的矛盾。

折外结果完整保留：两种方向的得失基本抵消，净效用微增0.00064121，仍不能改变GRID123种子与日期方向不稳的结论。

## 4. 梯度究竟是否冲突

之前4～11倍的判断只来自首个小批次。本轮在每个模型自己的训练日期上，按原抽样概率计算全体训练状态的期望梯度，并核对原初始化与训练终点。

30个辅助训练终点中，29个主任务/辅助任务共享层梯度内积为正，仅1个为负；把两者相加后，30个终点的方向对主任务均具有正内积。固定保存的Adam二阶矩进行对角缩放后，合并方向的这一结论仍成立。因此，**没有证据说明训练终点普遍存在梯度互相抵消，不应直接把梯度投影当作已确定的修复方案。**

尺度差异依然存在。改进辅助的三个完整训练模型，终点辅助/主任务共享梯度范数比分别约5.79、5.28、6.41，余弦相似度仅约0.112、0.054、0.114：辅助梯度更大，但大部分方向不与主梯度重合。这支持继续研究共享表示和任务取舍，不能证明辅助更新必然有害。

初始化有更多反向梯度，终点却大多同向；我们没有保存每一步的完整分任务梯度，不能据两个端点声称整段训练无冲突。固定Adam二阶矩的诊断也没有包含一阶动量，既不是实际下一步Adam更新，也不是调整权重后重新训练的反事实结果。

## 下一步判断

保留原收益回归模型。当前更值得解决的是：**少数高收益补查为何被低估，以及训练目标如何把注意力放到会改变继续/停止选择的状态上。** 不应仅因辅助分类训练得好就继续加大其权重，也不应因为梯度范数大就认定需要冲突投影。任何后续学习改动都应同时保留主回归误差、误停代价和完整补查效用三种检查，防止只把平均预测误差做得更好。

本轮没有挑新触发条件、调辅助系数、择种子或继续训练；没有新增强化学习、场景、最终测试、电网闭环或论文图表。

## 核验与材料

60组初始化/终点的训练分布梯度由独立NumPy手工反向传播核验，最大绝对误差低于7e−17；30组辅助验证/留出AUC由成对加权计算复核。另核对30组状态误差分解、17808条完整轨迹配对和6665条首次分歧的双输出记录，差值守恒。所有训练侧梯度均不包含对应留出日期。

梯度分析及其范围先封存；看到结果后增加的四类状态分解、首次分歧分解和双输出核查，均单独记录为事后描述性分析，不作预注册验证或统计显著性主张。原脚本一次NumPy布尔负号格式错误保留在工程收据中，仅修正常量BCE公式的等价写法；没有改动模型、数据或原结果。

这是已暴露同一电网的FULL教师路径局部搜索诊断。训练前缀相关，费用与尺度仍带有历史任务定义依赖；不是新的独立测试，也不能推出整周质量、实际控制时间或跨电网泛化结论。
'''
    (d.O/'GRID124_report.md').write_text(report,encoding='utf8')
    c=d.R/'论文协作资料/FRAME_辅助学习诊断_GRID124';c.mkdir(exist_ok=False)
    d.save(d.O/'complete.json',dict(passed=True,gradient_review_passed=True,addenda_review_passed=True,new_fits=0,optimizer_instances=0,weight_updates=0,new_physical_calls=0,final_test_materialized=False,report=str(d.O/'GRID124_report.md'),archive=str(d.A/'complete'),collaboration=str(c),ready_to_scale=False))
    d.archive('complete')
    for fn in ['GRID124_report.md','summary.json','review.json','review_addenda.json','complete.json']:shutil_copy(d.O/fn,c/fn)
    d.docs('## 2026-10-23 — GRID124 shared-gradient and false-stop diagnosis complete\n\nReadonly30auxendpoints/15cachedbasecontrols;60manualNumPygradientpairsaudit,30pairwiseAUCchecks,17808trajectorypairs/6665headrecordsconserved. AUX_GAINfitAUC0.9943,validation0.8092/OOFheld0.7710,butheldBCE/Brierworsethanfitprevalenceconstant. MainfitHuberall15pairsworse,validationHuberimprovesmostlyCONT_CONTstateswhiledecisionutilitydeclines. Finalsharedgradient29/30positivecosine,allcombinedraw/fixedAdamvmetricmaindotspositive,notpersistentconflictcausality. Validation13profitablefirstdivergencefalseSTOPrecords/7callscontribute-0.02278094,ofwhich6eventheadpositive;allgroupssum-0.01335989. PreserveOOF+0.00064121/inconsistentseeds. Noextraoptimizer/weighttuning/physical/test. Retainbase;prioritizehighvaluestopdecisionsratherthanblindgradientprojection.',
      '# Next steps — GRID124辅助学习诊断完成\n\n1. 向用户解释：事件分类学到信息，但少数高收益补查误停仍抵消查询节省；不能用平均回归误差代表补查质量。\n2. 如继续设计学习改动，优先对齐高代价误停与继续/停止决策；同时检查主回归误差、误停代价和完整搜索效用。\n3. 不自动调辅助权重、上梯度投影、重训、扩数据、物理回放或开启最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID124_shared_gradient_audit/complete/GRID124_report.md\n')
    with (d.REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID124辅助学习为何未改善补查](protocols/GRID124_shared_gradient_audit/complete/GRID124_report.md)：零训练，完整训练分布梯度、辅助预测误差及全部首次分歧差值守恒核查。\n')
    print('GRID124_COMPLETE_ARCHIVED',flush=True)

def shutil_copy(a,b):
    import shutil
    shutil.copy2(a,b)

if __name__=='__main__':main()
