"""Independent arithmetic/entropy checks; Chinese interpretation and selective archive."""
from analyze import *
import shutil,math
REPO=R/'work/GRL-results-integration'
def independent_review():
    verify113();sealed();configure();assert read(O/'analysis_receipt.json')['passed'];metrics=read(O/'endpoint_metrics.json');Q=np.load(S/'state_predictions.npz');model_records=read(S/'fits.json')['records'];checked=0;max_entropy_error=0.
    for part in ['train','validation']:
        z=np.load(S/f'{part}_targets.npz');y=z['target'].astype(float);meta=read(S/f'{part}_metadata.json');p=z['sampling_probability'] if part=='train' else np.ones(len(y))/len(y)
        raw=old_predictions(meta)
        for rec in model_records:raw[f"{rec['arm']}_seed{rec['seed']}"]=Q[f"{rec['arm']}_seed{rec['seed']}__{part}"]
        for name,q in raw.items():
            stop=0.;cont=0.;fp=0;fn=0;false_q1_nonpositive=0
            for i in range(len(y)):
                e0=float(q[i,0]);e1=float(q[i,1]-y[i]);stop+=p[i]*(.5*e0*e0 if abs(e0)<1 else abs(e0)-.5)/2
                cont+=p[i]*(.5*e1*e1 if abs(e1)<1 else abs(e1)-.5)/2
                go=bool(q[i,1]>q[i,0]);fp+=int(y[i]<=0 and go);fn+=int(y[i]>0 and not go);false_q1_nonpositive+=int(y[i]>0 and not go and q[i,1]<=0)
            m=metrics[name][part];assert abs(stop+cont-m['weighted_huber'])<1e-10
            assert fp==m['negative_continues'] and fn==m['positive_misses'] and false_q1_nonpositive==m['misses_qcontinue_nonpositive'];checked+=len(y)
            # Direct numerical derivative in score space, not an optimizer/model update.
            eps=1e-5
            def hscalar(t):return .5*t*t if abs(t)<1 else abs(t)-.5
            derivative=sum(p[i]*(hscalar(float(q[i,1]-y[i])+eps)-hscalar(float(q[i,1]-y[i])-eps))/(4*eps) for i in range(len(y)))
            assert abs(derivative-m['continue_bias_gradient']['total'])<1e-7
    # Independent NumPy softmax entropy on fixed every151st training row for every checkpoint.
    rows=pickle.loads((S/'train_public.pkl').read_bytes());ids=list(range(0,len(rows),151));sample_entropy=[]
    for rec in model_records:
        name=f"{rec['arm']}_seed{rec['seed']}";cp=torch.load(S/'models'/f'{name}.pt',map_location='cpu',weights_only=False);m=RelationalValue(rec['arm']);m.load_state_dict(cp['state_dict']);m.eval()
        weights={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()}
        def lin(x,k):return x@weights[k+'.weight'].T+(weights[k+'.bias'] if k+'.bias' in weights else 0)
        def action(x):return np.maximum(lin(np.maximum(lin(x,'A.0'),0),'A.2'),0)
        for i in ids:
            r=rows[i];a=action(r['context_action']);q=action(r['next_action']);l=lin(q,'Q')@lin(a,'K').T/math.sqrt(8);ex=np.exp(l-l.max(1,keepdims=True));att=ex/ex.sum(1,keepdims=True)
            value=float(np.mean(-np.sum(att*np.log(np.maximum(att,1e-30)),axis=1))/math.log(len(a))) if len(a)>1 else 0.
            with torch.no_grad():_,d=detailed_forward(m,collate([r]));original=float(d['attention_normalized_entropy'][0])
            max_entropy_error=max(max_entropy_error,abs(value-original));assert abs(value-original)<1e-6
            sample_entropy.append(dict(method=name,index=i,numpy_entropy=value,torch_entropy=original))
    seq=read(O/'sequence_objective_check.json');val=seq['results']['validation'];oldres=read(S/'validation_results.json')
    for name in oldres:
        if name in val:assert abs(val[name]['macro_utility']-oldres[name]['all']['macro']['mean_net_utility'])<1e-12
    assert abs(val['MYOPIC_LABEL_ORACLE']['macro_utility']-val['HINDSIGHT_PREFIX_CEILING']['macro_utility'])<1e-12
    assert sum(r['partition']=='validation' for r in seq['hindsight_bridge_cases'])==0
    # Different sums and grouping implementations for training complete sequences.
    sr=read(O/'sequence_diagnostic_rows.json')
    membership={part:{m['sample_id'] for m in read(S/f'{part}_metadata.json')} for part in ['train','validation']}
    for part in ['train','validation']:
        for name,r in seq['results'][part].items():
            groups=collections.defaultdict(list)
            for row in sr:
                if row['method']==name and row['sample_id'] in membership[part]:groups[row['date']].append(row['net_utility'])
            macro=sum(sum(v)/len(v) for v in groups.values())/len(groups);assert abs(macro-r['macro_utility'])<1e-12
    assert all(sha(R/n)==h for n,h in read(O/'input_manifest.json').items())
    save(O/'entropy_audit_samples.json',sample_entropy)
    save(O/'review.json',dict(passed=True,independent_scalar_loss_decision_checks=checked,score_derivatives_checked=True,
        independent_numpy_attention_states=len(sample_entropy),max_entropy_error=max_entropy_error,complete_sequence_macros_independently_checked=True,
        validation_zero_hidden_bridge_opportunities=True,all_six_weight_and_source_hashes_unchanged=True,new_fits=0,new_optimizer_instances=0,new_physical_calls=0,final_test_materialized=False))
def publish():
    assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();m=read(O/'endpoint_metrics.json');zero=read(O/'zero_prediction_loss_geometry.json');exp=read(O/'exposure_and_loss_windows.json');lat=read(O/'representation_diagnostics.json');grad=read(O/'gradient_decomposition.json');cross=read(O/'decision_loss_crosscheck.json');seq=read(O/'sequence_objective_check.json');binding=read(O/'binding_sensitivity.json');cons=read(O/'consensus_failures.json')
    avg=lambda part,fmt:float(np.mean([seq['results'][part][fmt.format(s)]['macro_utility'] for s in range(3)]))
    jtrain=avg('train','JOINT_seed{}');dtrain=avg('train','DIRECT{}');jval=avg('validation','JOINT_seed{}');dval=avg('validation','DIRECT{}')
    report=['# GRID114：正收益漏查的进一步核对','',
        '**结论：不是正例没进入训练，也不是停止价值抬高。当前模型确实把更多有价值的补查估成了不值得继续，但它同时减少了无利查询。因此，“训练集也漏得多”不能直接等同于整体拟合更差：训练回归损失与单步收益权衡反而更好，完整训练搜索效用接近旧模型，验证退步更明显。配对通道也尚未形成明显的决策作用。**','',
        '本轮只读取既有4438个状态、原抽样记录及六份固定权重，计算终点梯度、表示和逐序列记账。没有训练、优化器更新、电网预测或新回放，最终8测试场景未生成。前缀和日期副本相关，结果仅是同电网开发诊断。','',
        '## 1. 正收益样本确实进入了训练','',
        f"训练正收益211个状态，占原抽样权重{zero['positive_sampling_mass']*100:.2f}%，来自54次搜索。三个种子各抽到约1.31万次正样本；每个正状态都被抽到，最低8—9次，中位数60—62次。这里是原7250更新预算；GRID109的最低43次属于29000更新，不混用。",
        '因此可以排除“正例从未进入训练”的解释，仍不能把重复前缀当成独立运行模式，也不能据抽到过就认定训练充分。','',
        '## 2. 漏查来自继续价值预测，不是停止头','',
        'JOINT三个种子训练漏停82/72/81，验证漏停20/23/26。全部这些漏停的继续价值都已≤0；没有出现“继续值是正的，却被更高停止值压住”的情况。因此仅把停止输出固定为0，不能修复这些既有漏停。',
        '54个训练正状态被JOINT三个种子共同漏掉；其中21个在原DIRECT三个种子中都能继续，来自7次搜索、7个日期。验证中相应为5个状态、4次搜索、2个日期。完整清单均保存，不只看第552步。','',
        '## 3. 必须修正上一轮对“没学好”的概括','',
        '按原抽样概率重新计算Huber目标，JOINT三个种子均低于原DIRECT：','']
    for s in range(3):report.append(f"- 种子{s}：JOINT {m[f'JOINT_seed{s}']['train']['weighted_huber']:.8f}，DIRECT {m[f'DIRECT{s}']['train']['weighted_huber']:.8f}；正收益漏停却是{m[f'JOINT_seed{s}']['train']['positive_misses']}对{m[f'DIRECT{s}']['train']['positive_misses']}。")
    report += ['',
        '正、负两部分的加权回归损失也都降低了。JOINT剩余损失约89.7%—92.5%仍来自正收益状态，不能说优化器完全忽略了正例。更准确的问题是：奖励数值拟合变好了，收益符号和查询取舍却不一定更合适。',
        '漏停增加主要发生在小、中等正收益：归一化净收益0—0.1的59个训练状态，JOINT漏34—39个，DIRECT漏12—26个；0.1—1的86个状态，JOINT漏37—41个，DIRECT漏12—15个。大于1的66个状态，DIRECT全能继续，JOINT仍漏1—2个。分组只用于描述，未改查询门槛。',
        '但漏停次数不能单独代表全部决策质量。把无利查询的代价也计入，JOINT三个种子的训练单步加权决策遗憾均比DIRECT小；执行完整搜索后，训练净效用分别平均为'+f'{jtrain:.8f}与{dtrain:.8f}，JOINT差{jtrain-dtrain:+.8f}。验证则为{jval:.8f}与{dval:.8f}，差{jval-dval:+.8f}。',
        '这表明训练内的整体收益取舍接近，向验证日期迁移后的损失更明显。当前既有正收益捕获偏差，也有迁移问题，不能把原因全部归到“训练没做够”。这也不证明已经收敛：只保存固定终点，没有中间权重，批次损失有波动，无法恢复每个正状态何时学会或丢失。','',
        '## 4. Huber存在可核验的方向偏差，但不是充分原因','',
        '令y为原下一批归一化净收益，继续输出为a，停止目标固定为0。原两输出平均Huber对继续输出的导数为0.5·clip(a−y,−1,1)。当a=0时，按原概率加权，正收益样本贡献−0.01228335，非正样本贡献+0.01648925，合计+0.00420590。梯度下降会先把共享的继续输出向负方向推；但同一批标签的期望净收益为+0.03048721。',
        '这个算术例子说明：Huber的稳健位置估计与期望净收益不是同一个目标，少数大收益的推动会被限幅。它只说明共享截距的局部方向，不代表有条件模型必须学成全停，也不是本次六个模型训练过程的完整因果解释。',
        '终点也并非始终由负样本压过正样本：JOINT正例梯度范数都大于负例；两组梯度有冲突，但总梯度仍希望部分收益值回升。更不能据此直接重做MSE：GRID108在两个固定预算上都试过，验证未稳定胜过Huber，全部反例继续有效。','',
        '## 5. 当前模型没有明显利用配对关系来改变正收益决策','',
        'JOINT对正训练状态的注意力归一化熵中位数为0.9932/0.9993/0.9992（1表示均匀），查询对不同已查记录的权重差异较弱。但均匀注意力不等于联合值编码没有作用，不能只凭熵判定模块失效。',
        '为进一步检查，只对每条固定输入的已查反馈做一次循环移位：保留动作集合、反馈多重集、原41维和待查集合，仅改变动作—反馈对应关系。此变换与标签无关，未训练任何错配模型，也不把扰动当可部署策略。',
        'JOINT正训练状态的输出优势绝对变化中位数约0.00087/0.00147/0.00123，三个种子均未改变任何一个正收益状态的继续/停止；验证41个正收益状态也全部保持原决定。其余状态有少量决定变化，FACTOR输出按预期近似不变。',
        '这说明当前权重下这一次配对扰动对关键决策影响较弱，不足以证明模型完全不用对应关系，更不否定这种信息本身的价值。当前动作编码没有发现同批不同连接方案被精确编码成同一个向量；3582个训练头输入也不完全重复。没有发现整条输入通路彻底塌成常量的工程错误，仍不能排除近似压缩或表示不足。','',
        '## 6. 这批验证失败不是必须依赖长期规划才能解决','',
        '用已保存反馈作后验诊断：每步知道下一批真实净收益、只在其为正时继续，验证宏平均净效用为0.19070118，恰好等于知道所有后续批次的最优停止上界。验证没有“眼前亏、继续几批才赚”的桥接机会；训练有23条相关起始前缀，后验上界高于即时贪心，但不能把这些重复当独立场景。',
        '这些oracle读取了尚未购买的教师反馈，只是诊断上界，不能部署或作为新模型成绩。验证上的相等意味着本轮验证退步不需要通过更长规划才能解释；先估准什么时候下一批值得查就仍有明显空间。GRID103原gamma1与gamma0对照也未证明长期奖励更好，不能把再次换RL当成现成答案。','',
        '## 我的判断','',
        '当前最值得解决的不是规模，而是让模型保留原有收益判断能力，再检验新增配对信息究竟贡献了什么。当前新网络同时把原41维主干从64宽改为16宽，虽与FACTOR严格匹配，却不能把相对旧DIRECT的退步单独归因于配对表示；“加了更多输入”与“保住原来能力”需要分开研究。',
        '下一步应先制定一个保留原主干的模块对照方案，并明确以完整补查净效用、正收益漏失与无利查询共同判读。损失是否改、模块是否加，应分别控制；不同时换骨干、目标、费用和训练量，不靠挑阈值或第552步通过来救结果。具体方案需在训练前说明，本轮没有执行任何新增训练。','',
        '## 可复核材料','',
        '- endpoint_metrics.json：全部新旧模型、训练/验证的原目标、漏停来源和输出偏差。',
        '- decision_loss_crosscheck.json：全训练收益分组、单步决策遗憾与大收益漏失清单。',
        '- sequence_objective_check.json、sequence_diagnostic_rows.json：完整序列复算及事后上界。',
        '- gradient_decomposition.json：固定权重下原加权目标的正负梯度分解。',
        '- representation_diagnostics.json、binding_sensitivity.json：表示、注意力与单次配对敏感性。',
        '- review.json、input_manifest.json：独立标量复算、NumPy注意力核验和未改变的来源哈希。']
    (O/'GRID114_report.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    save(O/'decision.json',dict(no_missing_positive_exposure=True,stop_output_not_joint_miss_cause=True,regression_loss_improves_while_positive_misses_increase=True,
        train_one_step_weighted_regret_improves=True,train_complete_utility_joint=jtrain,train_complete_utility_direct=dtrain,
        validation_complete_utility_joint=jval,validation_complete_utility_direct=dval,validation_no_oracle_need_for_long_horizon=True,
        pairing_sensitivity_weak_for_this_single_perturbation=True,no_unique_causal_failure_identified=True,ready_to_scale=False,
        next='Define preservation of original backbone and modular matched comparison before any fit; do not simultaneously change loss and architecture.',new_fits=0,new_physical_calls=0))
    contextpaths=[R/f'outputs/{d}/{f}' for d,f in [('grid103_learning_mechanism','GRID103_report.md'),('grid101_gain_learning','GRID101_report.md')]]
    save(O/'context_source_manifest.json',{p.relative_to(R).as_posix():sha(p) for p in contextpaths})
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID114_fit_diagnosis/complete';collab=R/'论文协作资料/FRAME_学习失败核对_GRID114'
    archive.mkdir(parents=True,exist_ok=False);collab.mkdir(exist_ok=False)
    save(O/'complete.json',dict(passed=True,scope='Read-only fixed-endpoint diagnosis, not a trained improvement.',new_fits=0,optimizer_instances=0,new_physical_calls=0,
        archive=str(archive),collaboration=str(collab),report=str(O/'GRID114_report.md'),ready_to_scale=False,final_test_materialized=False))
    for p in O.iterdir():
        if p.is_file():shutil.copy2(p,archive/p.name)
    for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
    for name in ['GRID114_report.md','decision.json','review.json','complete.json']:shutil.copy2(O/name,collab/name)
    save(archive/'archive_manifest.json',{p.name:sha(p) for p in archive.iterdir() if p.is_file()})
    note='## 2026-10-23 — GRID114 read-only fit/decision diagnosis\n\nAll4438 outputs and6fixed weights,zero fits/optimizer/physical calls/test. Positive samples all drawn8-171times under original7250 budget; no missing exposure. JOINT all train/val false stops have qcontinue<=0,not stop-head inflation. JOINT original weighted Huber lower than DIRECT all3seeds and even positive-risk contribution lower,while positive misses rise. Positive residuals still89.7-92.5%ofloss,so not fully ignored. Train one-step weighted sign regret improves all3,complete train family utilityJOINT0.12252524 versusDIRECT0.12355273 (mixedseeds); validation0.14842746vs0.16292381,so previous blanket train-underfit framing refined: tradeoff/transfer and objective mismatch all matter. At zero score Huber intercept gradient+0.0042059 while weighted mean net+0.0304872; descriptive direction mismatch,not proof replacing loss helps (GRID108MSE adverse retained). Weak JOINT actual pairing sensitivity ononefixedcyclicperturbation,no positive sign flips;nearuniformattention,not causal nonuse proof. Zero exact next-action embeddingcollisions,unique3582headinputs. Validationmyopiclabeloracle equalsfutureprefixceiling0.19070118,no hidden bridge cases;train23correlatedprefixes. No need to invoke longer planning to explain current validation failure;GRID103gamma1failure preserved. Independent53256scalarchecks/144NumPyattentionstates/complete sequencecheckpassed. Next define preserved-original-backbone modular control before fitting; no automatic network/loss/budget/scalingchanges.'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID114只读诊断完成\n\n1. 汇报修正后的原因：正收益漏停增加并不等于整体训练损失更差；训练单步权衡改善而验证更差，配对通道决策作用弱。\n2. 下一轮先定义保留原41维主干的模块对照；损失与结构分开控制，训练前向用户说明具体方案。\n3. 当前不训练、不扩大、不做电网回放或最终测试；不重做已有不稳定的MSE/长训练/gamma1对照。\n\n报告：research_archive/power-grid-2026/protocols/GRID114_fit_diagnosis/complete/GRID114_report.md\n',encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID114学习失败核对](protocols/GRID114_fit_diagnosis/complete/GRID114_report.md)：零训练，核对目标与正收益决策、配对敏感性及训练/验证完整序列，修正笼统训练不足解释。\n')
    print('GRID114_AUDITED_AND_PUBLISHED',flush=True)
if __name__=='__main__':independent_review();publish()
