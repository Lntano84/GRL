"""Finalize a bounded research design,not fitted/deployed gate results."""
from pathlib import Path
import collections,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def main():
    C.verify();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();pools=read(O/'fit_support.json')['pools'];rows=read(O/'fit_stop_rows.json');case={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'}
    meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};checks=0
    for pool in pools:
        f=pool['fold'];s=pool['seed'];pp=[v for v in rows if v['fold']==f and v['seed']==s];assert len(pp)==pool['eligible'];assert {v['date'] for v in pp}<=set(pool['fit_dates']);z=np.load(C.sources()[f]/'state_scores.npz');q=z[f'FACTOR_seed{s}'];cnt=collections.Counter(v['physical_state_group'] for v in pp);gd=collections.defaultdict(set)
        for v in pp:gd[v['date']].add(v['physical_state_group'])
        for v in pp:
            c=case[v['sample_id']];ii=[lookup[c['sample_id'],j] for j in range(len(c['states']))];actions=q[ii,1]>q[ii,0];stops=np.flatnonzero(~actions);assert len(stops) and v['position']==int(stops[0]);j=v['position'];assert v['metadata_index']==ii[j]
            assert v['private_label']['next_action_changes']==(c['chosen'][j]!=c['chosen'][j+1]);assert v['private_label']['next_gain']==c['gains'][j]
            mass=1/(len(gd)*len(gd[v['date']])*cnt[v['physical_state_group']]);assert abs(v['proposed_sampling_mass']-mass)<1e-15;checks+=1
        assert pool['positive_action_changes']==sum(v['private_label']['next_action_changes'] for v in pp)
    spec=dict(status='Proposed next fit,not executed in GRID189',scope='12fixed gate probes=4existing folds x3existing NETseeds. No originalNETretraining. Everynewgate fits corresponding12datesonly;all12gates published before helddate feature extraction/scoring.',
        architecture='Reuse frozenNET mainlastReLU64 and correctionheadlastReLU16,plus currentq0,q1 and knownnextquerycost1=83public features; one logistic readout84trainableparameters. Extract with read-only hooks on originalforward;qualification must reproduce originalscores and unchangedparameters. No new grid/queryfeatures.',
        training='Weighted LogisticRegression(C=1,penalty=L2,solver=lbfgs,max_iter=2000),fixed0.5 threshold. Features standardized with fit-only date/call/prefix weights;constantfeature scale1. Sampleweight=Neligible*mass (mean1),no classbalance/positiveupweight/thresholdsearch. Gate seeds are attached to frozenNETseeds,not extra independent repeats.',
        label='Next single batch changes selected rank/ID. Continuous nextgain,NET and full-localchoice are private outcomes only for evaluation. Unlike GRID187,not predicting full suffix with a one-batch action.',
        population='Only first STOP states of frozenNET with actual remaining candidates. FitSTOPselection based on in-sample frozenNET;deployment conditionaldistribution may shift. Stop selection must not be made using helddate outcomes.',
        budget='Exactly12 convex gate fits at fixed configuration;not authorized/executed by this design receipt. No continuationfits/newbaselineweights. Proposed totalfitcap600s,no automaticextension/restarts. Convergence failures kept.',
        inference='NET unchanged until first STOP;gate once only if candidates remain. p>0.5 buys next<=64 then returns best;else keeps NET. No furtherNET/gate,ranking/station/global search unchanged.',
        comparisons=['FrozenNET','UnconditionalADD1_at_same_NETSTOP','SelectiveADD1','oldBATCH1/FULL_LOCAL and existingauxiliary asseparate references'],
        metrics='All16oldtrain datesOOF/3seeds: addedreward,full-localactionmatch,addedqueries,NET utility,eligiblegate rate;date andphysicalgroup macros. Newonlinefeatures/inference/queries must count in laterreal timing. No newphysical claims here.',
        prior_evidence='Choice of conditional frozen gate follows GRID188. This is not proof it improves. Ordinarysharedauxiliary existed GRID123;not silently repeat it. Full-week cost mayworsen evenif savedlocalreward nondecreasing.')
    C.save('next_probe_spec.json',spec)
    lo=lambda k:min(v[k] for v in pools);hi=lambda k:max(v[k] for v in pools)
    lines=['# GRID189：原NET停止后的单批追加方案','',
        '方案和拟合侧样本核查已完成，尚未训练新门控。核心改动是把学习模块限制在原NET停止以后：保留它已经执行的搜索，只判断是否额外购买一批候选。这样把“找回小改善”与“撤销原本有价值的继续”分开。','',
        '## 方法具体怎么运行','',
        '1. 原NET使用冻结权重、原候选顺序和原全局搜索运行，直到第一次决定停止。','2. 若同站候选已查完，直接返回，不调用新门控。','3. 若仍有候选，新门控仅判断一次。停止则返回原NET动作；追加则评估下一批至多64个候选，按原可接纳条件和奖励/编号规则返回已观察最佳动作。','4. 不进行第二次门控，也不把追加后的状态再交回NET。','',
        '因此，新门控不会提前终止原NET，也不会修改原NET参数；它只处理原NET已经准备放弃的机会。','',
        '## 能严格说明的性质','',
        '固定当前物理状态、基础命令、初始锁定站、候选代表与别名、原排序及评价规则。设原NET购买了b批，尚未评估的下一批为C，其大小m≤64。门控决定e∈{0,1}，则：','',
        '- 查询：Q_ext = Q_NET + e·m，因此Q_NET ≤ Q_ext ≤ Q_FULL_LOCAL，额外查询至多64。','- 选择：原已购集合V保留，新增集合为C，返回max_rank(V∪eC)。因此已观察最佳奖励不下降；相同奖励时仍沿用较小代表编号。','- 动作：若原NET已选中完整同站搜索的最终动作，它在V∪eC中仍是最优，因而不会被追加改变。在这个固定候选问题上，对完整同站动作的一致指示不会下降。','',
        '这些性质不依赖门控预测准确。它们来自保留原搜索、只加入候选；不保证新的物理控制轨迹整周成本更低或始终存活。追加后可能选出不同控制动作，长期效果仍需真实闭环检验。','',
        '## 计算代价要正面说明','',
        '这是质量补强方案，不是声称比原NET更快。单个相同状态上，原NET推理和查询路径保持，另有一次门控及至多64次候选查询。缓存复用、特征抽取和门控时间均须计入后续在线测试。与完整同站搜索的候选查询仍有上述上界，但实际控制总时间不能由查询数量直接推出。','',
        '若要节省原NET自己的计算，需要另一种问题定义；本轮不同时做两种方向。是否值得追加，最终要看奖励/动作改善、额外查询和真实控制时间的取舍，原NET效用不保证上升。','',
        '## 与以前试验的区别','',
        '|旧路线|本方案的区别|','|---|---|',
        '|GRID123共享事件辅助头|冻结原价值模型，新门控只控制一次可选追加；不通过共享梯度改变原继续决定|',
        '|GRID126/127替换收益目标或损失|保留原NET目标与权重，新学习问题发生在它首次停止之后|',
        '|GRID184双向共识|不能把NET的继续改成停止；不让另一个模型接管全部搜索|',
        '|GRID167单向提前停|方向相反：只允许后加一批，不允许提前删掉原查询|',
        '|原固定BATCH1|原固定BATCH1总共只查一批；这里先完成NET自己的所有批次，再考虑一批|','',
        '## 拟合侧的学习机会','',
        f"原4折×3种子共12个条件数据池。每池只使用对应12个拟合日期；有{lo('eligible')}–{hi('eligible')}个可追加起点，下一批会改选的有{lo('positive_action_changes')}–{hi('positive_action_changes')}个，分布于{lo('positive_groups')}–{hi('positive_groups')}个物理调用组。每池两类均存在，不是只有零收益。",'',
        f"下一批按原查询价格计算NET为正的只有{lo('positive_NET')}–{hi('positive_NET')}个。因此，追回动作变化不等于经济上回本；连续增量和NET会单独保留。门控标签只对应下一批，不把完整后缀会改变的答案贴到每一批上。",'',
        '按日期→物理调用组→该组起点等权构造候选训练权重，不按正例重加权。不同折和不同NET种子会多次使用同一物理调用，不能把总行数当作独立样本。原NET在拟合侧学得更好，筛出的STOP分布可能比部署时容易；这点不能用拟合侧覆盖替代日期留出验证。','',
        '|折|NET种子|可追加起点|会改选|正例物理组|NET为正|','|---|---:|---:|---:|---:|---:|']
    for v in pools:lines.append(f"|{v['fold']}|{v['seed']}|{v['eligible']}|{v['positive_action_changes']}|{v['positive_groups']}|{v['positive_NET']}|")
    lines+=['','## 下一轮最小学习试验的具体形式','',
        '候选门控使用冻结NET最后的64维主干表示、16维反馈分支表示、当前两个价值输出与已知下一批费用，共83维；只学习一个84参数的逻辑回归读出。原模型表示不反向更新。先核验表示抽取不改变原分数或决定，再做任何拟合。','',
        '固定L2、C=1、lbfgs最多2000迭代、阈值0.5；标准化与权重只由本折拟合日期计算，不扫描特征、类别权重或门槛。12个门控分别附着于原4折×3种子的NET，全部最终权重先封存上传，再计算留出日期的门控特征与结果。候选训练总窗口600秒，未开始。','',
        '原NET网络仍是已有学习主体，这个门控只检验其停止表示中是否还能分出被遗漏的单批机会；不是把整个控制器退回线性模型。','',
        '## 必须有的对照','',
        '除了原NET，必须加入“原NET停止后无条件追加一批”。两者使用相同追加位置和候选。若学习门控只是买更多查询，它未必比这个简单方法更值得用；要同时报告各日期/种子的增量、动作一致、追加查询、NET效用和推理开销。旧辅助模型与原固定BATCH1继续作为分别标注的参照，不只与较弱的纯二元新目标比较。','',
        '允许小而重复的改善，不增加全日期全赢或5%门槛；也不把形式化的局部不退化等同于已训练门控有效。','',
        '## 本轮做了什么、还没做什么','',
        f"核对了21402个拟合起点问题，{checks}条可追加记录的首次STOP、下一批标签与分组权重另行复核；每个可追加状态的停止/追加两条分支都核对查询和已观察奖励边界。只读取已有预测和教师标签，没有模型前向、优化器更新或新的电网预测。",'',
        '没有计算留出日期上的新门控策略成绩，没有挑选学习门槛；最终8未生成。next_probe_spec.json记录下一轮候选试验，fit_stop_rows.json把私有标签与公开记账字段分开保存。当前完成的是方案和样本资格，不能说门控已经有效。']
    (O/'GRID189_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('audit.json',dict(passed=True,independent_first_STOP_label_and_weight_rows=checks,both_gate_branches_qualified=read(O/'review.json')['checks']['both_gate_branch_bounds'],all12_pools_have_both_classes=True,scheme_not_fitted=True,new_fits=0,new_model_forwards=0,new_physics=0,held_new_strategy_scores=0))
    C.save('complete.json',dict(passed=True,stage='design_and_fit_support_only',ready_for_feature_qualification=True,gate_fitted=False,fit_pools=12,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True));C.verify();C.archive('complete')
    dest=R/'论文协作资料/FRAME_NET停止后单批追加方案_GRID189';dest.mkdir(exist_ok=False)
    for n in ['GRID189_report.md','design.json','next_probe_spec.json','fit_support.json','review.json','audit.json','complete.json']:shutil.copy2(O/n,dest/n);assert C.sha(O/n)==C.sha(dest/n)
    msg=f'GRID189方案/拟合侧资格完成：固定原NET运行至STOP，仅可追加下一批<=64后返回最佳，不撤销原继续；固定教师候选上查询介于NET与完整同站之间、已观察奖励/完整同站动作匹配不下降，但非整周成本或实际提速保证。12池各{lo("eligible")}–{hi("eligible")}起点、{lo("positive_action_changes")}–{hi("positive_action_changes")}改选/{lo("positive_groups")}–{hi("positive_groups")}物理组，NET正仅{lo("positive_NET")}–{hi("positive_NET")}，质量与费用分列。候选83维冻结表征+84参数LR、12fit、C1/L2/.5、先封存后held、ADD1_AT_STOP简单对照已定；尚未抽取表征/拟合/新策略held评分/物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID189原NET停止后的单批追加设计\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID189后\n\n1. 仅原NETSTOP后可选追加一批；方案和12池资格已完成。下一步先做冻结NET83维表征抽取与原分数/决定一致性核验，使用拟合日期，不能修改原价值权重。\n2. 准备12个84参数LR门控的固定训练(C1/L2/lbfgs2000/.5/600秒窗口)，拟合前通知具体范围；当前未拟合。全部门控先发布再做原16日期OOF，原4开发/最终8不动。\n3. BASE_F与原NETSTOP后无条件ADD1是主对照；旧辅助/原BATCH1分列。质量/收益幅度/额外查询与推理分开报告，不能只和弱二元模型比。局部单调性质非整周成本保证；不自动新物理/扩数据/论文。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID189单批追加设计\r\n\r\n[方案与样本资格](protocols/GRID189_baseline_extension/complete/GRID189_report.md)；保留原NET，只在其停止后可选多查一批，尚未新训练或仿真。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID189_COMPLETE',C.json.dumps(dict(eligible_range=[lo('eligible'),hi('eligible')],positive_range=[lo('positive_action_changes'),hi('positive_action_changes')],independent_rows=checks,new_fits=0)))
if __name__=='__main__':main()
