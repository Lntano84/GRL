"""Selective archive and evidence-based report for the auxiliary-task comparison."""
import argparse
from common123 import *

def docs(note,next_text):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_text,encoding='utf8')

def archive(phase):
    a=REPO/f'research_archive/power-grid-2026/protocols/GRID123_event_auxiliary/{phase}';a.mkdir(parents=True,exist_ok=False)
    for p in W.glob('*.py'):shutil.copy2(p,a/p.name)
    for p in O.iterdir():
        if p.is_file():shutil.copy2(p,a/p.name)
    if phase=='complete':shutil.copytree(O/'models',a/'models')
    save(a/'archive_manifest.json',{p.relative_to(a).as_posix():sha(p) for p in a.rglob('*') if p.is_file()});return a

def before():
    verify();assert read(O/'preflight.json')['passed'];archive('before')
    docs('## 2026-10-23 — GRID123 same-capacity auxiliary-event learning authorized and frozen\n\nUser approves net-regression plus raw-gain-event auxiliary. Compare AUX_GAIN vs AUX_NETSIGN with same6978architecture/init/7250x32/sampler/Adam/data/fee;mainHuberhalf coefficient retained,otherheadBCEhalf,zero-decisionreadoutforall. NET_ZERO uses old15trainedmodels withq1>0;originalq1>q0legacyseparatelyretained. New30fits full6+fourfold24,original16/4datesand12/4groupfolds,allfixedendpointsbeforeevaluation,1800s cap. No tuning/physical/test/newdata/networkexpansion. Train438rawgainevents/211netpositive;labelsnotonlineinputs. Initial/finalscheduled-minibatch sharedgradientdiagnosticsinsidebudget,not extra updates. Equal taskweightsfixednotclaimedoptimal.',
        '# Next steps — GRID123辅助监督对照已封存\n\n1. 完成30次固定训练，全部结束后统一评价原验证日期和原训练日期折外结果。\n2. 核对原容量/初始化/抽样、主损失系数、标签隔离、精确零读法及辅助输出不参与动作。\n3. 汇报辅助监督与原始改进标签各自的作用；不择种子或加训救结果，不自动物理扩大或生成测试。\n')
    print('GRID123_BEFORE_ARCHIVED',flush=True)

def complete():
    assert not (O/'complete.json').exists();verify();assert read(O/'review.json')['passed'];D=read(O/'design.json');F=read(O/'fits.json');dec=read(O/'decision.json');metrics=read(O/'state_metrics.json')
    keys=['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility'];family={};decomp={}
    for domain in ['validation','oof']:
        result=read(O/f'{domain}_results.json');family[domain]={a:{k:float(np.mean([result[f'{a}_seed{s}']['all']['macro'][k] for s in range(3)])) for k in keys} for a in ['NET_ZERO']+ARMS};decomp[domain]={}
        for ref in ['NET_ZERO','AUX_NETSIGN']:
            a,b=family[domain]['AUX_GAIN'],family[domain][ref];gain=(a['mean_gain']-b['mean_gain'])/D['gain_scale'];cost=-D['cost_per_query']*(a['mean_queries']-b['mean_queries'])/D['gain_scale'];net=a['mean_net_utility']-b['mean_net_utility'];assert abs(gain+cost-net)<1e-12;decomp[domain][ref]=dict(gain=gain,query_charge=cost,net=net)
    gradient=[]
    for r in F['records']:
        name=f"{r['split']}__{r['arm']}_seed{r['seed']}";rows=read(O/f'gradient_components_{name}.json')
        gradient.append(dict(name=name,**{f'first_{k}':v for k,v in rows[0].items() if k!='update'},**{f'last_{k}':v for k,v in rows[1].items() if k!='update'}))
    save(O/'summary_analysis.json',dict(family_metrics=family,gain_aux_minus_references=decomp,gradient_components=gradient,scope='Read-only accounting; scheduled mini-batch gradient norms are diagnostic,not global training-cause proof.'))
    lines=['# GRID123：净收益回归加改进事件辅助监督','',
        '**30次新训练和保存结果核验完成。网络容量、数据和费用保持不变；辅助输出没有直接参与补查动作。**','',
        '## 比较什么','',
        'NET_ZERO是原收益回归模型，沿用既有权重，将停止价值直接取精确0。AUX_NETSIGN同时学习净收益和“净收益是否为正”；AUX_GAIN同时学习净收益和“原始增益是否为正”。第二组用于区分普通辅助监督的作用与原始改进事件这一标签的作用。',
        '三组都是原41→64→64→2网络、6978参数。新模型复用原先拟合固定停止价值的输出做辅助分类，没有增加输出或模块；主输出继续预测原归一化即时净收益。动作统一为主输出>0继续，其他情况停止，辅助分类概率不作为触发条件。',
        '新损失为0.5×Huber(主输出,净收益)+0.5×BCE(辅助输出,事件标签)。原损失为两输出Huber的平均，因此主任务的0.5系数保留。两个辅助任务的权重预先固定相同，不扫描系数或调正例比例。BCE与Huber的数值/梯度尺度不同，等权不等于贡献相同，也不代表这个权重最优。',
        '为避免把改变读法当成学习收益，报告同时保留原模型q1>q0的历史读法；新增模型与NET_ZERO的比较始终采用相同的q1>0读法。原模型只复用，不重复训练。','',
        '## 数据与训练','',
        '原训练数据的原始改进事件有438个状态、来自93次调用；扣费后净收益为正的有211个、来自54次调用。原始增益为正不意味着值得查询，最终仍按净收益主输出决策。标签来自原教师反馈，只用于训练/评价，不进入模型输入。',
        f'新增训练共30次：完整16日期训练各辅助任务三个种子，共6次；四个12日期训练/4日期留出折各训练两个辅助任务、三个种子，共24次。原15份回归权重复用。总训练耗时{F["wall_s"]:.2f}秒，原1800秒总窗口没有延长。',
        '每次7250次更新、每批32状态，原初始化、逐批抽样、Adam和梯度裁剪保持一致。各折沿用训练侧标准化，留出日期不参与梯度；全部30个固定终点完成后才统一计算验证和折外结果。费用与收益尺度沿用历史任务；原16日期对任务常数的历史依赖仍保留，折外结果不是新独立测试。',
        '本轮没有新的物理回放、数据生成、安装、最终测试或论文图表修改。','',
        '## 完整补查结果','',
        '净效用已扣查询费用，越高越好；比例先按日期计算再宏平均。两个评价域分别汇报，不能把不同训练日期数/评价日期数的绝对值混合比较。']
    for domain,label in [('validation','原四个验证日期'),('oof','原16训练日期的折外评价')]:
        lines+=['',f'### {label}','']
        for arm in ['NET_ZERO']+ARMS:
            m=family[domain][arm];lines.append(f'- {arm}：净效用{m["mean_net_utility"]:.8f}；查询比例{100*m["query_fraction"]:.3f}%；收益保留{100*m["gain_capture"]:.3f}%；调用均衡净效用{m["physical_call_balanced_utility"]:.8f}。')
        readout=dec['readout_change'][domain];lines.append(f'- 原模型历史q1>q0读法净效用{readout["original_q1_vs_q0"]:.8f}；改为精确零读法的变化{readout["difference"]:+.8f}。')
        for ref,cmp in dec['comparisons'][domain].items():
            lines.append(f'- AUX_GAIN减{ref}的三个配对种子差：{[round(x,8) for x in cmp["paired_seed_differences"]]}；正向日期{cmp["positive_dates"]}/{4 if domain=="validation" else 16}；预定同向条件={cmp["signal"]}。')
            dc=decomp[domain][ref];lines.append(f'  收益项{dc["gain"]:+.8f}，查询费用项{dc["query_charge"]:+.8f}，净差{dc["net"]:+.8f}。')
        lines+=['','逐日期差值（AUX_GAIN减NET_ZERO）：']
        for date,x in dec['comparisons'][domain]['NET_ZERO']['date_differences'].items():lines.append(f'- {date}：{x:+.8f}。')
    lines+=['','历史原四日期参照：'+ '，'.join(f'{k} {v:.8f}' for k,v in dec['legacy_validation'].items())+'。','',
        '## 误停与梯度诊断','']
    for arm in ['NET_ZERO']+ARMS:
        for part in ['train','validation']:
            mm=[metrics[f'full__{arm}_seed{s}'][part] for s in range(3)]
            lines.append(f'- 完整训练{arm}，{part}：正净收益误停状态数{[m["positive_misses"] for m in mm]}；无利继续数{[m["unprofitable_continues"] for m in mm]}；层次抽样加权单步遗憾{np.mean([m["weighted_sign_regret"] for m in mm]):.8f}。')
    lines+=['','以下梯度只来自第1和7250次正式更新的小批次，已经计入训练预算；不能代表全部训练过程或证明唯一原因。完整记录见summary_analysis.json。']
    for arm in ARMS:
        gg=[g for g in gradient if g['name'].startswith('full__'+arm)]
        lines.append(f'- {arm}，三个完整训练种子首步辅助/主任务共享梯度范数比{[round(g["first_aux_over_main"],3) if g["first_aux_over_main"] is not None else None for g in gg]}；末步比值{[round(g["last_aux_over_main"],3) if g["last_aux_over_main"] is not None else None for g in gg]}。')
    lines+=['','## 预定判读','',
        f'辅助任务增量支持={dec["incremental_auxiliary_support"]}；原始改进事件相对净收益正负辅助的特异支持={dec["event_specific_support"]}；超过原验证全部历史参照={dec["legacy_anchors_exceeded"]}；组合开发条件={dec["useful_candidate"]}。',
        '增量支持要求AUX_GAIN在两个评价域都高于NET_ZERO家族均值、三个配对种子均为正，并至少3/4验证日期和12/16折外日期同向。事件特异支持对AUX_GAIN相对AUX_NETSIGN采用相同方向要求。若只有平均提高、但种子或日期不稳，必须按原值保留，不能挑一组结果证明辅助标签有效。',
        '即使原始改进事件优于另一个辅助任务，也不一定优于原回归；反过来，若两种辅助任务相近，就不能将收益归给原始改进标签。更改辅助系数或模型需要另一个预先匹配的研究，不能根据本轮成绩自动救结果。','',
        '## 核验与边界','',
        '核验包括112596个新模型状态的独立NumPy前向、4438个原目标与改进事件标签、30份优化器7250步、原初始化和逐批曝光、留出日期零梯度抽样、相同零停止读法、完整保存轨迹及新旧读法变化。所有不利结果、权重及来源哈希保留。',
        '本轮仍是同一电网、已暴露开发日期、固定FULL教师路径和即时局部搜索任务。前缀相关，原费用/尺度有历史任务定义依赖；没有整周质量、实际时延、统计显著或跨网结论。没有自动扩大、运行新电网闭环或生成最终8测试场景。']
    (O/'GRID123_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');collab=R/'论文协作资料/FRAME_改进事件辅助学习_GRID123';collab.mkdir(exist_ok=False);a=REPO/'research_archive/power-grid-2026/protocols/GRID123_event_auxiliary/complete'
    save(O/'state.json',dict(stage='complete',fits=30,review_passed=True));save(O/'complete.json',dict(passed=True,completed_fits=30,review_passed=True,report=str(O/'GRID123_report.md'),archive=str(a),collaboration=str(collab),ready_to_scale=False,new_physical_calls=0,final_test_materialized=False));archive('complete')
    for n in ['GRID123_report.md','decision.json','summary_analysis.json','review.json','complete.json','fits.json']:shutil.copy2(O/n,collab/n)
    docs(f"## 2026-10-23 — GRID123 same-capacity event auxiliary comparison complete\n\nAll30fits7250x32,original6978/init/sampler/Adam/splits/price,netHuberhalf+auxBCEhalf,zero-readoutforall. Full6fits+fold24;old15regressionweightsreused. Wall{F['wall_s']:.2f}s. Families={dec['family_utility']};readoutchange={dec['readout_change']};incremental={dec['incremental_auxiliary_support']},eventspecific={dec['event_specific_support']},legacyanchors={dec['legacy_anchors_exceeded']},candidate={dec['useful_candidate']}. Independent112596NumPyforwards/targets/optimizer/exposure/sequencesauditpassed;alladverseweights/dates/seedsretained. Auxlabelneverdirectlycontrolsaction;equalhalfweightsnotoptimalorgradientscalematch. Exposedteacherpath/conditionalOOFonly,no physical/test/autoscaling.",
        '# Next steps — GRID123辅助监督训练与核验完成\n\n1. 汇报辅助任务增量、原始改进标签的独立作用，以及原四日期和16日期折外的一致性。\n2. 结合误停、查询费和共享梯度解释，不择种子/加训/调系数补救本轮。\n3. 是否继续修改学习目标据实决定，不自动新电网回放、扩规模或生成最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID123_event_auxiliary/complete/GRID123_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID123净收益回归与改进事件辅助监督](protocols/GRID123_event_auxiliary/complete/GRID123_report.md)：同容量两种辅助标签、30次固定训练、15份原回归复用及独立保存结果核验。\n')
    print('GRID123_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args();before() if a.phase=='before' else complete()
