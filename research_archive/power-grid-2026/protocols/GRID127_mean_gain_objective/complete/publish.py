"""All proper-mean loss endpoints and adverse outcomes;no selected seed or new fit."""
from common127 import *

def main():
    assert not (O/'complete.json').exists();verify();assert read(O/'review.json')['passed'];D=read(O/'design.json');F=read(O/'fits.json');dec=read(O/'decision.json');metric=read(O/'state_metrics.json');div=read(O/'divergence_summary.json');family={};account={};profiles={}
    for dom in ['validation','oof']:
        rr=read(O/f'{dom}_results.json');family[dom]={arm:{k:float(np.mean([rr[f'{arm}_seed{s}']['all']['macro'][k] for s in range(3)])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility']} for arm in ALL};account[dom]={}
        for arm in ARMS:
            a,b=family[dom][arm],family[dom]['NET_ZERO'];dg=(a['mean_gain']-b['mean_gain'])/D['gain_scale'];dc=-D['cost_per_query']*(a['mean_queries']-b['mean_queries'])/D['gain_scale'];net=a['mean_net_utility']-b['mean_net_utility'];assert abs(dg+dc-net)<1e-12;account[dom][arm]=dict(gain=dg,query_charge=dc,net=net)
    for scope in ['full','folds']:
        profiles[scope]={}
        for arm in ARMS:
            rr=[r for r in F['records'] if r['arm']==arm and ((r['split']=='full')==(scope=='full'))];profiles[scope][arm]={stage:{k:float(np.mean([r[stage+'_fit_profile'][k] for r in rr])) for k in rr[0][stage+'_fit_profile']} for stage in ['initial','final']}
    # All15 existing positive-Huber endpoints,fit-only gradient arithmetic at identical saved scores.
    before=np.load(BASE/'predictions.npz');fixed=[]
    for split in ['full','f0','f1','f2','f3']:
        x,y,G,c,ids,p=getdata(split)
        for seed in range(3):
            r=before[f'{split}__COST_POSITIVE_seed{seed}__train'][ids,1].astype(float);cost=c[ids];g=numpy_gain(r,cost,'POS_HUBER');h=np.hypot(r,cost);jac=1+r/h;err=g-G[ids];bad=(y[ids]>0)&(r<=0)
            grad=dict(HUBER=.5*np.clip(err,-1,1)*jac,MSE=err*jac,KL=.5*err/h)
            fixed.append(dict(split=split,seed=seed,profitable_fit_misses=int(bad.sum()),miss_mass=float(p@bad),gradient_abs_on_profitable_fit_misses={k:float(p[bad]@abs(v[bad])/p[bad].sum()) if bad.any() else None for k,v in grad.items()}))
    save(O/'frozen_huber_gradient_diagnostic.json',dict(posthoc_descriptive=True,records=fixed,scope='Analytic main-output derivatives at frozen existingHuberfittingscores,no updates or held-side gradients;not proof of historical optimization cause.'))
    save(O/'summary_analysis.json',dict(family_metrics=family,gain_and_query_accounting=account,fit_profiles=profiles,decision=dec))
    labels=dict(NET_ZERO='原净收益Huber回归',POS_HUBER='相同非负链接的Huber（复用）',POS_MSE='相同非负链接的平方损失',POS_KL='相同非负链接的广义KL')
    lines=['# GRID127：同一非负收益链接下的均值目标对照','',
      '**完成30次固定新训练及独立核验；原净收益回归15份和GRID126非负Huber15份结果直接复用，没有重复拟合。**','',
      '## 为什么继续这一组','',
      'GRID126中，非负Huber模型训练侧的继续比例降至约1.8%，漏掉较多正收益机会。本轮不换输入、网络或非负链接，只比较损失，检验是否能让低估的大收益样本获得更有效的修正，并改善完整补查决策。',
      '所有新组使用原41维输入、6978参数、原初始化、已知费用c及g(r,c)=r+sqrt(r²+c²)。净收益预测为g−c，继续条件为r>0。GAIN_DIRECT的既有失败结果也保留，但不在本轮重复训练。','',
      '## 两种新目标分别学什么','',
      '- POS_MSE：主损失0.5×(g−G)²。',
      '- POS_KL：主损失0.5×[g−G+G(logG−logg)]，其中G=0时GlogG定义为0；计算logg=logc+asinh(r/c)，避免接近零时的数值问题。',
      '另一输出仍按原0.5×Huber拟合停止价值0，没有辅助分类。G是归一化的非负连续奖励改进，不要求为整数；广义KL不是对数据服从泊松分布的假设。',
      '在固定公开上下文中，对任意正预测g，广义KL总体风险导数为1−E[G|上下文]/g，因此其最小值对应条件均值；平方损失也对应同一个均值。这个总体性质不保证有限网络在现有数据上一定更准。',
      '对于本轮链接，dg/dr=g/sqrt(r²+c²)。KL主损失对r的导数为0.5×(g−G)/sqrt(r²+c²)；相比Huber先限制大误差再乘链接导数，严重低估正收益时可以产生更大的修正信号。梯度大小和裁剪次数按实记录，不能把相同更新次数说成梯度尺度也相同。','',
      '## 固定训练与判读','',
      f'共30次新训练：两种损失×3个种子×完整训练及4个日期留出折。每次7250更新×32状态，Adam学习率0.0003及裁剪10不变，逐批抽样顺序逐项重现。总耗时{F["wall_s"]:.2f}秒，未延长1800秒上限。',
      '两个新组与既有非负Huber的链接、随机初始张量及初始预测函数相同。全部新训练终点完成后才评价验证/留出日期，没有按中途结果选种子、步数、链接或系数。',
      '原32训练/8验证场景副本、16/4日期、4个12训练/4留出折不变。历史费用和尺度定义依赖全16训练日期，日期也已经暴露，因此折外是条件开发诊断，不是新的独立测试。即时收益与代理费用只刻画局部补查，不代表物理整周质量或速度。','',
      '## 完整补查结果','']
    for dom,label in [('validation','原四验证日期'),('oof','原16训练日期折外')]:
        lines+=['### '+label,'']
        for arm in ALL:
            m=family[dom][arm];lines.append(f'- {labels[arm]}：净效用{m["mean_net_utility"]:.8f}；查询比例{100*m["query_fraction"]:.3f}%；收益保留{100*m["gain_capture"]:.3f}%；调用均衡效用{m["physical_call_balanced_utility"]:.8f}。')
        for key,cmp in dec['comparisons'][dom].items():
            lines.append(f'- {key}：三个配对种子差{[round(x,8) for x in cmp["paired_seed_differences"]]}，正向日期{cmp["positive_dates"]}/{4 if dom=="validation" else 16}，预定方向条件={cmp["signal"]}。')
            for date,diff in cmp['date_differences'].items():lines.append(f'  - {date}：{diff:+.8f}。')
        lines.append('')
    lines+=['## 收益与查询费用分别贡献多少','']
    for dom in account:
        for arm,m in account[dom].items():lines.append(f'- {dom}/{arm}相对原净收益回归：收益项{m["gain"]:+.8f}，查询费用项{m["query_charge"]:+.8f}，合计净差{m["net"]:+.8f}。')
    lines+=['','## 误停及训练梯度诊断','']
    for arm in ALL:
        for part in ['train','validation']:
            rr=[metric[f'full__{arm}_seed{s}'][part] for s in range(3)];lines.append(f'- {arm}/{part}：正净收益误停{[v["positive_misses"] for v in rr]}；无利继续{[v["unprofitable_continues"] for v in rr]}；主Huber诊断误差{np.mean([v["main_huber"] for v in rr]):.8f}；单状态遗憾{np.mean([v["sign_regret"] for v in rr]):.8f}；继续比例{np.mean([v["continue_mass"] for v in rr]):.4f}。')
    for scope,arms in profiles.items():
        for arm,pp in arms.items():
            aa,bb=pp['initial'],pp['final'];lines.append(f'- {scope}/{arm}：本损失训练目标{aa["main_objective"]:.8f}→{bb["main_objective"]:.8f}；正净收益样本主输出梯度绝对值均值{aa["positive_main_gradient_abs"]:.6f}→{bb["positive_main_gradient_abs"]:.6f}；大收益样本对应值{aa["large_positive_main_gradient_abs"]:.6f}→{bb["large_positive_main_gradient_abs"]:.6f}。')
    lines+=['','主Huber在此只用于跨组描述数值误差，不是MSE或KL组的训练目标。不同损失的数值和梯度单位不应直接比较为“谁收敛得更好”。初始与终点曲线/范数完整保留。',
      '在已完成的15份非负Huber权重上，还按相同训练状态解析计算三种主损失对输出的梯度，文件frozen_huber_gradient_diagnostic.json保留全部结果。它没有产生新更新，只用于核对梯度机制；该补充是事后描述性分析，不冒充独立验证。','',
      '完整搜索首次共同前缀的分歧记账：']
    for dom,arms in div.items():
        for arm,gg in arms.items():
            for category,v in gg.items():lines.append(f'- {dom}/{arm}/{category}：{v["records"]}条前缀—种子记录、{v["distinct_physical_calls"]}次调用；完整保存搜索尾部效用贡献{v["utility_contribution"]:+.8f}。')
    lines+=['','这些尾部差值逐项守恒，但同一调用及前缀相关，不能把记录数当独立场景。下一批标签仅用于事后分组，不是对整个尾部效应的单步因果解释。','',
      '## 预定结论与范围','',
      f'超过原净收益回归的双域条件：{dec["incremental_support"]}；超过相同链接Huber条件：{dec["vs_positive_huber_support"]}；KL相对平方损失的特异条件：{dec["kl_specific_support"]}；超过历史验证参照：{dec["legacy_anchors_exceeded"]}；组合开发条件：{dec["useful_candidate"]}。',
      '方向条件要求两个评价域均值提高、三个配对种子均正向，且至少3/4验证日期与12/16折外日期改善。总体均值、单种子或个别日期好看都不能替代这些条件。',
      '历史原验证参考：'+'，'.join(f'{k}={v:.8f}' for k,v in dec['legacy_validation'].items())+'。这些不是本轮新训练。',
      '本轮属于监督学习，没有新增强化学习、数据生成、电网环境、预测/物理回放或最终8测试场景；没有修改论文和图表。均值目标的数学性质不等于模型已实现该性质，也不等于整周经济或安全保证。既有MSE、加权分类、辅助学习和GRID126失败结果均保留。','',
      '## 核验','',
      '独立NumPy复算新模型前向、非负链接与动作关系；逐项核对原容量/初始化、7250次Adam、原抽样序列与次数、留出日期零抽样、4438个标签与已知费用，以及全部局部搜索动作/成本/选中动作和日期宏平均。两个复用模型家族的保存评分逐项相同，原回归及非负Huber结果分别精确重现。详见review.json。']
    (O/'GRID127_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');c=R/'论文协作资料/FRAME_均值收益目标学习_GRID127';c.mkdir(exist_ok=False);save(O/'complete.json',dict(passed=True,completed_fits=30,reused_models=30,review_passed=True,report=str(O/'GRID127_report.md'),archive=str(A/'complete'),collaboration=str(c),ready_to_scale=False,new_physical_calls=0,final_test_materialized=False));archive('complete')
    for fn in ['GRID127_report.md','GRID126_127_阶段汇总.md','two_stage_synthesis.json','summary_analysis.json','decision.json','review.json','complete.json']:shutil.copy2(O/fn,c/fn)
    docs(f'## 2026-10-23 — GRID127 fixed-link mean objectives complete\n\n30fixedPOS_MSE/POS_KLfits,30cachedNET/POS_HUBERcontrols,original6978/init/link/7250x32/Adam/clip/streams/task,wall{F["wall_s"]:.2f}s. Families={dec["family_utility"]};vsNET={dec["incremental_support"]};vsPOSHuber={dec["vs_positive_huber_support"]};KLspecific={dec["kl_specific_support"]};candidate={dec["useful_candidate"]}. IndependentNumPy/updates/exposure/targets/trajectoryauditpassed;alladverseweights/seeds/datespreserved. GeneralizedKLproperconditionalmeanpopulationtargetnotPoissondataassumptionorfinitefitguarantee. No physical/newdata/test/RL/autoscaling.',
      '# Next steps — GRID126/127两组收益学习对照完成\n\n1. 汇报已知费用、非负链接和均值损失的各自结果，区分训练修正与留出决策效果。\n2. 用全部种子、日期、查询/收益分解确定研究方向；不按最优种子采用，不自动调损失参数或加训。\n3. 未达到双域稳定条件时保留原回归；达到时先拟定有限物理闭环资格方案，不直接扩数据或开启最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID127_mean_gain_objective/complete/GRID127_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID127同一非负链接的均值收益损失](protocols/GRID127_mean_gain_objective/complete/GRID127_report.md)：30次固定MSE/KL训练、30份既有模型复用、全种子与条件折外局部补查核验。\n')
    print('GRID127_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':main()
