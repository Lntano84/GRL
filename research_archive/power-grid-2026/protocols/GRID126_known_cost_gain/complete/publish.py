"""Publish all paired endpoints and failure accounting without selecting a model."""
from common126 import *

def main():
    assert not (O/'complete.json').exists();verify();assert read(O/'review.json')['passed'];D=read(O/'design.json');F=read(O/'fits.json');dec=read(O/'decision.json');metric=read(O/'state_metrics.json');div=read(O/'divergence_summary.json');family={};decomp={};profiles={}
    for dom in ['validation','oof']:
        rr=read(O/f'{dom}_results.json');family[dom]={arm:{k:float(np.mean([rr[f'{arm}_seed{s}']['all']['macro'][k] for s in range(3)])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility']} for arm in ALL};decomp[dom]={}
        for arm in ARMS:
            a,b=family[dom][arm],family[dom]['NET_ZERO'];dg=(a['mean_gain']-b['mean_gain'])/D['gain_scale'];dc=-D['cost_per_query']*(a['mean_queries']-b['mean_queries'])/D['gain_scale'];dv=a['mean_net_utility']-b['mean_net_utility'];assert abs(dg+dc-dv)<1e-12;decomp[dom][arm]=dict(gain_contribution=dg,query_charge_contribution=dc,net_difference=dv)
    for scope in ['full','folds']:
        profiles[scope]={}
        for arm in ARMS:
            rows=[r for r in F['records'] if r['arm']==arm and ((r['split']=='full')==(scope=='full'))];profiles[scope][arm]={stage:{k:float(np.mean([r[stage+'_fit_profile'][k] for r in rows])) for k in rows[0][stage+'_fit_profile']} for stage in ['initial','final']}
    save(O/'summary_analysis.json',dict(family_metrics=family,gain_and_query_accounting=decomp,fit_profiles=profiles,criteria=dec))
    labels=dict(NET_ZERO='原净收益回归',GAIN_DIRECT='直接回归改进量，再扣已知费用',COST_POSITIVE='平滑非负改进量，再扣已知费用')
    lines=['# GRID126：已知费用与非负改进量学习对照','',
      '**30次固定训练、全部保存结果核验完成。原15份净收益回归权重复用；本轮没有新电网回放或最终测试。**','',
      '## 研究问题与对照','',
      'GRID125发现，少数大收益机会承载大部分正净收益，且一些预测低于已知查询费下界。本轮区分两件事：让模型预测不确定的奖励改进，把可精确计算的费用放到模型外；以及对改进量施加非负结构，是否能改善停止决策。',
      '三组均为原41→64→64→2网络、6978参数。原回归NET_ZERO使用主输出预测净收益；GAIN_DIRECT使用主输出预测归一化奖励改进G，再扣已知费用c。两组新训练的损失均为0.5×Huber(预测改进,G)+0.5×Huber(另一输出,0)，beta=1，没有辅助分类或正例重抽样。',
      'COST_POSITIVE使用同一网络输出r，定义预测改进g(r,c)=r+sqrt(r²+c²)，预测净收益为g−c。负r分支采用等价稳定写法，避免相消误差。该链接不引入参数，满足g≥0；r=0时g=c、净收益为0，净收益对r的导数为1；继续条件严格等价于r>0。',
      '因此，约束组与原回归有相同的潜在动作边界和零点附近斜率。它改变的是训练中的非线性参数化和梯度，不是给旧权重套一个公式就获得新动作。直接改进回归组的初始动作函数与原回归不同，本轮如实保留初始训练侧诊断，不声称三者初始函数处处相同。',
      '只对原净收益预测和目标同时加上已知费用，Huber损失及梯度完全相同；这作为零效应对照检查。把正链接用于原权重的56298个保存评分时，动作也全部不变。不能把这样的代数变换包装成改进。','',
      '## 固定训练与评价','',
      f'新增训练30次：完整16训练日期上每组3个种子共6次，四个12日期训练/4日期留出折共24次。每次7250更新×32状态，原初始化张量、逐批抽样序列、Adam、梯度裁剪、数据、费用、原排序和控制器保持不变。总训练耗时{F["wall_s"]:.2f}秒，没有延长1800秒上限。',
      '全部30个固定终点完成后才计算新模型的验证/留出预测。原32训练/8验证场景副本、16/4日期不变，前缀相关；折内标准化只用12个训练日期。费用和尺度沿用历史全16日期任务定义，所以折外评价是条件开发诊断，不是新的独立测试。',
      '主指标是完整局部补查搜索的日期宏平均净效用，已经扣除原代理查询费。查询比例、即时收益保留率、调用均衡效用、每日期及全部种子均保留。代理费用不是实际秒数或电费；没有物理整周质量、存活或提速结论。','',
      '## 结果：两个评价域分别看','']
    for dom,label in [('validation','原四个验证日期'),('oof','原16训练日期的折外评价')]:
        lines+=['### '+label,'']
        for arm in ALL:
            v=family[dom][arm];lines.append(f'- {labels[arm]}：净效用{v["mean_net_utility"]:.8f}；查询比例{100*v["query_fraction"]:.3f}%；收益保留{100*v["gain_capture"]:.3f}%；调用均衡效用{v["physical_call_balanced_utility"]:.8f}。')
        for arm in ARMS:
            c=dec['comparisons'][dom][arm+'_minus_NET_ZERO'];v=decomp[dom][arm];lines += [f'- {arm}相对原回归：三个配对种子差{[round(x,8) for x in c["paired_seed_differences"]]}，正向日期{c["positive_dates"]}/{4 if dom=="validation" else 16}；预定方向条件={c["signal"]}。',f'  收益项{v["gain_contribution"]:+.8f}，查询费用项{v["query_charge_contribution"]:+.8f}，净差{v["net_difference"]:+.8f}。']
            for date,diff in c['date_differences'].items():lines.append(f'  - {date}：{diff:+.8f}。')
        c=dec['comparisons'][dom]['COST_POSITIVE_minus_GAIN_DIRECT'];lines+= [f'- 非负链接相对直接改进回归：种子差{[round(x,8) for x in c["paired_seed_differences"]]}，正向日期{c["positive_dates"]}/{4 if dom=="validation" else 16}；方向条件={c["signal"]}。','']
    lines+=['## 已保存轨迹能解释哪些变化','',
      '下面覆盖每条实际共同前缀的首次分歧，没有选择好案例。下一批标签只用来事后分组；效用贡献包含从分歧到各自搜索结束的整个已保存尾部，不能把尾部差额归因于一条反馈，更不是整周物理控制因果效果。记录共享前缀和种子，调用数跨组可重叠。','']
    for dom in ['validation','oof']:
        for arm in ARMS:
            lines.append(f'- {dom} / {arm}：')
            for group,v in div[dom][arm].items():lines.append(f'  - {group}：{v["records"]}条记录、{v["distinct_physical_calls"]}次调用，效用贡献{v["utility_contribution"]:+.8f}。')
    lines+=['','## 训练与参数化诊断','']
    for scope in profiles:
        for arm in ARMS:
            p=profiles[scope][arm];a,b=p['initial'],p['final'];lines.append(f'- {scope}/{arm}：初始→终点主Huber {a["main_huber"]:.8f}→{b["main_huber"]:.8f}；训练侧遗憾{a["sign_regret"]:.8f}→{b["sign_regret"]:.8f}；初始继续比例{a["continue_mass"]:.4f}；终点正收益状态链接导数均值{b["positive_target_jacobian_mean"]:.6f}，最小导数{b["minimum_jacobian"]:.6g}。')
    lines += ['','全体状态的主回归和决策指标：']
    for arm in ALL:
        for part in ['train','validation']:
            rs=[metric[f'full__{arm}_seed{s}'][part] for s in range(3)];lines.append(f'- {arm}/{part}：主Huber {np.mean([v["main_huber"] for v in rs]):.8f}；遗憾{np.mean([v["sign_regret"] for v in rs]):.8f}；正净收益误停数{[v["positive_misses"] for v in rs]}；无利继续数{[v["unprofitable_continues"] for v in rs]}。')
    lines += ['','非负链接在零点与原净收益参数化的一阶行为一致，保留原Huber也不等于已经消除了GRID125展示的混合分布均值/稳健损失差异。它是一种可核验的结构对照，不是期望净收益一致性证明。若结果变差，所有梯度敏感性、稀有正收益误停和反向种子均保留，不按本轮结果更换链接或调尺度。','',
      '## 预定判读','',
      f'相对原回归的双域增量条件：{dec["incremental_support"]}；非负链接相对直接回归的特异条件：{dec["positive_link_specific_support"]}；超过原验证历史参照：{dec["legacy_anchors_exceeded"]}；组合开发条件：{dec["useful_candidate"]}。',
      '每个新组都要求两个评价域的均值高于原回归、三个配对种子均正向，且至少3/4验证日期和12/16折外日期改善。非负链接的独立作用需要相对直接改进回归也满足同样的方向条件。平均更高但种子/日期不稳时，不能按最好的种子采用。',
      '历史原验证参考：'+ '，'.join(f'{k}={v:.8f}' for k,v in dec['legacy_validation'].items())+'。这些是保留的历史参照，没有在本轮重训，不能据此声称所有训练过程完全匹配。',
      '本轮全部结果属于已暴露开发数据上的监督学习，不是新增强化学习。未自动扩大数据或运行电网闭环，最终8场景保持未生成；既有MSE、代价加权分类、辅助事件学习的不利结果继续保留。','',
      '## 核验与归档','',
      '独立NumPy复算全部新模型前向、非负链接值域与动作等价关系，核对原初始化、6978参数、7250次Adam更新、每个采样序列哈希与次数、留出日期零梯度抽样、4438个改进标签与公共查询费用，以及全部保存局部搜索轨迹的动作/成本/选中动作与宏平均。核对结果见review.json。所有权重、曲线、初始/终点诊断及失败方向保留。']
    (O/'GRID126_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');c=R/'论文协作资料/FRAME_已知费用收益学习_GRID126';c.mkdir(exist_ok=False);save(O/'complete.json',dict(passed=True,completed_fits=30,review_passed=True,report=str(O/'GRID126_report.md'),archive=str(A/'complete'),collaboration=str(c),ready_to_scale=False,new_physical_calls=0,final_test_materialized=False));archive('complete')
    for fn in ['GRID126_report.md','summary_analysis.json','decision.json','review.json','complete.json']:shutil.copy2(O/fn,c/fn)
    docs(f'## 2026-10-23 — GRID126 known-cost/gain learning complete\n\n30fixedfits6978/7250x32/originalinitandstreams,15baselinesreused,wall{F["wall_s"]:.2f}s. Families={dec["family_utility"]};incremental={dec["incremental_support"]};positivelinkspecific={dec["positive_link_specific_support"]};candidate={dec["useful_candidate"]}. IndependentNumPy/optimizer/exposure/publiccharge/trajectoryauditpassed. Initial/finalJacobianandfirst-commonprefixdifferencesretained,no seed/linkrescue. Exposedteacherlocal-searchdevelopmentonly,not newRL/wholeweekspeed/independenttest;no physical/dataexpansion/finaltest.',
      '# Next steps — GRID126已知费用收益学习完成\n\n1. 按全部种子与双评价域汇报改进量回归和非负参数化的实际结果。\n2. 结合已保存误停尾部、收益/查询分解及训练导数解释，不择种子或自动加训救结果。\n3. 根据已确认效果再确定下一设计，不自动扩规模、运行物理回放或开启最终测试。\n\n报告：research_archive/power-grid-2026/protocols/GRID126_known_cost_gain/complete/GRID126_report.md\n')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID126已知费用与非负改进量学习](protocols/GRID126_known_cost_gain/complete/GRID126_report.md)：30次固定匹配训练、已知费用外置/非负参数化、完整局部搜索与全种子核验。\n')
    print('GRID126_COMPLETE_ARCHIVED',flush=True)

if __name__=='__main__':main()
