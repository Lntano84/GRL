"""Paper mechanism draft and scientific figures from frozen GRID60/61 data."""
import csv,hashlib,json,math,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid61_mechanism';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mean=lambda v:math.fsum(v)/len(v)
def main():
    if OUT.exists():
        assert (OUT/'render_environment_failure.json').exists()
        assert not (REPO/'research_archive/power-grid-2026/protocols/GRID61_mechanism').exists()
        retained={p.name:sha(p) for p in OUT.iterdir() if p.name in ['mechanism_values.json','框架方法与机制工作稿.md']}
    else:
        OUT.mkdir();retained={}
    R=read(ROOT/'outputs/grid61/review.json');Q=read(ROOT/'outputs/grid60/qualification.json');assert R['passed_saved_data'] and Q['passed_engineering']
    with (ROOT/'outputs/grid61/per_run.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ix={(int(r['repeat']),r['week'],r['method']):r for r in rows};weeks=list(dict.fromkeys(r['week'] for r in rows))
    s={r['method']:r for r in R['summaries']}
    assert s['FRAME']['quality_pass']==8 and s['NO_LOCAL']['quality_pass']==6 and s['FIRST_SAFE_BEST']['quality_pass']==4
    assert R['verdict']=='LOCAL_COMPLETION_SUPPORTED_IN_MATRIX'
    first_gain=mean([1-float(ix[rep,w,'FRAME']['controller_s'])/float(ix[rep,w,'FIRST_SAFE_BEST']['controller_s']) for rep in (0,1) for w in weeks])
    completion_speed_tradeoff=s['NO_LOCAL']['macro_control_reduction_full']-s['FRAME']['macro_control_reduction_full']
    march_gain=float(ix[0,weeks[0],'FRAME']['local_cost_gain_of_full'])
    september_first_advantage=float(ix[0,weeks[2],'FRAME']['cost_delta_full'])-float(ix[0,weeks[2],'FIRST_SAFE_BEST']['cost_delta_full'])
    assert march_gain>.04 and september_first_advantage>0
    values={'frame_control_reduction_full':s['FRAME']['macro_control_reduction_full'],
        'frame_query_reduction_full':s['FRAME']['macro_query_reduction_full'],
        'frame_control_reduction_first_safe':first_gain,'completion_controller_tradeoff_full_points':completion_speed_tradeoff,
        'march_completion_cost_gain_full':march_gain,'september_first_safe_cost_advantage_full':september_first_advantage,
        'new_fits':0,'new_forecasts':0,'new_rollouts':0}
    (OUT/'mechanism_values.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
    table='|策略|质量通过|对FULL控制减少|对FULL查询减少|最坏完整周成本变化/FULL|\n|---|---:|---:|---:|---:|\n'
    for a in R['summaries']:table+=f"|{a['method']}|{a['quality_pass']}/8|{a['macro_control_reduction_full']:.2%}|{a['macro_query_reduction_full']:.2%}|{a['worst_complete_cost_delta_full']:+.5%}|\n"
    text=f'''# 框架为什么有效：方法与机制工作稿

本文档依据GRID60只读资格与GRID61实际闭环。用户已选择框架为论文主线，GNN作为可替换排序组件。原版、质量补强版、频次先验版及全部反例保留。本稿没有新增实验或训练。

## 当前可支持的核心命题

在相同作者频次排序、相同候选库、相同基础连续控制和共有缓存下，反馈驱动的渐进搜索配合变电站补全，能在本轮四日期中守住预设质量尺度，同时减少大量预测查询与实际控制时间。作用无需依赖本轮新GNN或RL训练。

{table}

每策略3/4周完成，两个重复合计6/8；六月所有方法均在1901步失败。质量通过还要求无新增提前失败、完整周成本不超过FULL的0.5%。质量8/8不能写成完成8/8。两个计时重复不是两份独立日期数据。

## 方法：按反馈扩展，并保留局部结构完整性

基础连续控制、恢复、断线处理及作者动作库沿用现有系统。排序可使用原作者频次、PPO或GNN，本文重点不要求GNN最强。

1. 根据当前公开状态构造可执行候选，屏蔽不能设置的断线端点，并对当前实际动作向量的等价候选规范化。
2. 按给定排序查询首批128个候选。只接纳预测未结束、无异常/非法/歧义且最大负荷率严格改善的候选，以原fp32奖励及固定ID破并列。
3. 若首批优胜候选负荷率已低于作者既有0.9阈值，则结束。否则每批再查询64个：发现低于0.9的候选，或在至少256次查询后连续两批未改善当前奖励，则停止继续遍历其他批次。无可接纳候选时遍历至候选池耗尽。
4. 对需要补查的情形，在结束前查询当前优胜候选所属变电站中尚未查询的全部可执行代表，更新优胜候选，然后交回原控制器执行。

128/64/256/2是冻结的搜索日程，不是论文新定理，也不是硬时间上限。当前方法最坏仍可能查询完整候选池。不能称其拥有严格实时截止保证。

## 一个可以证明的单步性质

记已查询且可接纳的候选集合为C，当前优胜候选为a_C，原公开动作库中与其变电站对应的未查询代表为L。将选择键定义为f(a)=(fp32奖励,−动作ID)，按字典序比较。

在局部补全被触发的条件下，补全后的选择为C与L中可接纳候选的最大键，因此f(a_after)不小于f(a_C)。额外查询数恰为未查询的局部代表数，每个代表只查询一次，总查询不超过当前规范化候选池大小。

这说明补全不会降低同一当前状态中的已观测选择键，并给出明确的附加计算范围。它不证明整周成本不增加、不保证其他变电站没有更好候选，也不证明电网在未知未来扰动下绝对安全。安全阈值内的首批直接停止情形不自动获得局部全覆盖性质。

## 机制一：节省来自渐进搜索

NO_LOCAL相对FULL查询少71.84%、控制少14.25%，但三月成本增幅4.81017%，超预设尺度。这个对照说明简单地更早停止确实能节约计算，也能丢掉重要行动。不能仅以查询减少宣布框架成功。

FIRST_SAFE_BEST来自已有“按排序模拟直到首次安全候选”的思想；当无安全候选时，我们还加强为保留完整遍历中最佳可接纳动作，而非原论文的DoNothing回退。它的最坏成本差3.95164%，质量4/8；控制只少7.03%。因此本矩阵中，普通首个安全停止未替代现有框架。

## 机制二：结构补全修复部分停止损失

FRAME只比NO_LOCAL增加所属变电站补查，排序/其余停止规则/基础控制/缓存均相同。三月成本增幅从4.81017%降至0.37200%，差{march_gain:.3%}以FULL成本归一化。其余完整周成本相同；六月共同失败、成本不作为成功收益。故本轮观察到的质量修复集中在三月，不能称所有日期普遍改善。

更细的单状态证据：已知1733步中，NO_LOCAL查询256次选动作26；局部补全额外查询198次，用454次找到与909次完整搜索相同的动作309。这解释的是一次结构性漏选及其恢复，不是对全周损失来源的完整分解。

FRAME对FULL仍省{ s['FRAME']['macro_query_reduction_full']:.2%}查询、{s['FRAME']['macro_control_reduction_full']:.2%}控制。局部补全使控制提速比NO_LOCAL少{100*completion_speed_tradeoff:.2f}个FULL归一化百分点，却守住了本轮质量尺度。这是测得的质量—计算权衡，不是所有组分相加的因果贡献。

## 不利结果与边界

FIRST_SAFE_BEST在九月成本更低，比FRAME低出的成本相当于FULL的{september_first_advantage:.3%}；FRAME没有逐日期、逐指标全面支配。十月该参照成本仅高FULL0.51914%，靠近0.5%门槛，应报告原数而不渲染成严重失败。六月共同失败保留。

FRAME相对FIRST_SAFE_BEST的配对宏平均控制减少{first_gain:.2%}，这是四个既有开发日期的测量。并非对完整SoftGNN代理或其原论文成绩的胜出证明。

## 机制三：共有编译复用

GRID29C/30/31已有真实参数和记忆条件下的数值、同路径与组件计量：DC及安全QP复用减少重复建模/编译，GRID31三月两重复完整控制减少23.49%/21.33%，汇总22.42%，交付向量和成本相同。这是清楚的工程机制，但属于常规可复用优化，所有适用对照都享有；不作为新求解算法或GNN独有贡献。

GRID61比较是在所有臂已经有相同缓存的条件下，测搜索策略的作用。不能把22.42%和13.06%相加宣称35.48%整体提速，日期、参照、版本和分母不同。

## 与已有工作的具体关系

[LJN官方实现](https://github.com/lajavaness/l2rpn-2023-ljn-agent)已有连续与拓扑混合控制、动作库和NN Top-k模块。这些基础组件需要承认来源。

[SoftGNN 2025，第4.4节](https://arxiv.org/html/2503.15190v2#S4.SS4)已有排序后逐候选仿真、首个0.9以内候选停止、断线端点屏蔽及Top10 N-1扩展。物理验证、首个安全停止或断线掩码不能各自作为我们的首创。

[Generalizable GNNs，第4.9节](https://arxiv.org/html/2501.07186v1#S4.SS9)已有预测验证和贪心/N-1回退混合方案。我们的待确认贡献应具体到“渐进反馈停止与变电站语义补全如何取得质量—计算权衡”，而非笼统声称模型结合搜索/回退就是创新。

检索与源码对照只建立差异候选，没有穷尽文献，也没有认证首创。本轮已有足够具体的机制实验，可围绕该框架继续写方法和实验，而不必证明GNN结构突破。

## 论文贡献的建议写法

提出可使用不同排序器的渐进拓扑搜索框架，以已实际获得的仿真反馈决定扩展，并在停止前保持当前优胜变电站的候选完整性。分析补全的单步选择性质和附加查询范围，通过同排序闭环消融展示其对停止损失与实际控制成本的作用。编译复用作为共享系统实现单列，所有质量、存活、查询和完整控制时间均按统一口径公开。

应避免“认证安全”“全局最优”“GNN造成全部加速”“全面胜过SoftGNN”和“已证明CCF B录用”。八日期旧主表继续保留，本轮四日期机制结果不替代或扩充其独立样本数。
'''
    (OUT/'框架方法与机制工作稿.md').write_text(text,encoding='utf-8')
    for name,h in retained.items():assert sha(OUT/name)==h,'Figure repair must not change existing numerical report.'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    labels=['FULL','First-safe\n+ best fallback','No local\ncompletion','Framework'];colors=['#8997A5','#75A6C9','#D5A556','#294D70']
    summaries=R['summaries'];fig,axes=plt.subplots(1,3,figsize=(12.4,4.2))
    for ax,field,title in zip(axes,['macro_query_reduction_full','macro_control_reduction_full','worst_complete_cost_delta_full'],['N1 query reduction','Controller time reduction','Worst complete-week cost increase']):
        values=[100*r[field] for r in summaries];bars=ax.bar(labels,values,color=colors,width=.63)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(values)*.025,f'{v:.2f}%',ha='center',va='bottom',fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_ylabel('Relative to FULL (%)');ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(0,max(values)*1.22)
        if field=='worst_complete_cost_delta_full':ax.axhline(.5,color='#7E3136',linestyle='--',lw=1);ax.text(.05,.55,'0.5% screen',fontsize=9,color='#7E3136')
    fig.suptitle('Common-order framework ablation: four old development weeks, two timing repeats',fontsize=12)
    fig.text(.02,.015,'All policies share ranking, base control and DC/safe-QP caches. One common failed week retained; cost uses complete weeks.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(OUT/'framework_mechanism.png',dpi=170);fig.savefig(OUT/'framework_mechanism.pdf');plt.close(fig)
    sources=[ROOT/'outputs/grid61/review.json',ROOT/'outputs/grid61/per_run.csv',ROOT/'outputs/grid60/qualification.json',ROOT/'outputs/grid60/per_state.csv',ROOT/'outputs/grid31/review.json',ROOT/'outputs/grid31/GRID31_report.md',Path(__file__)]
    (OUT/'sources.json').write_text(json.dumps([{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sources],indent=2),encoding='utf-8')
    dest=REPO/'research_archive/power-grid-2026/protocols/GRID61_mechanism';assert not dest.exists();dest.mkdir();receipts=[]
    for p in [Path(__file__),Path(__file__).with_name('framework_mechanism_v0.py')]+list(OUT.iterdir()):
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)
        receipts.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    (dest/'manifest.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print(json.dumps({'written':True,'mechanism':values}))
if __name__=='__main__':main()
