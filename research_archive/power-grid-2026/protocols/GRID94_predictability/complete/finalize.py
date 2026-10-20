"""Report all fixed fits and errors; archive without training or deploying a gate."""
from pathlib import Path
import json,shutil,datetime
from data import ROOT,W,OUT,read,save,sha,csvsave

KEY='GRID94_SKIP_PREDICTABILITY_complete'


def main():
    review=read(OUT/'review.json');numeric=read(OUT/'numerical_review.json')
    assert review['passed'] and numeric['passed']
    freeze=read(OUT/'code_freeze.json');assert all(sha(ROOT/p)==h for p,h in freeze['files'].items())
    ms=read(OUT/'metrics.json');ps=read(OUT/'predictions.json');ds=read(OUT/'decision.json')
    def metric(ev,model,threshold):return next(r for r in ms if r['evaluation']==ev and r['model']==model and r['cohort']=='ALL' and r['threshold']==threshold)
    errors=[{k:r[k] for k in ['evaluation','model','row_id','cohort','week','step','probability','completion_queries','reward_gain','role','source']}
        for r in ps if r['model'] in ['LOGISTIC','TREE2'] and r['probability']>=.95 and not r['label_skip_same_choice']]
    csvsave(OUT/'conservative_errors.csv',errors)
    report=['# GRID94：何时不启动同站补查，能否从已有反馈预测？',
        '', '2026-10-20。冻结提交3c9f4cd8。仅两个预先固定的小模型；没有新仿真、强化学习、控制器修改或门控部署。',
        '', '## 结论', '',
        '**目前有弱的排序信号，但还没有可靠的不启动判据。** 两种模型的留日期AUC约0.62。在固定0.95阈值下，逻辑回归仍误删改变动作的补查；浅层树全部保留，不能节省查询。旧场景到生成场景的检验也未改变这一判断。不加模型、改阈值或补样本来挽救本轮结果。',
        '', '本轮首先检验的是“能否识别不会改变选择的补查”。它不是对“会让整周变贵的补查”的预测。后者尚缺逐次补查的对照标签；原G2七月成本反例仍未被解决。',
        '', '## 1. 训练答案是什么', '',
        '原补查前后返回完全相同的代表动作，标为1：在该保存状态上省略补查不会改变搜索返回动作。返回不同动作，标为0：暂时保留补查，但不声称这个变化有益。使用原float32奖励和精确破并列规则，不用一个奖励幅度阈值代替动作一致。',
        '', '179次实际FRAME调用中，109次原本启动补查；其余70次不进入模型。109次来自17个日历日期组：77次返回原动作，32次改变动作。原状态上的补查查询共6457次，其中2869次属于返回原动作的补查，占44.43%；这是全知标签下的局部潜力，不是模型或新策略已经实现的节省。',
        '', '|来源|补查样本|返回原动作|改变动作|日历日期组|', '|---|---:|---:|---:|---:|',
        '|old13|69|45|24|11|','|G1|15|13|2|5|','|G2|25|19|6|5|',
        '', 'G1/G2日历日期重合，合并后为6个日期组；全部数据已暴露，不是新的独立测试。没有把185个FULL状态复制进训练集增加样本量，也没有把整周成本较差的标签分发给每次调用。',
        '', '## 2. 如何避免记住场景', '',
        '固定15项补查前特征：当前最佳预测ρ、当前改善限值、前缀情况、全局核查的改善、已观察最佳差距、已查询比例、本站剩余比例、合法候选比例、站内反馈支持等。日期、步数、场景名、站点编号、动作编号和补查后结果均不作为输入。前处理只在训练部分拟合。',
        '', '两个模型均在拟合前固定：C=1的逻辑回归；最大深度2、叶节点至少10样本的决策树。训练时每个日历日期总权重相同。没有超参数或特征筛选。',
        '', '检验一：每次留出一个完整日历日期，G1/G2同日期种子同时留出，共17折。检验二：仅用old13的69次调用训练，检查G1/G2的全部40次调用。两个模型、两个固定阈值和全部误判均报告，不根据结果选择最优模型。',
        '', '## 3. 预测表现', '',
        '|检验|模型|AUC|平均精确率AP|不改变动作比例|', '|---|---|---:|---:|---:|']
    for ev,evname in [('GROUPED_OOF','留日期'),('OLD_TO_GENERATED','旧场景→生成场景')]:
        for model in ['LOGISTIC','TREE2','TRAIN_PRIOR']:
            r=metric(ev,model,.95)
            report.append(f"|{evname}|{model}|{r['roc_auc']:.4f}|{r['average_precision']:.4f}|{100*r['prevalence']:.2f}%|")
    report += ['', 'AUC接近0.5表示区分能力有限，约0.62只能视为弱信号，17组数据不足以据此宣称稳定泛化。留组TRAIN_PRIOR的先验会随留出的组变化，所以其AUC并不固定为0.5；旧到新检验的训练先验为常数，AUC为0.5。',
        '', '## 4. 重点看误删，而不是只看准确率', '',
        '主阈值为预测“返回原动作”的概率至少0.95。模型概率没有经过额外校准；0.95不是实际正确率的保证。',
        '', '|检验|模型|建议取消|其中误删动作变化|原状态名义省查询|其中同动作标签正确部分|',
        '|---|---|---:|---:|---:|---:|']
    for ev,evname in [('GROUPED_OOF','留日期'),('OLD_TO_GENERATED','旧→生成')]:
        for model in ['LOGISTIC','TREE2']:
            r=metric(ev,model,.95)
            report.append(f"|{evname}|{model}|{r['skipped_calls']}|{r['missed_choice_changes']}|{r['nominal_queries_saved']}|{r['same_choice_queries_saved']}|")
    report += ['', '这些查询数字只是在原FRAME状态上逐次计算。如果某次误删改变了返回动作，后续状态就可能改变，不能把表中总和称为闭环实际节省。遗漏动作变化也不是已经证明运行质量下降，两者不能混为一谈。',
        '', '更宽松的0.5阈值同样完整保留：', '', '|检验|模型|建议取消|误删动作变化|准确率|始终取消的准确率|', '|---|---|---:|---:|---:|---:|']
    for ev,evname in [('GROUPED_OOF','留日期'),('OLD_TO_GENERATED','旧→生成')]:
        base=metric(ev,'ALWAYS_SKIP',.5)
        for model in ['LOGISTIC','TREE2']:
            r=metric(ev,model,.5)
            report.append(f"|{evname}|{model}|{r['skipped_calls']}|{r['missed_choice_changes']}|{100*r['accuracy']:.2f}%|{100*base['accuracy']:.2f}%|")
    report += ['', '类别比例造成明显陷阱：完全不看特征、每次都取消，留日期准确率就有70.64%，生成组为80%。两个模型在0.5下的准确率没有超过该简单基线；高准确率本身不能证明学会了可靠门控。',
        '', '## 5. 必须保留的误判', '', '|检验|场景与控制步|模型预测可取消概率|补查实际奖励增益|查询数|', '|---|---|---:|---:|---:|']
    for e in errors:
        report.append(f"|{e['evaluation']}|{e['week']}，步{e['step']}|{e['probability']:.4f}|{e['reward_gain']:.8f}|{e['completion_queries']}|")
    report += ['', '以上是在0.95主阈值下的全部误删，没有选取某个阈值绕开它们。判读NO_DEMONSTRATED_CONSERVATIVE_SKIP_SIGNAL。两项检验的结果不是两份独立数据，也不合并计算样本量。',
        '', '## 6. 用户关心的“有害补查”还没有学到', '',
        'G2七月2036-07-07_10第113步，是此前完整策略成本反向的首次分歧。它确实改变了动作，所以在本轮标签里属于“保留”，而不是“无效”。留日期时两模型给它的可取消概率约为0.3841和0.1232；旧到生成时约为0.3282和0.2096，0.95和0.5均保留。',
        '', '**这揭示了标签的边界：学习“有没有找到不同动作”，无法自动学会“这个不同动作是否值得采用”。** 本轮不能声称模型成功回避整周成本反例，也不能由此否定学习真正补查价值的可能。',
        '', '若继续推进，优先补可靠的价值标签，而不是换更大的网络：在相同物理状态、控制器记忆和后续策略下，对补查与不补查做有限分支对照。先区分运行风险、成本和计算代价，保留正、零、负收益；再检验模型能否预测这些价值。当前每候选完整预测状态与逐次分支答案都不足，尚未启动这批数据采集。',
        '', '## 7. 核验与材料', '',
        '原FRAME179调用重现；109条精确标签与隐藏反馈扰动通过。36个分类器拟合（17折×2模型加2次迁移拟合），36项训练/测试日期与物理状态隔离检查、298个独立概率复算、745条预测来源与70项指标计数核验通过。没有最终全数据部署模型。',
        '', '拟合中既有加权StandardScaler对近常量列发出sqrt方差警告，最终归一化尺度由常量处理设为1。额外使用长双精度中心化方差独立核对18个归一化器，所有尺度、均值、系数与概率有效；没有重训、换运行时或修改协议。依赖由既有运行时引用的历史包目录提供，不代表重启历史LG01训练。',
        '', '原方案、194项冻结哈希、数据、全部36次拟合参数与树规则、逐条概率、按日期与来源指标、保守误判、数值说明均保存。核验入口review.json、numerical_review.json，判读见decision.json。未新增独立场景、预测调用、闭环运行、强化学习或控制器权重。']
    report_text='\n'.join(report)+'\n';(OUT/'GRID94_report.md').write_text(report_text,encoding='utf-8')
    save(OUT/'artifact_receipt.json',dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['complete.json','artifact_receipt.json']},
        postfit_readonly_scripts={p.name:sha(p) for p in [W/'numerical_check.py',Path(__file__)]}))
    repo=ROOT/'work/GRL-results-integration';dst=repo/'research_archive/power-grid-2026/protocols/GRID94_predictability/complete';dst.mkdir(exist_ok=False)
    for p in list(OUT.iterdir())+list(W.glob('*.py')):
        if p.is_file():shutil.copy2(p,dst/p.name);assert sha(p)==sha(dst/p.name)
    collab=ROOT/'论文协作资料/FRAME_补查不启动可预测性_20261020';collab.mkdir(exist_ok=True)
    for n in ['GRID94_report.md','decision.json','review.json','data_summary.json','metrics.json','conservative_errors.csv']:
        shutil.copy2(OUT/n,collab/n);assert sha(OUT/n)==sha(collab/n)
    note='已完成固定两小模型的无效补查可预测性诊断。109样本/17日期，标签为返回同一动作77次，改变动作32次；15项补查前可见特征，按日期留组与old13到G1/G2。AUC约0.62；0.95阈值逻辑回归留组取消11误删3、迁移取消6误删1，浅层树均不取消。未建立可靠保守取消信号，原G2七月成本反例未解决。不追加模型/阈值，不部署，长期有害补查仍缺逐次分支价值标签；无新仿真/RL。'
    for name in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
        p=repo/name;s=p.read_text(encoding='utf-8');p.write_text(s.rstrip()+'\n\n## 2026-10-20 — '+KEY+'\n\n'+note+'\n',encoding='utf-8')
    p=repo/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines();p.write_text(lines[0]+'\n\n- **'+KEY+'：** '+note+'\n\n'+'\n'.join(lines[1:]).lstrip()+'\n',encoding='utf-8')
    p=repo/'docs/NEXT_STEPS.md';s=p.read_text(encoding='utf-8');lines=s.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('1. '):
            lines[i]='1. GRID94固定可预测性诊断完成：简单模型尚不能可靠取消无效补查；没有部署。下一步先汇报无效标签与长期有害标签的区别。如继续，优先明确有限分支价值采集的状态、完整控制器记忆、后续策略和风险/成本目标；不直接加模型、扫阈值或做整周标签广播。';break
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p=repo/'research_archive/power-grid-2026/README.md';s=p.read_text(encoding='utf-8');p.write_text(s.rstrip()+'\n\n## '+KEY+'\n\n'+note+'\n\n[报告](protocols/GRID94_predictability/complete/GRID94_report.md)。\n',encoding='utf-8')
    complete=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,
        classifier_fits=36,new_forecasts=0,new_closed_loop_runs=0,new_RL=0,result=ds['result'],
        deployment=False,report_sha256=sha(OUT/'GRID94_report.md'),archive=str(dst),collaboration_copy=str(collab))
    save(OUT/'complete.json',complete);shutil.copy2(OUT/'complete.json',dst/'complete.json');shutil.copy2(OUT/'complete.json',collab/'complete.json')
    print(json.dumps(complete,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
