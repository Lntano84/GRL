"""Read-only interpretation after fixed learning; no fits, new scores or selection."""
from pathlib import Path
from collections import Counter
import datetime,hashlib,json,platform,sys
import numpy,scipy,sklearn
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert read(OUT/'review.json')['passed_saved_data'] and read(OUT/'learning_review.json')['independent_saved_export_audit']['passed']
    data=read(OUT/'labels.json');lookup={r['case_id']:r for r in data};pred=read(OUT/'predictions.json');results=read(OUT/'learning_results.json')
    groups={}
    for scheme in ['calendar_loo','old_to_generated']:
        for model in ['LOGISTIC','FOREST64']:
            rr=[r for r in pred if r['scheme']==scheme and r['model']==model and not r['pilot_known']]
            for group in ['all','identical_physical_path','different_physical_path']:
                gg=[r for r in rr if group=='all' or lookup[r['case_id']]['physical_path_identical']==(group=='identical_physical_path')]
                selected=[r for r in gg if r['prediction']=='SKIP' and r['probabilities'][1]>=.95]
                groups[f'{scheme}|{model}|{group}']=dict(n=len(gg),labels=dict(Counter(r['label'] for r in gg)),proposed=len(selected),selected_labels=dict(Counter(r['label'] for r in selected)),case_ids=[r['case_id'] for r in selected])
    save('posthoc_path_diagnostics.json',dict(scope='Descriptive stratification by realized physical outcome after fitting; never an input, sample selector or tuning criterion.',new_fits=0,new_predictions=0,groups=groups))
    save('learning_environment.json',dict(python=platform.python_version(),executable=sys.executable,numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    lines=['# GRID96学习结果解读','',
        '本轮没有学出可靠的补查触发器。扩大到56个状态、加入64棵浅树随机森林后，现有15个反馈特征仍未稳定区分“可以取消”和“存在代价权衡”的补查；现在不应将模型接入控制器。',
        '','这不是把负面结果解释成模型容量不足。本轮模型和阈值均已固定，逻辑回归与随机森林都没有给出足以支持替换原规则的证据，因此不继续加树、调阈值或挑种子。',
        '','## 标签说明了什么','',
        '56例中，39例取消后整条物理运行轨迹不变，只少了查询；另17例确实改变后续运行。其中5例取消在运行成本、查询和所记录过载指标上均不差，1例取消会提前结束，11例存在指标之间的权衡。共有44个SKIP、1个KEEP、11个MIXED标签。',
        '','所以，能够取消的补查不全是“选到相同动作”的冗余查询：确实还有5例改变轨迹后取消更有利。但这5例中有1例是已有试点，新的有利变化仅4例。不能把39个轨迹不变的简单例子，与这些需要判断长期后果的例子混为一种能力。',
        '','唯一的KEEP仍是已知的2035-07-09第1901步：保留补查完成2017步，单次取消只到1906步。它从主要预测成绩中排除，单独诊断。因此53个新标签中没有新的KEEP，不能用“没有误取消KEEP”证明模型学会保护重要补查。',
        '','## 固定模型的表现','']
    for scheme in ['calendar_loo','old_to_generated']:
        title='按日历日期留出' if scheme=='calendar_loo' else 'old13来源状态训练、G1/G2检验'
        lines += [title+'：','']
        for model in ['LOGISTIC','FOREST64']:
            h=next(r['headline'] for r in results if r['scheme']==scheme and r['model']==model);q=h['0.95'];sub=groups[f'{scheme}|{model}|different_physical_path']
            lines += [f"- {model}：主要样本{h['n']}例，SKIP识别AUC {h['skip_auc']:.3f}，平衡准确率{h['balanced_accuracy']:.1%}。固定0.95阈值建议取消{q['proposed_skips']}例，真正SKIP {q['correct_skips']}例，MIXED {q['disputed_mixed']}例。",
                f"  其中改变物理轨迹的{sub['n']}例，建议取消{sub['proposed']}例，对应标签{sub['selected_labels']}。"]
        lines+=['']
    lines += [
        '日期留出时，逻辑回归的13次高分取消建议只有8次属于SKIP，另5次是权衡样本。随机森林只有1次高分建议，属于轨迹不变的冗余补查，占53个新状态的1.9%；不能把1/1解释为可靠的100%精度。',
        '','53个新状态本来就有42例SKIP。全部预测SKIP的普通准确率已达79.2%，平衡准确率为50%。随机森林普通预测大多仍落在SKIP，平衡准确率48.8%，没有超过这个简单参照；不能用多数类的高命中率声称模型有效。逻辑回归平衡准确率49.6%，同样没有改善。',
        '','在已知试点的日期留出诊断中，随机森林在0.5阈值会取消唯一KEEP例；0.95阈值将它保留下来，但总体只能给出极少建议。阈值0.95是未校准的模型分数门槛，不是95%的可靠性保证。所有试点结果均保留，未计入主要成绩。',
        '','## 对下一步的判断','',
        '本轮最明确的收获是把数据问题分清了：多数标签是在识别无效查询，真正改变运行后果的样本少，而且很多没有单一的好坏答案。现在不值得继续增加模型复杂度，也不应改标签或阈值来追求更好成绩。',
        '','先保留这批逐状态的反事实记录，用它们解释哪些补查只是多查了一遍、哪些补查会改变运行后果。若继续学习，首先要解决的是目标和信息：哪些代价权衡允许取消、补查前哪些可见信息能够区分这几类后果。仅扩大森林或换神经网络，不能补上当前缺少的证据。',
        '','本轮没有部署门控、没有新增闭环学习策略比较。单次取消之后仍沿用原FRAME，多个单次收益不能相加成整套策略的收益。21条已有场景记录覆盖17个日历日期，均来自同一已暴露公开电网；这里也没有跨电网或独立AC重求解结论。',
        '','本解读的轨迹分类是拟合完成后的描述性分析，没有用于取样、拟合、调参或选择阈值。原预设评价及全部预测保留在GRID96_report.md、learning_results.json、predictions.json中。']
    (OUT/'学习结果解读_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save('readout_receipt.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='Read-only interpretation after frozen evaluation',new_fits=0,new_replays=0,new_predictions=0,
        sources={n:sha(OUT/n) for n in ['labels.json','predictions.json','learning_results.json','review.json','learning_review.json']},script_sha256=sha(W/'readout.py'),report_sha256=sha(OUT/'学习结果解读_中文.md')))
    print(json.dumps(groups,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
