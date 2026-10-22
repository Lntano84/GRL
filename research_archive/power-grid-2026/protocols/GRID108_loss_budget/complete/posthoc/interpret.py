"""Postfit error transitions: locate benefits and reversals without another fit."""
import collections,gzip,json,shutil
import numpy as np
from common108 import ROOT,OUT,REPO,ARCHIVE,read,save,sha
def main():
    assert not (OUT/'error_transitions.json').exists()
    with gzip.open(OUT/'state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        rows=[json.loads(x) for x in f]
    selected={}
    for r in rows:
        if r['partition']=='validation' and r['method'].startswith('HUBER_'):
            selected[r['method'],r['sample_id'],r['position']]=r
    summary={};events=[]
    for seed in [0,1,2]:
        rs=[r for (method,_,_),r in selected.items() if method==f'HUBER_7250_{seed}'];count=collections.Counter();groups=collections.defaultdict(collections.Counter)
        for a in rs:
            b=selected[f'HUBER_29000_{seed}',a['sample_id'],a['position']];assert a['target_net']==b['target_net']
            positive=a['target_net']>0;kind=None
            if positive and a['action']==0 and b['action']==1:kind='positive_recovered'
            elif positive and a['action']==1 and b['action']==0:kind='positive_newly_missed'
            elif not positive and a['action']==1 and b['action']==0:kind='unprofitable_query_removed'
            elif not positive and a['action']==0 and b['action']==1:kind='unprofitable_query_added'
            if kind:
                count[kind]+=1;groups[a['date']][kind]+=1
                events.append(dict(seed=seed,date=a['date'],sample_id=a['sample_id'],position=a['position'],physical_state_group=a['physical_state_group'],
                    event=kind,target_net=a['target_net'],old_continue_value=a['q1'],new_continue_value=b['q1']))
        summary[str(seed)]=dict(counts=dict(count),per_date={d:dict(c) for d,c in groups.items()},
            newly_missed_physical_groups=len({e['physical_state_group'] for e in events if e['seed']==seed and e['event']=='positive_newly_missed'}))
    save(OUT/'error_transitions.json',dict(posthoc=True,summary=summary,events=events,new_fits=0,
        note='All reachable validation-state action flips, not independent samples or sums of complete-policy/week causal effects.'))
    D=read(OUT/'decision.json');f=D['families'];hb=f['HUBER_7250'];hl=f['HUBER_29000'];ms=f['MSE_7250'];ml=f['MSE_29000']
    interpretation=dict(critical552_huber_continuation_corrects_all_seeds=True,
        longer_training_huber_mean_utility_improves=False,mse_replacement_mean_superiority_supported=False,
        cause_not_reduced_to_single_loss_or_undertraining=True,representation_insufficiency_proven=False,
        convergence_established=False,ready_to_scale=False,no_additional_experiments=True)
    save(OUT/'overall_interpretation.json',interpretation)
    lines=['# GRID108结果解读：能学对关键状态，但整体泛化仍不稳定','',
        '**延长训练能修正第552步，却没有提高整体验证收益；换平方损失也没有解决问题。** 因此，目前不能把原因简化成“训练太少”或“Huber损失不合适”。','',
        '## 1. 更长训练确实改变了关键决策','',
        'Huber从7250延长到29000次更新后，第552步三个种子都选择继续，原FRAME路径和RF路径两个上下文均如此。短训练时是“停、停、继续”。这表明这一处预测能够随训练得到修正，不能再说该状态在现有输入下完全学不会。',
        '但这只是固定已暴露上下文的预测动作修正，没有新增闭环，也不等于整个学习控制器的整周成本已经恢复。','',
        '## 2. 它同时漏掉了更多别处的正收益','',
        f"Huber训练集正净收益漏查由{hb['train_false_stops']}降到{hl['train_false_stops']}；验证集却由{hb['validation_false_stops']}升到{hl['validation_false_stops']}。四日期平均净效用从{hb['mean_net_utility']:.8f}降到{hl['mean_net_utility']:.8f}（{hl['mean_net_utility']/hb['mean_net_utility']-1:+.2%}）。",
        '按日期看，三月平均净效用上升，但五月、八月和十月下降。不能只盯住已知的第552步而忽略其他日期。原固定一批净效用为0.16286891，较长Huber为0.15999804，没有超过这一强对照。']
    for seed,s in summary.items():
        c=s['counts'];lines.append(f"- 种子{seed}：修正原先漏查的正收益状态{c.get('positive_recovered',0)}个，却新增漏查{c.get('positive_newly_missed',0)}个；这些新漏查来自{s['newly_missed_physical_groups']}个物理搜索调用。")
    lines += ['', '上述是同一批856个可达状态的事后动作变化，多个前缀可能属于同一次搜索。它有助于定位误差转移，不等同于新增独立场景，也不能把各状态损失简单加成闭环损失。','',
        '## 3. 平方损失没有带来预期的稳定收益','',
        f"7250次更新时，MSE三种子平均净效用{ms['mean_net_utility']:.8f}，Huber为{hb['mean_net_utility']:.8f}，三个配对种子全部下降。29000次时，MSE为{ml['mean_net_utility']:.8f}，Huber为{hl['mean_net_utility']:.8f}，仍有两个配对种子下降。",
        'MSE本身延长训练后，三个种子的净效用都改善，但还没有超过原Huber短训练、原DQN或固定一批的家族平均。第552步MSE仍只有两个种子继续，延长后也没有新增修正。',
        '上一轮观察到总体标签均值与Huber中心符号不同，只能说明损失值得检验；本轮真实网络对照没有支持“换MSE就能解决”的推断。总体标签统计不是第552步条件价值的证明。','',
        '## 4. 目前更明确的问题是什么','',
        '模型可以减少训练误差，也能改变某个关键验证状态的判断，却不能把这种改善稳定转移到其他日期。这更接近泛化不稳定与收益估计取舍，而不是单纯训练接口错误。两因素严格匹配初始化、Adam、抽样序列、数据、费用及网络，原短端点轨迹也精确重现。',
        '仍不能直接断言一定缺特征或已经过拟合：有限预算没有证明收敛，部分种子的末段训练损失仍下降；不同损失也改变了误差权重。需要进一步检查新增漏查状态与训练覆盖、可见输入之间的关系。','',
        '## 下一步建议','',
        '先停止增加训练量或切换损失，不扩大数据/网络。下一步优先对比“被修正的552”和“延长训练后新漏掉的状态”：当前41维输入是否把需要不同补查决策的情况混在一起，训练样本是否覆盖这些差异，再决定补哪类信息。',
        '这不是立即增加特征或新模型的许可；应先做现有数据的误差分组与输入可区分性检查。若以后提出新输入，先固定其来源和动机，再单独比较。',
        '全部6条新训练路径、9个新权重端点、3个复用端点及不利结果已保存并通过核对。没有新物理回放，最终8测试场景未生成。']
    (OUT/'结果解读_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dest=ARCHIVE/'complete';post=dest/'posthoc';post.mkdir(exist_ok=False)
    shutil.copy2(__import__('pathlib').Path(__file__),post/'interpret.py')
    collab=__import__('pathlib').Path(read(OUT/'complete.json')['collaboration'])
    for n in ['结果解读_中文.md','overall_interpretation.json','error_transitions.json']:
        shutil.copy2(OUT/n,dest/n);shutil.copy2(OUT/n,collab/n)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'})
    note='\n\n## 2026-10-22 — GRID108 calibrated loss/budget conclusion\n\nHuber continuation corrects both March552 contexts to CONTINUE in all3seeds, but validation positive misses13/14/13 ->18/19/17 and family utility0.16292381 ->0.15999804 (-1.80%); March improves, other3 dates worsen. Training errors drop, so critical-case rescue is not general benefit. MSE7250 utility0.15375892, MSE29000 0.15953367; loss replacement inferior in all3 short-budget pairs and2/3 long pairs. Longer MSE helps3/3 relative to MSEshort but neither new family beats BATCH1/oldDQN mean. Hypothesis that robust loss alone causes failure unsupported; convergence and feature insufficiency unproven. Next recommendation read-only error-group/input distinguishability, not further budget/loss tuning or new features automatically. No new physical rollout/final-test access; no best endpoint chosen.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as out:out.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID108完成，关键修正未转成整体收益\n\n1. 向用户汇报：Huber延长后552三种子全修正，但其他日期漏查增加、总体净效用下降；MSE无稳定替换优势。\n2. 建议后续只读比较552与新增漏查的输入和训练覆盖，先查可区分信息；不自动加特征、换模型或再延长预算。\n3. 不启动新闭环或扩大数据/网络，最终8测试场景保持封存。\n\n解读：research_archive/power-grid-2026/protocols/GRID108_loss_budget/complete/结果解读_中文.md\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
