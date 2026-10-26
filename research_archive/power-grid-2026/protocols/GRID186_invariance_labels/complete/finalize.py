"""Review and publish availability of recorded action-invariance supervision."""
from pathlib import Path
import json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;sys.path.insert(0,str(W));import run as C
O=C.O;REPO=C.REPO;A=C.A;read=C.read;sha=C.sha
def main():
    C.verify();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists();summary=read(O/'summary.json');rows=read(O/'label_rows.json');folds=read(O/'existing_date_fold_coverage.json');assert all(f['both_classes_in_fit'] and f['both_classes_in_held'] for f in folds)
    before=O/'interrupted_attempts/serialization01';samefiles={}
    for name in ['label_rows.json','train_labels.npz','validation_labels.npz']:
        samefiles[name]=sha(before/name)==sha(O/name);assert samefiles[name]
    detail={}
    for part in ['train','validation']:
        rr=[v for v in rows if v['partition']==part];negchanged=[v for v in rr if not v['invariant'] and v['next_batch_NET']<=0];delayed=[v for v in rr if v['negative_first_positive_later']]
        detail[part]=dict(next_NET_nonpositive_but_action_changes=len(negchanged),delayed_physical_groups=len({v['physical_state_group'] for v in delayed}),delayed_dates=sorted({v['date'] for v in delayed}),all_dates_have_both_classes=all(v['invariant']>0 and v['changed']>0 for v in summary[part]['per_date']))
    C.save('interpretation_receipt.json',dict(details=detail,label_files_unchanged_after_serialization_fix=samefiles,all4existing_whole_date_folds_have_both_classes=True,leakage_boundary='Futurechosen/gainsarelabels only,nevernewfeatures;normalizationandanyfuturefitmustexcludehelddates.',new_fits=0,new_model_forwards=0,new_physics=0))
    t=summary['train'];v=summary['validation'];lines=['# GRID186：已有教师数据能否监督动作不变型停止','',
        '可以构造这类标签，而且有覆盖，并不只有GRID185找到的13种状态。本轮只做已有记录的标签核查，没有训练、模型前向或电网预测。','',
        '## 标签问的是什么','',
        '在一个已保存的搜索前缀上，当前最优动作是否会在查完剩余同站候选后改变？当前动作与最终已观察最优动作相同，标为动作不变；否则标为会改变。比较规则沿用float32奖励与候选编号。完整教师后缀只用于产生监督答案，不能进入在线输入。','',
        '这个标签与原下一批NET标签不同：NET问的是下一批奖励增益能否抵消查询价格；动作不变标签问的是后续评估是否还会改动当前控制选择。它不直接表示整周安全、低成本或全局最优。','',
        '## 覆盖情况','',
        f"训练部分：{t['states']}个前缀状态、{t['starts']}个搜索起点，来自{t['dates']}个日期、{t['scenarios']}份场景和{t['physical_groups']}个物理调用组。动作不变{t['invariant']}个，会改变{t['changed']}个；每个训练日期都有两类，按日期等权的动作不变比例为{100*t['date_macro_invariance']:.2f}%。",'',
        f"原开发部分：{v['states']}个前缀状态、{v['starts']}个起点，来自{v['dates']}个已暴露日期、{v['scenarios']}份场景和{v['physical_groups']}个物理调用组。动作不变{v['invariant']}个，会改变{v['changed']}个；每个开发日期也都有两类，按日期等权比例为{100*v['date_macro_invariance']:.2f}%。",'',
        '同一物理调用会被拆成多个前缀，多个起点也可能共享后缀；因此4438个状态不是4438个独立场景。四个既有12/4训练日期划分逐一检查，所有fit和held分区均有两类，不改原日期分组。','',
        '## 原目标混合了哪些情况','',
        f"训练状态里，{t['classes'].get('changed_action_nonpositive_suffix_NET',0)}个存在动作改善，但查完完整后缀的NET不为正；开发部分有{v['classes'].get('changed_action_nonpositive_suffix_NET',0)}个。同样，按下一批标签看，训练有{detail['train']['next_NET_nonpositive_but_action_changes']}个下一批NET不为正、最终动作仍会改变的状态，开发有{detail['validation']['next_NET_nonpositive_but_action_changes']}个。",'',
        '这不是原NET标签计算错了，而是它允许为了少计算而放弃会改变动作的小幅改善。若论文希望强调在保持控制决定的同时减少仿真，这类取舍应与动作不变型停止单独学习、单独评估。完整后缀NET与下一批NET不能互换，表述中均分别保留。','',
        '## 强化学习是否有额外的序列机会','',
        f"原训练记录中，有{t['delayed_opportunity_states']}个状态出现下一批NET不为正、但再多查若干批次后累计NET可以为正，对应{t['delayed_opportunity_starts']}个起点、{detail['train']['delayed_physical_groups']}个物理调用组，集中在{len(detail['train']['delayed_dates'])}个日期。旧开发记录中这类状态为{v['delayed_opportunity_states']}。",'',
        '因此当前四日期开发集不能有效检验“先付计算代价、后获收益”这一强化学习优势，训练中这类机会也比较集中。这不能证明强化学习没有价值，也不能把全部失败归因于长期目标；先把监督目标区分清楚更直接。这里只统计教师已查询后缀的可见机会，没有求新的最优策略。','',
        '## 下一轮学习该如何组织','',
        '优先做一个目标清楚的对照：保持现有在线输入、网络、种子、日期分组和训练预算，改为预测剩余搜索是否还会改变当前动作，原NET模型继续作为对照。先看错误停止率、动作一致率、真实查询节省及每日期表现，再决定是否进入物理闭环；不把多数类准确率当成有效学习。','',
        '标签中动作不变占多数，全部预测不变也可能看似准确，却会遗漏所有528个训练动作变化状态。因此要明确报告把会改变错判为不会改变的数量与比例，并同时保留少量稳定节省的研究价值，不额设全日期全赢门槛。当前没有启动这轮训练，也没有确定新的阈值。','',
        '## 适用范围与材料','',
        '这些是旧FULL教师路径上的完整同站反馈，不是FRAME或新学习策略在所有新状态上的反事实答案；改了停止策略后会走到新轨迹，需要后续独立的真实闭环验证。现有开发日期已暴露，不能作为盲测。最终8个测试场景仍未生成。','',
        'label_rows.json保留全部4438个状态和原分区/日期/物理组标识；train_labels.npz与validation_labels.npz按原模型元数据顺序保存标签，可直接核对而不混入未来特征。existing_date_fold_coverage.json给出四个既有日期划分的覆盖。第一次输出遇到NumPy整数序列化问题，修正只改变JSON输出，三个已落盘标签文件逐SHA保持一致；错误与原源文件均保留。']
    (O/'GRID186_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.verify();C.save('complete.json',dict(passed=True,labels_available=True,all4folds_both_classes=True,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_动作不变标签核查_GRID186';dest.mkdir(exist_ok=False)
    for name in ['GRID186_report.md','summary.json','review.json','interpretation_receipt.json','complete.json']:shutil.copy2(O/name,dest/name);assert sha(O/name)==sha(dest/name)
    msg=f"GRID186旧标签核查通过：训练3582状态中3054动作不变/528会变，旧开发856中741/115，全部日期及既有四折fit/held均有两类。训练{detail['train']['next_NET_nonpositive_but_action_changes']}个、开发{detail['validation']['next_NET_nonpositive_but_action_changes']}个下一批NET非正但后续动作会变，说明原目标允许定价取舍；训练40延迟收益状态集中3日期，旧开发0，不能靠该开发集检RL延迟优势。仅已有FULL教师标签，尚未拟合/前向/物理/最终8；建议同输入架构预算只改动作不变目标，先评错误停止和查询再决定闭环。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID186动作不变标签覆盖\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID185/186后\n\n1. 保留13种状态上的动作不变节省；三处物理分歧实为1继续/2先停，分别处理估计错误与局部NET/整周成本取舍。\n2. 动作不变标签已有训练3054/528、旧开发741/115及四折覆盖。下一轮只拟定同输入/架构/预算/日期的目标对照，不再叠共识；拟合前说明具体范围，未开始新训练或物理。\n3. 以错误停止、动作一致、每日期真实查询/成本评估，前缀非独立；旧开发无延迟收益机会，不据其否定RL。FULL路径标签不能当新策略所有状态的答案，最终8与论文不动。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID186动作不变标签\r\n\r\n[报告](protocols/GRID186_invariance_labels/complete/GRID186_report.md)；旧教师标签覆盖已核对，尚未训练或新物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID186_COMPLETE',json.dumps(detail))
if __name__=='__main__':main()
