"""Reconcile next-step advice with prior matched auxiliary trials."""
from pathlib import Path
import shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def main():
    C.verify();assert read(O/'history_match_review.json')['passed'];assert not (O/'history_reconciliation.json').exists()
    plan=read(O/'history_match_plan.json')
    for p,h in plan['sources'].items():assert C.sha(R/p)==h
    m=read(O/'history_match.json')['summaries'];d=read(R/'outputs/grid123_event_auxiliary/decision.json');f=d['family_utility'];oof=f['oof']['AUX_GAIN']-f['oof']['NET_ZERO'];dev=f['validation']['AUX_GAIN']-f['validation']['NET_ZERO']
    lines=['# GRID188补充：对齐旧辅助学习后，再确定下一步','',
        '阅读顺序：先看GRID188_report.md的当前机制证据，再看本说明。主报告提出保留收益幅度、考虑动作变化辅助任务；历史回查进一步表明，普通辅助头本身已经试过，不能直接把这一建议当成新的训练方案。原冻结报告和材料保留。','',
        '## 以前已经做过什么','',
        'GRID123比较过同容量的NET回归、NET＋下一批原始增量事件辅助、NET＋净收益符号辅助。使用旧41维摘要网络、相同初始化/预算/日期划分，辅助输出不直接决定停止。GRID124又检查了主辅梯度和决策误差。因此不能把“加一个事件辅助头”称为尚未尝试。','',
        f"旧AUX_GAIN在16日期折外的效用差为{oof:+.8f}，保留这点小幅信号；在原4开发日期则为{dev:+.8f}。结果并不稳定，既不能说辅助任务没有价值，也不能说它已经解决当前问题。旧增量/已知费用分解和MSE/KL损失替换也做过，不直接重复这些路线。",'',
        '## 按同一起点、位置、种子和留出日期匹配','',
        '以下只读旧折外预测，不进行新模型前向。旧模型为6978参数摘要网络，当前为12980参数FACTOR；旧辅助目标是下一批有无增量，当前纯二元目标是完整后缀会不会改选。因此匹配说明案例是否反复出现，不构成当前网络上辅助监督的同容量消融。','',
        '|当前案例集合|配对记录|旧NET继续|旧AUX_GAIN主输出继续|旧事件头判有增量|','|---|---:|---:|---:|---:|']
    labels={'focus_restore':'本次三种子共同恢复的8组','focus_lost_profitable':'本次共同退化且下一批NET为正','top3loss':'本次负贡献最大的3组'}
    for k,v in m.items():lines.append(f"|{labels[k]}|{v['records']}|{v['old_NET_continue']}|{v['old_AUX_GAIN_continue']}|{v['old_AUX_GAIN_event_positive']}|")
    lines+=['',
        '本次最大的3组损失中，旧AUX_GAIN主输出23/23继续，而事件头仅11/23判有增量。不能把事件头看作已经可靠的补查开关；另一方面，带幅度信息的价值输出确实保留了这些继续机会。这里的23条仍来自3个物理组，不能当23个独立场景。','',
        '本次8组恢复案例中，旧辅助模型也比旧NET多一些继续，但并非全部恢复。当前“找回小改善、同时丢掉大改善”并不是只能通过重新做一遍辅助分类来解决的问题。','',
        '## 对下一步的修正','',
        '保留原NET主模型；将“找回小改善时，不丢失它已识别的大收益机会”作为最小改动的设计目标。先明确新模块究竟允许改变哪些原有继续/停止决定，以及相应增加的查询和推理代价。辅助监督只是候选实现，不能代替这个定义。','',
        '新方案必须说明与GRID123普通共享辅助头、GRID126/127目标替换及GRID184双向共识的具体区别。优先考虑基于固定价值基线的定向学习改进，先做同预算和行为边界推导，再决定是否有必要训练；当前没有生成新门控规则或试算一种组合来挑选有利结果。','',
        '后续比较应同时保留当前原NET、已有辅助学习和简单固定一批；不能只与本次较弱的纯二元目标比较。全部日期、种子和收益/查询分解继续保留，小而重复的改善有价值，不增加全赢门槛。','',
        '当前仍为零新训练、零模型前向、零仿真。history_match.json包含179条逐项匹配，history_match_plan.json保存旧来源哈希；旧模型分数未混入当前总体成绩。']
    (O/'历史对照后的下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('history_reconciliation.json',dict(passed=True,old_auxiliary_trial_acknowledged=True,old_aux_OOF_small_gain=oof,old_aux_old_dev_difference=dev,matched_rows=179,next='Define a value-baseline-preserving targeted improvement and cost before another auxiliary fit; do not duplicate GRID123 or new bilateral consensus.',new_fits=0,new_model_forwards=0,new_physics=0));C.archive('history_reconciliation')
    dest=R/'论文协作资料/FRAME_重复改善机制_GRID188'
    for n in ['历史对照后的下一步.md','history_match.json','history_match_review.json','history_reconciliation.json']:shutil.copy2(O/n,dest/n);assert C.sha(O/n)==C.sha(dest/n)
    msg='GRID188历史对齐补充：GRID123普通事件辅助已经试过，旧OOF小幅+.000641保留、旧4开发−.01336，不据此重新做一样训练。本次top3loss的23配对旧AUX_GAIN主值全部继续、事件头仅11正；新旧容量/输入/目标不同，只作179条同状态/种子/held日期匹配，非同容量因果对照。下一步先定义保留价值基线能力的定向增量学习与查询成本，必须区别旧辅助头/损失替换/双向共识，不直接再训或试组合。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID188历史对齐与下一步修正\n\n'+msg+'\n',encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID188历史对齐后\n\n1. 原NET保留为主模型。先定义“找回小改善、保留已有大收益继续”的定向学习改进及查询/推理成本，说明与旧GRID123共享辅助头、GRID126/127损失替换、GRID184双向共识的区别；不直接重复普通辅助训练。\n2. 同时保留当前NET、已有辅助模型和固定一批参照。8组重复恢复仍值得研究，3组大损失及17组重复退化也保留；不能只与弱二元模型比，不要求全赢。\n3. 新训练前通知具体范围；当前零新拟合/前向/仿真，未生成最终8、未改论文。新物理/扩数据/第二网范围另定。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# GRID188补充：历史对齐后的下一步\r\n\r\n[机制报告](protocols/GRID188_repeated_cases/complete/GRID188_report.md)；[历史对齐说明](protocols/GRID188_repeated_cases/history_reconciliation/历史对照后的下一步.md)。普通辅助头已有旧试验，先定义保留价值基线的改进，不重复盲训。\r\n\r\n'.encode('utf8')+p.read_bytes())
    print('GRID188_HISTORY_RECONCILED',C.json.dumps(dict(oof_delta=oof,dev_delta=dev,matched=179)))
if __name__=='__main__':main()
