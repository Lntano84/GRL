"""Publish the fixed-population diagnostic; preserve original experimental results."""
from pathlib import Path
import collections,json,math,shutil,sys
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;sys.path.insert(0,str(W));import run as C
O=C.O;REPO=C.REPO;A=C.A;read=C.read;sha=C.sha
def main():
    C.verify();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists();s=read(O/'summary.json');safe=s['savings'];cases=read(O/'consensus_override_cases.json')['cases'];allcases=read(O/'factor_frame_cases.json')['cases'];assert len(cases)==3
    stats=read(R/'outputs/grid184_search_adapter/results_summary.json');look={(v['repeat'],v['week'],v['method']):v for v in read(R/'outputs/grid184_search_adapter/review.json')['runrows']}
    other=[r for r in allcases if r['class_name']=='fewer_different_chosen'];assert all(v['all_extra']['gain']>0 and v['all_extra']['normalized_net']<0 for v in other)
    annotations=[]
    for c in cases:
        a=look[0,c['week'],f"CONSENSUS_seed{c['seed']}"];b=look[0,c['week'],f"FACTOR_seed{c['seed']}"]
        annotations.append(dict(seed=c['seed'],week=c['week'],step=c['step'],direction=c['direction'],whole_week_candidate_query_delta=a['candidate_queries']-b['candidate_queries'],next_batch_normalized_NET=c['next_batch']['normalized_net'],whole_week_raw_cost_delta=c['whole_week_cost_delta'],cost_component_units='native score cost units, price-weighted, NOT MWh',cost_component_deltas=c['cost_component_deltas'],rounding_residual_delta=c['cost_rounding_residual_delta']))
    C.save('interpretation_receipt.json',dict(cases=annotations,correction_to_previous_summary='Threeactualdifferences are1Ccontinue/Fstop and2Cstop/Fcontinue,not3overridesofFstop.',all10different_action_skips_positive_raw_gain_but_negative_priced_NET=True,new_model_forwards=0,new_fits=0,new_physics=0))
    lines=['# GRID185：学习停止到底省掉了什么，三处分歧为什么吃亏','',
        '本轮只解释GRID184的已有记录，保留全部62个同观测、同基础动作的FACTOR/FRAME配对，以及全部3个共识/FACTOR首次物理动作分歧。没有重新训练、做模型前向、生成场景或调用电网仿真。','',
        '## 先更正分歧方向','',
        '上一轮把三处分歧概括为共识覆盖原模型的停止决定，这个概括不准确。实际是一处原模型停而共识继续，两处原模型继续而共识先停。分析程序对错误假设触发断言后，原代码、部分输出和错误均已保留；案例、查询代价、种子和判读目标没有修改。','',
        '## 23个省查询调用的证据','',
        f'23个配对调用来自3个日期，去重后是{safe["unique_physical_state_base"]}种观测与基础动作、{safe["unique_stop_prefixes"]}种停止前缀。按种子配对累计省下{safe["skipped_queries_seed_paired"]}次候选查询、{safe["skipped_batches_seed_paired"]}个批次；同一FRAME调用可能被不同种子重复作为对照，这不是1945个独立场景，也不是新增实验。','',
        '这23个调用中，FRAME在FACTOR停止后实际查询的每个后续批次，最优已观察奖励增益都是0。双方最终搜索动作相同，实际施加到电网的动作也相同，下一步观测逐字节一致。这里省掉的是没有改变当前控制决定的评估，证据比单纯查询更少更直接。','',
        '其中15次在补查入口直接停，6次查完一批后停，另各1次在查完两批、三批后停。21次停止时上一批没有奖励改善，但另外2次上一批有改善，仍可省去无效尾部。因此不能从这些案例临时拼一个“上一批没进步就停”的规则；这里展示的是后验效果，不是可直接使用的在线触发条件。','',
        '固定候选反馈和比较规则下，有一个简单性质：若所有尚未查询的可接纳候选，其（float32奖励，负候选编号）都不能严格超过当前最优，则跳过剩余评估会返回同一个动作。学习器只能预测这种情况，不能在没有查询时直接证明条件成立。以上23例验证了记录中的动作等价，不能作为未见状态的安全保证。','',
        '## 另10个停止不能混在无损节省里','',
        '其余10个少查询调用选择了不同动作。FRAME额外查询确实找到了更高的即时奖励，但增益都不足以抵消预设的查询代价；这些案例的下一批及实际完整补查净收益均为负。因此，它们符合原NET目标的取舍，却不是动作不变的省查询。另29个配对调用的查询和最终动作均相同，也完整保留。','',
        '这正是现有目标中需要拆开的两件事：一类是剩余评估不会改变当前动作，另一类是剩余评估会改善动作，但按照查询价格决定放弃。后者仍会改变物理轨迹，不能因为NET为负就声称没有质量代价。','',
        '## 三处分歧逐项解释','',
        '净收益仍沿用原公式：下一批净收益＝（最优已观察奖励增益－原每次查询价格×该批查询数）/原收益尺度。价格为1.2442348640059487e-5，尺度为0.008627818251142696。这是学习任务的代理单位，不是实际秒数、货币或整周价值。', '']
    names={'losses_mwh':'网损项','redispatch_mwh':'再调度项','curtailment_delta_mwh':'限发变化项','storage_throughput_mwh':'储能吞吐项'}
    for c,n in zip(cases,annotations):
        direction='共识继续，FACTOR停止' if c['direction']=='F_STOP_to_C_CONTINUE' else '共识停止，FACTOR继续'
        labels=c['next_batch'];label='共识对原下一批NET的符号判断错误' if not c['consensus_agrees_with_next_NET'] else '共识符合原下一批NET的符号目标，但整周成本更高'
        lines += [f"### {c['week']}，种子{c['seed']}，第{c['step']}步",'',f"{direction}。双方在同一观测、基础动作和已付费反馈前缀上分开；下一批实际查询{labels['queries']}次，最佳奖励增加{labels['best_reward_gain']:.10f}，标准化NET为{labels['normalized_net']:.6f}。{label}。",'',
            f"实际继续一方总共额外查询{c['all_actual_extra']['queries']}次，完整额外部分NET为{c['all_actual_extra']['normalized_net']:.6f}。两条真实轨迹比较，共识整周成本高{100*c['whole_week_cost_relative_delta']:.4f}%，累计候选查询变化{n['whole_week_candidate_query_delta']:+d}次。"]
        if c['seed']==1:lines += ['','这一处共识当下少查128次，但其整周候选查询后来反而多271次。少查当前一步并不等于整周少查；改变动作后，后续运行状态与搜索调用都会变化。']
        if c['week'].startswith('2037-03-02') and c['seed']==0:lines += ['','两个便宜模型都把这一批NET估成明显正值，而已记录结果略为负值。这一处包含明确的即时估计错误；不能全部解释成长期目标不一致。']
        if c['direction']=='F_CONTINUE_to_C_STOP':lines += ['','原FACTOR继续查询带来的即时奖励增益虽然不抵查询价格，却选择了另一动作，并对应更低的整周成本。把模型训练得更符合当前NET符号，也不能自动解决这种目标差异。']
        lines += ['', '成本账本的已执行后缀差额（原评分成本单位，不是MWh）：'+ '；'.join(f"{names[k]}{v:+.4f}" for k,v in c['cost_component_deltas'].items())+f"；浮点重算残差差{c['cost_rounding_residual_delta']:+.4f}。账本与保存逐步成本已在原容差内复核。",'',
            '这些是两条已执行路径的成本分解，只说明差额记在哪些项上，不把第一次动作分歧当成全部整周差额的独立因果证明。', '']
    lines += ['## 该保留什么，接下来查什么','',
        '保留FACTOR已经表现出的动作不变型省查询：23个配对、13种物理观测与基础动作、15种停止前缀，且当步实际动作和下一步状态均相同。新增共识不应直接整体替代FACTOR，也不应只保留继续或只保留停止来救这三个结果。','',
        '下一步最直接的问题是：现有教师记录能否把“后续不会改变当前最优动作”和“有小幅动作改善但不抵查询价格”分成可靠标签，再检查模型能否识别前一类。先确认标签覆盖、日期分组和教师路径适用范围；不用这3条反向轨迹直接造通用触发规则，也不把整周差额复制成每个批次的标签。','',
        '只凭本轮不能决定应增加强化学习复杂度还是改学习目标。明确存在一处即时估计错误和两处局部目标/整周评价的取舍，需要分别处理。','',
        '## 可复查材料','',
        'factor_frame_cases.json保留全部62个配对、停止前已有41维特征、原F预测和FRAME实际额外批次；consensus_override_cases.json保留全部3处分歧、双方模型边际值、已揭示的下一批结果和实际成本分解。原未来反馈仅供事后诊断，没有进入当时决策输入。summary.json、review.json、source_hashes.json与direction_correction.json记录复算、冻结及方向纠错。']
    (O/'GRID185_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.verify();C.save('complete.json',dict(passed=True,population62=True,physical_divergences3=True,new_physics=0,new_fits=0,new_model_forwards=0))
    C.archive('complete');dest=R/'论文协作资料/FRAME_停止机制解释_GRID185';dest.mkdir(exist_ok=False)
    for n in ['GRID185_report.md','summary.json','review.json','interpretation_receipt.json','complete.json']:shutil.copy2(O/n,dest/n);assert sha(O/n)==sha(dest/n)
    msg='GRID185只读解释：23个动作不变省查询配对去重为13种状态+base/15种STOP前缀，按种子配对省1945查询/46批，所有实际施加动作和下一观测相同；10个改动作的少查询例均有小奖励提升但NET不抵价格。纠正三处分歧方向：1继续/2提前停；1例即时NET判断错，2例符合NET目标但整周成本更高。须区分动作不变节省与有质量取舍的省查询，下一步只检标签可用性，不据3例调规则或追加实验。无新拟合/前向/物理。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID185学习停止机制解释\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID185机制解释后\n\n1. 保留动作不变省查询证据，23种子配对并非23独立状态，去重13种观测+base/15STOP前缀。\n2. 三处分歧实际为1多查/2先停；分别处理即时NET估计误差与局部NET/整周成本不一致，不只按一个方向改规则。\n3. 下一步仅查已有教师数据能否区分动作不变停止与有动作改善的定价取舍，检查覆盖、日期分组与路径局限；不新训练、物理、最终8或论文，不把3例整周差当通用批次标签。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID185学习停止机制\r\n\r\n[报告](protocols/GRID185_stopping_mechanism/complete/GRID185_report.md)；动作不变节省与价格取舍分开，1多查/2先停，未新训练或物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID185_COMPLETE')
if __name__=='__main__':main()
