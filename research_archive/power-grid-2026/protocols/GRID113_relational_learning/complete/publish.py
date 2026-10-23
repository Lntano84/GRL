"""Stage frozen setup or audited final result; never fit or select."""
import shutil,datetime
from common import *
def copytree_files(dest,final):
    dest.mkdir(parents=True,exist_ok=False)
    # Large reproducible caches stay on D:, original frozen sources live in prior archives.
    for p in O.rglob('*'):
        if not p.is_file() or p.suffix=='.pkl' or p.name.endswith('.tmp') or p.name=='frozen_push.json':continue
        if not final and p.name not in ['design.json','preflight.json','code_freeze.json','data_freeze.json','state.json']:continue
        d=dest/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    shutil.copy2(R/'work/grid112_relational_plan/schema.py',dest/'schema112.py')
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
def docs(note,nexttext):
    for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+note+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(nexttext+'\n',encoding='utf8')
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['frozen','complete'],required=True);phase=ap.parse_args().phase
    verify();base=REPO/'research_archive/power-grid-2026/protocols/GRID113_relational_learning'
    if phase=='frozen':
        assert not (O/'training_started.json').exists();copytree_files(base/'frozen',False)
        docs('## 2026-10-23 — GRID113 authorized paired relational experiment frozen\n\nUser approved GRID112. Implemented exactly5938params/arm, paired seed initial tensors, old7250x32 index streams/counts, fixed train-only labels/fees, synthetic forward/backward with zero optimizer updates. Context/next/padding invariance and FACTOR feedback permutation pass; JOINT pairing witness responds.4438 public inputs cached separately from targets; train/validation dates disjoint. Authorized exactly6serial fits,3seeds,1800s total wall cap, final endpoint only; no physical calls/test/scaling. Frozen code/data/preflight archived before research updates.',
             '# Next steps — GRID113固定六次训练已封存\n\n1. 封存推送后仅执行JOINT/FACTOR各三种子7250更新，1800秒总上限。\n2. 全部完成后才统一离线评价、审计并报告全部种子/日期与原对照；不自动扩大或闭环。\n3. 不改损失/费用/数据/预算，不读取最终测试集。')
        print('FROZEN_ARCHIVED',flush=True);return
    assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists()
    D=read(O/'decision.json');res=read(O/'validation_results.json');metrics=read(O/'state_metrics.json');fits=read(O/'fits.json')
    lines=['# GRID113：动作—反馈配对学习比较','',
        '本轮只比较保留动作与已查反馈的对应关系（JOINT），和先汇总反馈再结合动作（FACTOR）。两组各5938参数、三个配对种子；初始权重、7250更新、抽样序列、Huber损失、费用和目标相同。共六次固定终点监督回归，未运行新的强化学习、电网回放或最终测试。','',
        f"六次训练共耗时{fits['wall_s']:.1f}秒，原总上限1800秒。训练3582状态/2378起始前缀，验证856状态/590起始前缀；日期16/4、场景32/8，同一个电网。前缀与同日期副本相关，四个验证日期已暴露。",'',
        '## 完整补查序列结果','',
        '净效用为增益减去查询费后除以原收益尺度；主指标先按日期平均，再对四日期平均。它衡量固定教师路径上的补查停止决策，不等于整周成本或控制时间。','']
    for arm in ['JOINT','FACTOR']:
        lines.append(f"{arm}三种子平均净效用：{D['family_mean_utility'][arm]:.12f}。")
        for s in range(3):
            name=f'{arm}_seed{s}';v=res[name]['all']['macro'];m=metrics[name]['validation']
            lines.append(f"- 种子{s}：净效用{v['mean_net_utility']:.12f}，物理调用均衡效用{v['physical_call_balanced_utility']:.12f}，收益保留率{v['gain_capture']*100:.3f}%，查询比例{v['query_fraction']*100:.3f}%，正收益漏停{m['false_stop']}/41，无利继续{m['unprofitable_continue']}，继续价值RMSE{m['continue_value_rmse']:.6f}。")
        lines.append('')
    lines += [f"固定一批BATCH1净效用{D['batch1_utility']:.12f}；历史DIRECT三种子平均{D['family_mean_utility']['DIRECT']:.12f}；历史gamma0 DQN平均{D['family_mean_utility']['DQN0_']:.12f}。后两者网络为原6978参数MLP，对它们的比较是历史竞争参照，不能单独归因为配对编码。",'','## 配对差异与预设判读','']
    for p in D['paired_seeds']:lines.append(f"- 种子{p['seed']}：JOINT−FACTOR={p['delta']:.12f}。")
    lines.append('')
    for d,v in D['date_deltas'].items():lines.append(f'- {d}：种子平均JOINT−FACTOR={v:.12f}。')
    lines += ['',f"机制信号判据（家族均值更高、三个种子都同向、至少3/4日期同向）：{'达到' if D['mechanism_signal'] else '未达到'}。进一步同时超过固定一批、历史DIRECT与DQN家族：{'达到' if D['all_baseline_anchors_exceeded'] else '未达到'}。",'',
        '该比较隔离的是本小型网络中反馈聚合位置，不涵盖所有关系建模方法。即使平均改进，也不能据三个种子和四个开发日期声称统计显著或跨网泛化；未满足判据时保留全部混合和不利结果，不追加模型或预算救结果。','',
        '## 固定难例','']
    for r in read(O/'critical_probes.json'):
        lines.append(f"{r['context']}，已知下一批归一化净收益{r['known_batch_normalized_net']}：")
        for arm in ['JOINT','FACTOR']:lines.append(f"- {arm}三个种子继续决策：{[r['scores'][f'{arm}_seed{s}']['action'] for s in range(3)]}（1继续，0停止）。")
    lines += ['','这些已暴露难例仅作诊断，不作为训练加权、挑种子或整轮接受依据。','',
        '## 工程核验与边界','',
        '全部4438原始目标核对；六份权重和Adam7250步收据、原抽样计数与流哈希、同参数初始权重、集合/填充不变性通过。独立NumPy实现对全部正收益状态和固定间隔状态、三个难例核对输出与决策；所有完整补查序列重新计费。输入只含已买反馈和公共连接方案，待查结果只在forward之后用于loss。最终8测试场景未生成。详见review.json、state_metrics.json、validation_results.json和逐序列文件。','',
        '网络仍采用原即时净收益监督目标，并未解决长期闭环信用分配。现有数据来自FULL教师路径，学习策略进入真实控制后的状态可能不同。输入构造与模型推理的真实控制开销本轮未计时，因此不能据此声称实际提速或整周质量改善。是否扩大规模或安排闭环需另行决定。']
    report=O/'GRID113_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    collab=R/'论文协作资料/FRAME_配对学习结果_GRID113';collab.mkdir(exist_ok=False)
    save(O/'complete.json',dict(passed=True,experiment_completed=True,new_fits=6,mechanism_signal=D['mechanism_signal'],competitive_progression=D['competitive_progression'],
        ready_to_scale=False,report=str(report),archive=str(base/'complete'),collaboration=str(collab),new_physical_steps=0,final_test_materialized=False))
    copytree_files(base/'complete',True)
    for name in ['GRID113_report.md','decision.json','review.json','complete.json','validation_results.json','state_metrics.json','critical_probes.json']:shutil.copy2(O/name,collab/name)
    note=f"## 2026-10-23 — GRID113 fixed paired relational learning complete\n\nExactly6fits,5938parameters each,7250updatesx32,paired init/index stream,old Huber/fee/labels. Total training{fits['wall_s']:.1f}s<1800. JOINT/FACTOR family utility{D['family_mean_utility']['JOINT']:.12f}/{D['family_mean_utility']['FACTOR']:.12f}; paired seed deltas{[p['delta'] for p in D['paired_seeds']]},date deltas{D['date_deltas']}. Mechanism gate{D['mechanism_signal']};competitive progression{D['competitive_progression']}.All adverse outcomes retained. Independent NumPy/sampling/targets/optimizer/whole-search audit passed. Offline exposed development teacher-path utility only, not physical wholeweek/time or cross-grid evidence; final8test ungenerated, noautomaticrollout/scaling."
    docs(note,'# Next steps — GRID113已完成，等待下一路线决定\n\n1. 向用户汇报全部配对种子/日期、BATCH1与历史家族参照，说明机制判据及失败边界。\n2. 不自动追加训练、扩大规模、电网闭环或最终测试；本轮只回答配对编码是否改善已有教师路径停止决策。\n\n报告：research_archive/power-grid-2026/protocols/GRID113_relational_learning/complete/GRID113_report.md')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID113配对学习结果](protocols/GRID113_relational_learning/complete/GRID113_report.md)：固定六次JOINT/FACTOR监督回归、全部种子和日期、离线完整补查评价与审计。\n')
    save(O/'state.json',dict(stage='published_awaiting_remote_verification',fits=6));print('PUBLISHED',flush=True)
if __name__=='__main__':main()
