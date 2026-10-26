"""Publish descriptive mechanism evidence without changing models."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;P=C.P;read=C.read
def main():
    C.verify();assert read(O/'review.json')['passed'] and read(O/'native_feedback_audit.json')['passed'];assert not (O/'complete.json').exists()
    a=read(O/'analysis.json');rr=read(O/'unique_divergence_states.json');events=read(O/'events_all_seeds.json');d=read(O/'top3_loss_details.json');rest=a['groups']['RESTORED'];lost=a['groups']['LOST']
    # Reconstruct all222 contributions using only action-changing paired records.
    cases=[v for v in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if v['partition']=='train'];nc=collections.Counter(v['date'] for v in cases);gg=collections.defaultdict(float)
    for e in read(P/'paired_action_audit.json')['all_changed_action_examples']:gg[e['physical_state_group']]+=(e['new_gain']-e['old_gain'])/(3*16*nc[e['date']])
    for g in a['loss_concentration']['all_groups']:assert abs(g['gain']-gg[g['group']])<1e-12
    assert len(events)==156 and len(rr)==53 and len({(v['sample_id'],v['position']) for v in rr})==53
    assert sum(len(v['seeds']) for v in rr)==156
    for cat,n in [('RESTORED',24),('LOST',28)]:assert len({v['sample_id'] for v in rr if v['category']==cat})==n
    assert not a['overlap_physical_groups']
    allprofitable=all(v['next_NET']>0 for v in d['all_action_changing_pairs']);assert allprofitable
    ratio=lost['suffix_gain_group_mean']/rest['suffix_gain_group_mean']
    sourceaudit=read(O/'native_feedback_audit.json');names=list(rest['all41_observed_group_balanced'])
    selected=['station_admissible_fraction','admissible_fraction','incumbent_rho','reward_gap','last_batch_reward_gain']
    lines=['# GRID188：重复改善和重复退化到底有什么区别','',
        '结论：纯动作变化目标确实找回了一些小幅改善，但也更容易在已有明显即时收益的状态上停止。下一步应优先保留原NET的收益幅度监督，再研究动作变化辅助目标；当前证据不支持先扩大网络、补全尾部或加长强化学习时域。这个优先级来自已有案例，不是新模型已经通过验证。','',
        '## 比较对象与口径','',
        '保留GRID187全部16个旧训练日期、3个种子及222个物理调用组。重点比较全部种子均恢复完整同站动作的24个起点/8组，以及全部种子均丢失原NET完整同站动作的28个起点/17组；两组没有重叠的物理调用。按起点和首次实际策略分歧位置去重后分别为24和29个状态，种子层面共156条配对。它们是按结果选出的开发子集，不能当作新的独立测试。','',
        '首次分歧取两策略共同查到的位置：一方选择STOP，另一方继续下一批。比较到这一刻，双方的输入与已购反馈相同；未来奖励和最终动作只用于事后核对。','',
        '## 1. 主要不是看不到更远的候选','',
        '|核查项|重复改善|重复退化|','|---|---:|---:|',
        f"|去重后的首次分歧状态|{rest['unique_divergence_states']}|{lost['unique_divergence_states']}|",
        f"|补查入口即分歧|{rest['divergence_position'].get('0',0)}|{lost['divergence_position'].get('0',0)}|",
        f"|完整同站最终动作就在下一批至多64个计划中|{rest['final_in_next64']}|{lost['final_in_next64']}|",
        f"|下一批已经产生正奖励增量|{rest['first_gain_next64']}|{lost['first_gain_next64']}|",
        f"|下一批按原查询价格计算NET为正|{rest['next_NET_positive']}|{lost['next_NET_positive']}|",'',
        '绝大多数差异在补查尚未购买第一批时就出现，并非得到几批新反馈后才形成。这里仍有全局搜索的已购反馈，不能称为没有反馈。决定性最终动作多数已经位于下一批显式计划内，因此单纯补全更远尾部不是这批错误的首要解释。','',
        '## 2. 恢复的多是小收益，漏掉的包含大收益','',
        f"按物理调用组先平均、再跨组等权，首次分歧后的完整剩余奖励增量为：改善组{rest['suffix_gain_group_mean']:.9f}，退化组{lost['suffix_gain_group_mean']:.9f}，后者约{ratio:.1f}倍。这是局部一步奖励增量，不是整周成本、货币或安全指标；也不同于GRID187全量加权损失/恢复增量约11.4倍的分解口径。",'',
        f"改善组只有{rest['next_NET_positive']}/{rest['unique_divergence_states']}个状态下一批NET为正，而退化组为{lost['next_NET_positive']}/{lost['unique_divergence_states']}。原NET在改善组停止，很多时候符合它的计算价值目标；新模型恢复这些动作有动作一致收益，但不等于原NET估计错误。反过来，退化组这些即时有利的继续机会，是原NET已经识别、新目标又丢掉的。",'',
        '这支持一个具体的设计判断：不能用“是否改选”的二元答案完全替代收益幅度。仍然需要区分会改变一点和会改善很多。但这轮同时改变了标签含义、数值范围与损失对样本幅度的敏感性，尚不能断言某一种加权损失必然修复问题。','',
        '## 3. 不能只看全种子共同退化：大损失集中在三组','',
        f"把全部222组按日期等权的奖励差异分解，负贡献总量的{100*a['loss_concentration']['top3_fraction']:.2f}%来自3组："+'、'.join(d['top3_groups'])+'。这里的分母是各负贡献组的负值绝对量之和，不是净损失。','',
        '其中8月17日和9月7日，恰好属于GRID187所有种子动作一致率都提高的日期。更多小动作匹配与少数大增量被丢失可以同时发生；只看动作匹配次数会错过这种变化。','',
        '|物理调用|核对到的典型下一批奖励增量|新模型提前停的种子|','|---|---:|---|',
        '|2037-08-17_101 / call003|0.186113596|0、2|',
        '|2037-09-07_100 / call001|0.104874134|1、2|',
        '|2037-07-06_100 / call011|多个起点，约0.00647–0.04233|三种子均有，位置不完全相同|','',
        f"三组中全部{len(d['all_action_changing_pairs'])}条动作变化配对，在首次分歧处的下一批NET都为正。7月那组的最终动作有时更远，但下一批已经能获得明显增量，所以也不能只用终点较远来解释即时停止。全部起点和种子保存在top3_loss_details.json，没有只留大数值案例。",'',
        '## 4. 现有反馈能看到什么','',
        '下表按物理组等权，仅描述两个结果子集；完整41项均已输出，没有按差异大小挑选模型特征。所有范围大量重叠，均值差不能直接变成触发阈值。','',
        '|已观察量|改善组均值|退化组均值|','|---|---:|---:|']
    labels=dict(station_admissible_fraction='同站已购候选的合格比例',admissible_fraction='全部已购候选的合格比例',incumbent_rho='当前动作最大负载率',reward_gap='当前前两名奖励差',last_batch_reward_gain='最近已购批次的奖励增量')
    for k in selected:lines.append(f"|{labels[k]}|{rest['all41_observed_group_balanced'][k]['physical_group_mean']:.6f}|{lost['all41_observed_group_balanced'][k]['physical_group_mean']:.6f}|")
    lines+=['',
        '退化组平均合格比例更低、当前负载率与前两名奖励差更高，但存在明显交叉，尚没有得到可靠的简单开关。两组下一批的每个单独规范化接线计划，在对应折的拟合日期中都出现过；改善组41项摘要均未越过拟合范围，退化组只有两个去重状态出现单项越界。这排除了“几乎全是新计划或单项越界”的粗略解释，不能证明联合反馈模式已经充分覆盖。','',
        '因此，当前更应检查如何利用已有反馈估计收益的重要程度，而不是先假定增加输入规模就会解决。原NET与新模型输入相同，原NET能继续的这些高收益案例，应作为下一轮保留价值信号的见证。','',
        '## 下一步的最小方向','',
        '保留原NET主模型和收益幅度目标，把动作变化标签作为候选辅助监督，而非直接替换。下一轮设计应同时看原NET已识别的大收益机会是否保住，以及这8组小幅动作恢复能否继续出现；不要求全日期全赢，也不能只报告恢复次数。','',
        '先明确辅助目标接在哪、损失量纲如何匹配、如何与原NET保持训练和输入预算可比，再通知具体训练范围。当前没有加头、改损失、拟合新权重、调门槛或运行新闭环；不先叠一个新的硬规则来绕过本次反例。','',
        '## 核验与材料','',
        f"156条策略分歧由原OOF分数和保存动作核对，去重为53状态；最终动作是否在下一批同时由候选编号和教师选择位置交叉核对。另直接读取{sourceaudit['native_source_calls']}份原教师调用，重算{sourceaudit['checks']['native_starts']}个起点、{sourceaudit['checks']['native_batches']}个批次的严格可接纳候选选择、float32奖励增量和下一批候选顺序，逐项重现。没有构建电网环境、发新预测或进行独立AC求解。",'',
        'analysis.json给出全部41项摘要、222组贡献、8/17组明细；unique_divergence_states.json与events_all_seeds.json保留逐状态和逐种子记录。native_feedback_audit.json保留原调用路径与SHA；所有反向结果及原GRID187冻结材料不改。']
    (O/'GRID188_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('decision.json',dict(priority='Retain magnitude-sensitive NET supervision; define action-change auxiliary learning before any new fit.',not_primary_explanation=['missing final action beyond next64 for most repeated failures','lack of delayedRLcredit for top3immediate profitable losses','unseen single plan or single-feature range for most focus states'],not_proven='No causal loss-encoding attribution or validated trigger; outcome-selected descriptive evidence.',no_automatic_training=True,new_physics=0))
    C.save('publication_review.json',dict(passed=True,all222group_contributions_independently_reconstructed=True,all156seed_events_and53unique_states=True,all23top_loss_pairs_immediate_NET_positive=True,report_scope='Local recorded reward,not full-week quality',new_fits=0,new_model_forwards=0,new_physics=0))
    C.save('publisher_freeze.json',{p.relative_to(R).as_posix():C.sha(p) for p in C.W.glob('*.py')});C.verify()
    C.save('complete.json',dict(passed=True,focus_starts=52,unique_divergence_states=53,paired_seed_events=156,native_starts_checked=62,native_batches_checked=140,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_重复改善机制_GRID188';dest.mkdir(exist_ok=False)
    for name in ['GRID188_report.md','analysis.json','decision.json','review.json','native_feedback_audit.json','publication_review.json','complete.json']:shutil.copy2(O/name,dest/name);assert C.sha(O/name)==C.sha(dest/name)
    msg='GRID188只读机制核查完成：24恢复起点/8组与28丢失起点/17组，去重24/29分歧状态；23/24和28/29最终动作已在下一批，23/24与27/29在补查入口分歧，非主要尾部不可见。下一批NET正为1/24对18/29；丢失组剩余奖励按物理组均值约16.1倍，两个正向日期也有巨大损失。全部222组分解负贡献51.01%集中3组，23相关配对即时NET全正；保留原NET收益幅度监督，动作变化可做辅助设计，尚不训练。原28教师调用/62起点/140批重现，零模型前向/物理/最终8。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID188重复改善与大损失机制\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a0,b0=t.split('\n',1);t=a0+'\n'+entry+'\n'+b0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID188后\n\n1. 原NET保留为主模型。先设计保留收益幅度、动作变化辅助的最小学习对照，明确模块位置、损失量纲与同预算性；具体训练范围通知在拟合前。当前未授权新物理范围，不自动训练或新增硬触发规则。\n2. 全部16日期/3种子保留，重点同时核对8组重复动作恢复与大损失见证；不要求全赢，不把动作匹配次数替代奖励幅度或整周成本。目标编码和类别估计的因果作用尚未分开。\n3. 当前证据不优先支持补全尾部、扩网络或加长RL时域。最终8未生成，旧数据为教师路径；论文、第二网和真实闭环另行确定范围。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID188重复改善机制\r\n\r\n[报告](protocols/GRID188_repeated_cases/complete/GRID188_report.md)；同批已见高收益被误停，保留NET收益监督，尚未新训练或物理。\r\n\r\n'.encode('utf8')+p.read_bytes())
    print('GRID188_COMPLETE',json.dumps(dict(reward_ratio=ratio,top3_negative_share=a['loss_concentration']['top3_fraction'],new_fits=0,new_forwards=0,new_physics=0)))
if __name__=='__main__':main()
