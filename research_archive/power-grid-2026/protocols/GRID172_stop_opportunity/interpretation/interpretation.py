"""Evidence-led next step; retain recurring positive cases without repeating old losses."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid172_stop_opportunity';REPO=R/'work/GRL-results-integration';BASE=REPO/'research_archive/power-grid-2026/protocols'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    k=read(O/'key_changes.json');m=read(R/'outputs/grid170_remaining_date_folds/key_changes.json');t=read(R/'outputs/grid171_stop_execution_cost/key_changes.json')
    assert k['negative_pairs']==15 and k['loss_immediate']==15 and k['loss_cases_with_changed_label_sign']==0
    text=f'''# 这条小改善值得留，但下一步先不换标签

四折日期迁移后，正向节省不是只出现在原4个开发日期：16个留出日期中有{m['positive_dates']}个平均提高，48个种子×日期组合中{m['positive_seed_dates']}个提高、{m['unchanged_seed_dates']}个不变。共有{k['positive_pairs']}条配对净正，{k['correct_same_gain_action_pairs']}条同收益、同动作而少查。正向变化涉及{k['positive_physical_groups']}个物理状态组，{k['recurring_positive_groups_atleast2seeds']}组至少两个种子重现，{k['recurring_positive_groups_all3']}组在三个种子都重现。保留这条线，不用全日期全种子获胜的门槛抹掉已有改善。

整体净效用仍为{m['relative_change']:+.2%}，查询减少{m['mean_query_reduction']:.2%}，收益保留率降低{abs(m['gain_capture_percentage_points']):.4f}个百分点。{k['negative_pairs']}条净损失全部保留，它们集中在少数有利查询机会。这里是旧局部NET效用，不是整周成本或电网质量改变了同样百分比。

## 接下来为什么不先换标签或加长时域

15条损失全都漏掉了下一批本身就有正净收益的机会。把当前下一批NET目标换成原FACTOR继续路径的NET目标，在3352条继续记录中仅15条改变标签符号，在这15条损失上则一条也不改变。新标签对当前最主要错误没有直接对应关系，不能把换名字当成解决办法。

少量大收益机会被漏掉，GRID125和GRID128已经诊断过；本轮是新固定规则在日期留出后的具体重现，不是首次发现。已有GRID120/121加权分类、GRID123辅助标签、GRID126/127收益目标与损失替换的结果保留，不盲目再试一轮相似损失。

## 真正值得放大的是什么

原FACTOR路径内，事后保持同收益同动作的查询节省上界为每起点{k['potential_same_gain_action_queries_saved']:.3f}，对应净效用提升上界{k['zero_loss_net_utility_upper_relative']:.2%}；当前同收益动作节省只覆盖{k['zero_loss_opportunity_capture']:.2%}。这表明仍有可研究空间，但上界读取未来反馈，不能部署或当成绩。

下一步优先查两件事：现有公共输入是否能区分“后续没收益”与“下一批有明显收益”；原FACTOR继续判断中，哪些有利机会被两个帮助模型共同低估。先核对已有公开电网状态、动作和已购反馈的可用信息与编码，避免只在同样的摘要、标签上继续堆网络。任何新输入都必须能在查询前取得，先做来源与打包核验，再通知一组有针对性的配对训练；当前没有启动新训练。

## 已经可以保留的实现改进

同一单向规则用便宜小模型先判断、必要时再运行结构模型，模型头耗时相对三个模型每次全运行减少{t['short_vs_eager_head_reduction']:.2%}，所有停止决定、查询和动作完全相同。相对原单模型，每起点额外{t['extra_head_ms_vs_factor']:.4f}毫秒。这是等价执行优化，可以保留；尚未包含完整特征构造和物理预测，不能说控制时间已经改善。

不扩大物理场景，不生成最终8，不修改论文。下一轮仍以放大反复出现的正确节省为目标，同时逐条核查收益损失；不靠挑日期、种子或新阈值救平均结果。
'''
    path=O/'阶段结论与下一步.md';path.write_text(text,encoding='utf8',newline='\n')
    dest=BASE/'GRID172_stop_opportunity/interpretation';dest.mkdir(exist_ok=False)
    for p in [Path(__file__),path]:shutil.copy2(p,dest/p.name)
    (dest/'source_results.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in [O/'key_changes.json',R/'outputs/grid170_remaining_date_folds/key_changes.json',R/'outputs/grid171_stop_execution_cost/key_changes.json']},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(path,R/'论文协作资料/FRAME_无损停止研究空间_GRID172'/path.name)
    interrupted=R/'outputs/grid172_stop_opportunity_interrupted_01';error=BASE/'GRID172_stop_opportunity/recovery01';error.mkdir(exist_ok=False);shutil.copytree(interrupted,error/'interrupted_attempt')
    (error/'archive_manifest.json').write_text(json.dumps({p.relative_to(error).as_posix():sha(p) for p in error.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
    nextsteps='''# Next steps — 放大正确节省，识别有利机会

1. 保留GRID170四折12/16日期正向及226条同收益动作节省，不用全赢门槛否定；15条大收益漏停完整保留，局部效用/收益/查询分别报告。
2. GRID172已核对：原FACTOR继续结果标签与下一批标签只在15/3352状态改变符号，15个损失均不改变。暂不优先换标签/损失或加长RL时域，不重复旧120—128实验。
3. 下一步只读核对原始公共电网状态、动作及已购反馈的信息与编码，解释两个帮助模型共同漏掉有利下一批的原因；新输入先证明查询前可得、打包可复现，再通知配对训练。当前没有启动下一轮拟合。
4. 保留GRID171等价短路执行：策略决定完全不变，模型头开销降低；不得称作完整控制提速。新物理/数据扩大先给范围，最终8和论文不动。
'''
    (REPO/'docs/NEXT_STEPS.md').write_text(nextsteps,encoding='utf8',newline='\n')
    msg='GRID170—172统一判读：保留12/16日期正、226同收益动作节省/107正向物理组（20至少双种子）；15损失全下一批净正。基线继续目标改标签只15/3352符号变且15损失均不变，暂不优先换标签/长时域RL；与125/128已知稀有机会问题对应。下一步核对查询前公共物理状态/动作/已购反馈编码，先来源打包再通知训练；不否定小改善、不继续同损失重复。短路等价执行保留，整周质量/完整控制提速未建立。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID172阶段解释与下一步收敛\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    print('GRID172_INTERPRETATION_COMPLETE')
if __name__=='__main__':main()
