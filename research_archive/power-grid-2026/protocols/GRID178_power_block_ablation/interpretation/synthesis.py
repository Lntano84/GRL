"""Self-contained stage summary and artifact checks for GRID173--178."""
from pathlib import Path
import datetime,hashlib,json,shutil
R=Path('D:/Codex/GRL研究');REPO=R/'work/GRL-results-integration';BASE=REPO/'research_archive/power-grid-2026/protocols';O=R/'outputs/grid178_power_block_ablation'
TOPICS=['grid173_public_state','grid174_power_helper','grid175_helper_gate_audit','grid176_gate_support','grid177_support_mixture','grid178_power_block_ablation']
COLLAB=['FRAME_公共电气输入核对_GRID173','FRAME_电气输入帮助模型_GRID174','FRAME_小模型改善与组合差异_GRID175','FRAME_组合决策区域覆盖_GRID176','FRAME_决策区域混合采样_GRID177','FRAME_电气输入分块消融_GRID178']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert read(O/'complete.json')['passed_audit'];h=read(R/'outputs/grid175_helper_gate_audit/key_changes.json');mix=read(R/'outputs/grid177_support_mixture/key_changes.json');abl=read(O/'key_changes.json');fam=read(O/'family_summary.json');factors=read(R/'outputs/grid174_power_helper/mechanism_counts.json')
    detail=read(O/'summary.json');state_seed=[]
    for seed in range(3):
        a=detail[f'STATE4_HELPER_seed{seed}'];b=detail[f'ZERO8_HELPER_seed{seed}']
        state_seed.append(dict(seed=seed,relative=a['macro']['mean_net_utility']/b['macro']['mean_net_utility']-1))
    state_info=dict(relative_utility_vs_zero8=abl['state4_vs_zero8'],query_reduction_vs_zero8=1-fam['STATE4_HELPER']['mean_queries']/fam['ZERO8_HELPER']['mean_queries'],gain_capture_percentage_points=100*(fam['STATE4_HELPER']['gain_capture']-fam['ZERO8_HELPER']['gain_capture']),positive_seeds=sum(v['relative']>0 for v in state_seed),seed_changes=state_seed,relative_utility_vs_factor=fam['STATE4_HELPER']['mean_net_utility']/fam['FACTOR']['mean_net_utility']-1)
    (O/'state4_detailed_changes.json').write_text(json.dumps(state_info,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    lines=['# GRID173—178：从小改善走到一个更明确的学习进展','',
        '本轮最值得保留的是：补入查询前已知的电气状态与动作描述后，便宜小模型同时减少了查询、保留了更多收益，而且三个训练种子都改善。组合规则仍有局限，不能把小模型进步与整套控制器已经更好混为一谈。','',
        f"固定四折、16日期、三个种子的配对结果：POWER8小模型相对同容量ZERO8，局部净效用提高{h['helper_relative_change']:.2%}，平均查询减少{h['helper_query_reduction']:.2%}，收益保留率提高{h['helper_gain_capture_percentage_points']:.2f}个百分点。三个种子分别提高4.83%、6.59%、2.15%，10/16日期平均正向。",'',
        '## 为什么值得保留','',
        '这次没有改变标签、损失、候选顺序、费用、训练预算或模型总参数。补入的8项中，前4项是旧GRID98已有的物理负载描述，最大负载率还完全重复旧输入；后4项把当前站功率与已知候选分区联系起来。向量/哈希、独立算术与未来字段拒绝均已核验，没有额外仿真。',
        '', '两组都是7554个总参数，初始函数相同；ZERO8的额外列梯度为0，活跃输入维度不同。结果支持这块输入在当前学习任务中的价值，不证明每项都必要或整个神经网络架构是新的。','',
        '## 用分块消融继续查原因','', '| 小模型输入 | 局部净效用 | 平均查询 | 收益保留率 |','|---|---:|---:|---:|']
    for name in ['ZERO8_HELPER','STATE4_HELPER','POWER8_HELPER']:
        v=fam[name];lines.append(f"| {name} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['gain_capture']:.2%} |")
    lines+=['',f"只保留负载STATE4，对ZERO8变化{abl['state4_vs_zero8']:+.2%}，查询减少{state_info['query_reduction_vs_zero8']:.2%}、收益保留率提高{state_info['gain_capture_percentage_points']:.2f}个百分点，{state_info['positive_seeds']}/3种子正向。三个种子相对变化分别为"+'、'.join(f"{v['relative']:+.2%}" for v in state_seed)+'。', '',f"完整POWER8再相对STATE4变化{abl['all8_vs_state4']:+.2%}，增量{abl['all8_increment_positive_seeds']}/3种子、{abl['all8_increment_positive_dates']}/16日期正向。各输入块互相作用，两个百分比不能直接相加。"]
    if abl['all8_vs_state4']>0:
        lines+=['', '该消融保留了功率—分区描述的平均增量线索，下一步值得追它具体识别了哪些有利动作，哪些状态仍有损失；不据此声称每项都必要。']
    else:
        lines+=['', '这个消融说明，补回负载状态本身就值得关注，新增功率—分区描述的平均增量尚未建立。STATE4平均收益更高，POWER8三个种子的方向更一致，两者是不同取舍；完整描述不能仅因平均值较低就被全盘否定，简版也不能仅因平均值较高就被宣布最终最优。全部结果仍是已暴露开发数据上的比较。']
    lines+=['','## 组合为什么没有同步进步','',
        f"接入原F继续 AND（R继续 OR C继续）规则后，POWER8组合相对上一版为-1.10%。旧15条漏停恢复了{factors['old15_losses_fully_restored']}条，旧226条同收益动作节省保留{factors['old226_same_gain_action_savings_preserved']}条，新增正反变化全部保存。",'',
        '小模型的改变只有在F继续且R停止的位置才能改变外层决定。它独立运行时的许多改善发生在F已经停止或R会强制继续的位置，固定外层不会采用这些新判断。因此必须将模块本身、决策区域和最终组合分别评估。',
        '',f"该区域在原训练抽样中只占3.32%—9.95%。固定采用一半原分布、一半区域条件分布后，组合相对POWER8又提高{mix['gate_relative_vs_power']:.2%}，2/3种子、8/16日期正；小模型单独运行则变化{mix['helper_relative_vs_power']:+.2%}。这只是一个小的角色适配线索，不说明覆盖问题已全部解决，也没有扫描混合比例。",'',
        f"当前混合采样组合相对原FACTOR的局部净效用仍为{mix['gate_relative_vs_factor']:+.2%}，查询减少{mix['query_reduction_vs_factor']:.2%}。因此更明确的学习进展在输入和小模型，整套组合尚未建立优势。",'',
        '## 下一步如何投入','',
        '保留已经反复出现的输入改善，按本轮分块结果追真实恢复有利机会的状态与动作，不再盲目换标签、加长强化学习时域或堆网络。组合方面先研究它为什么拒绝或误用这些有用判断，再设计一个具体改法；不从本轮留出记录挑触发阈值，不自动删除原限制。',
        '', '后续若要验证真正控制收益，需要先给出冻结候选、比较对象和物理回放范围；本轮没有扩大数据、生成最终8或修改论文。局部净效用、收益保留率、候选查询与整周质量、完整控制时间仍分别报告。',
        '', '本轮共48次新拟合：24份电气输入配对、12份固定混合采样、12份分块消融。全部最终权重在对应留出评分前发布，全部种子、日期和不利结果保留。没有重训原F/R或重复完整条目。']
    path=O/'本轮学习进展与后续重点.md';path.write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');dest=BASE/'GRID178_power_block_ablation/interpretation';dest.mkdir(exist_ok=False)
    for p in [Path(__file__),path,O/'state4_detailed_changes.json']:shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');shutil.copy2(path,R/'论文协作资料'/COLLAB[-1]/path.name);shutil.copy2(O/'state4_detailed_changes.json',R/'论文协作资料'/COLLAB[-1]/'state4_detailed_changes.json')
    nexttext='# Next steps — 保留电气输入进步，分清模型与组合\n\n1. POWER8小模型+4.36%、3/3种子、查询少10.47%、收益保留率+5.39pp为当前正向线索；STATE4消融与组合成绩单列，不抹去小改善。\n2. 根据GRID178分块结果，先只读追恢复有利机会的状态与动作，固定输入来源核对；不按单个错例挑字段、阈值或混合比例。\n3. 组合F AND(R OR C)尚未建立整体优势，研究何时采用小模型判断；新学习/组合方案先具体冻结并通知训练，不盲加网络或改标签。\n4. 不自动新物理/数据/最终8，不改论文。若提出真实控制验证，先给冻结候选、对照和回放范围。\n'
    (REPO/'docs/NEXT_STEPS.md').write_text(nexttext,encoding='utf8',newline='\n')
    msg=f"本轮173—178综合：保留POWER8小模型+4.36%效用/查询少10.47%/收益保留率+5.39pp，3/3种子、10/16日期正；STATE4对ZERO{abl['state4_vs_zero8']:+.2%}、完整8对STATE4{abl['all8_vs_state4']:+.2%}分列。混合采样组合仅+0.23%，整体未胜原F。48拟合全部权重先推送，现不自动新物理/数据/最终8；追有利机会和组合取舍，不以全赢门槛否定小改善。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID173至178学习进展\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    count=0;models=0;copies=0
    for topic,collab in zip(TOPICS,COLLAB):
        root=R/'outputs'/topic;assert read(root/'complete.json')['passed_audit'] and read(root/'review.json')['passed'];arch='GRID'+topic[4:7]+topic[7:]
        for mf in (BASE/arch).glob('*/archive_manifest.json'):
            for n,hsh in read(mf).items():assert sha(mf.parent/n)==hsh,(mf,n);count+=1
        for n in ['source_hashes.json','code_freeze.json']:
            if (root/n).exists():
                for p,hsh in read(root/n).items():assert sha(R/p)==hsh,p
        for p in (R/'论文协作资料'/collab).iterdir():
            if p.is_file():assert sha(p)==sha(root/p.name);copies+=1
        if (root/'fits.json').exists():
            for rec in read(root/'fits.json')['records']:
                name=f"fold{rec['fold']}_{rec['arm']}_seed{rec['seed']}" if 'arm' in rec else f"fold{rec['fold']}_seed{rec['seed']}"
                assert sha(root/'models'/f'{name}.pt')==rec['checkpoint_sha256'];assert rec['held_draws']==0;models+=1
    assert models==48
    result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_files_checked=count,models_checked=models,collaboration_copies_checked=copies,scope='Artifact/currenthash verification,not another experiment or physical solver rerun.')
    (O/'publication_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n');vdir=BASE/'GRID178_power_block_ablation/verification';vdir.mkdir(exist_ok=False);shutil.copy2(O/'publication_verification.json',vdir/'publication_verification.json');(vdir/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':sha(vdir/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID173_TO178_VERIFIED',json.dumps(result))
if __name__=='__main__':main()
