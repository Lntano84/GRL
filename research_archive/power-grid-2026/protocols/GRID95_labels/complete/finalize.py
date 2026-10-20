"""Summarize qualified single-intervention labels; learning remains paused."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid95_labels';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def append(p,key,text):
    s=p.read_text(encoding='utf-8')
    if key not in s:p.write_text(s.rstrip()+'\n\n'+text+'\n',encoding='utf-8')


def main():
    review=read(OUT/'review.json');assert review['passed_saved_data']
    labels=read(OUT/'labels.json');cases=read(OUT/'case_results.json');D=read(OUT/'design.json')
    names=dict(C0_SAME_ACTION='同动作校验',C1_LOW_RHO='旧七月低负载率',C2_OVERLOAD='旧七月尚存过载',C3_COST_REVERSE='生成七月成本反例')
    lines=['# GRID95：单次补查价值标签研究', '',
        '2026-10-20。学习已暂停；本阶段没有分类器、神经网络或强化学习拟合。冻结提交38c78c55。', '',
        '## 这次回答什么', '',
        '标签应描述一次具体补查的作用，而不是把整周策略的优劣分发给这一周的每次调用。我们固定4个已有状态，只取消其中那一次补查，之后仍用原FRAME控制。两条原FRAME完整重现用于资格核对，另4条为单次取消，共6条串行回放。', '',
        '**主要发现：** 第100步同动作补查可省29查询，整周路径不变；第1901步取消补查后1906步提前结束，原版完成2017步；生成七月第113步单次取消后成本少78078.900、后续总预测少699次。旧七月第709步虽然来自原整策略对比受益场景，单独取消这一次却使成本少1754.908、预测少177次。整策略的受益不能直接当作每一次补查的受益标签。', '',
        '主评价时域为从干预开始到原场景终点，同时固定报告一步、一小时和一天。先看存活，再比较共同达到评价终点时的运行成本；查询量单列。没有拟合运行成本与计算代价的换算权重，也没有选择有利窗口。', '',
        '## 1. 标签能否复现', '',
        '两条新原FRAME轨迹逐步重现历史动作、观测、原始成本和查询。每条取消分支从头运行相同环境、代理及完整前缀，核对干预前全部物理步骤、公共基础动作、全局查询反馈，以及已审阅的控制器/优化参数与对象随机状态见证。求解器原设定warm_start=False保持。', '',
        '不是仅恢复一条观测，也没有声称克隆了所有引擎内部对象。程序只记录数值状态以核对，不把这些内部记忆交给学习器。未显式设种、且原控制未使用的全局随机状态单独记录，不伪称相同。外部输入时序冻结，实际对手攻击逐步对照。', '',
        '原始内存整体哈希检查曾失败：5类子模块继承了未显式设种、但执行路径没有读取的随机对象。原失败日志和见证均保留，audit_memory_v2.py只将这些无效对象从一致性比较中分离；根代理、环境、对手的对象随机状态及全部其余数值记忆仍须相同。源码检查与8组比较见audit_memory_v2_receipt.json，没有重跑或修改控制。convex_optim.py依赖源码未列入继承的原冻结清单，本次另存事后哈希作静态核对，不伪称此前已封存该依赖。', '',
        f"核验：{review['checks']['steps']}个物理步骤、{review['checks']['vectors']}个保存向量哈希、{review['checks']['independent_calls']}次独立策略复算、{review['checks']['feedback']}条候选反馈；4组干预前数值记忆和反馈一致。同动作校验分支省略29个候选查询后，整周物理轨迹与成本完全一致。", '',
        '## 2. 剩余整周的单次补查标签', '',
        '下表Δ成本=不补查−补查，正值表示保留补查的运行成本更低；Δ查询=补查−不补查，正值表示保留补查使用更多实际一步预测。成本仅在两条路径都完成整周时比较。', '',
        '|固定案例|干预步|原FRAME/单次取消存活步数|Δ成本|Δ全部预测查询|成本与计算关系|',
        '|---|---:|---:|---:|---:|---|']
    primary=[]
    for c in cases:
        p=next(l for l in labels if l['case_id']==c['id'] and l['horizon']=='remaining_week');primary.append(p)
        cost='不可直接比较' if p['delta_cost_skip_minus_complete'] is None else f"{p['delta_cost_skip_minus_complete']:+.3f}"
        q='—' if p['delta_all_queries_complete_minus_skip'] is None else f"{p['delta_all_queries_complete_minus_skip']:+d}"
        lines.append(f"|{names[c['id']]}|{c['step']}|{c['complete_steps']}/{c['skip_steps']}|{cost}|{q}|{p['label']}|")
    lines += ['', '标签解释：COMPLETE_SURVIVAL_ADVANTAGE表示保留补查才能达到本次评价终点；SKIP_COST_COMPUTE_DOMINATES表示取消在运行成本与计算两个维度上占优；COST_COMPUTE_TRADEOFF表示一个维度改善、另一个变差；SAME_COST表示所记录成本在纯数值求和容差内相同。过载步数等运行风险指标仍单列，不能把成本/计算占优扩大为所有安全维度占优。', '',
        '原“始终补查”和“始终不补查”的整周比较，仅帮助选定这些已知案例。表中是新的、只取消一次补查的结果；不能用旧整策略成本替代它。', '',
        '## 3. 短期标签是否与整周一致', '',
        '|案例|评价时域|共同窗口步数|Δ成本|Δ全部预测查询|标签|', '|---|---|---:|---:|---:|---|']
    for p in labels:
        cost='不可比' if p['delta_cost_skip_minus_complete'] is None else f"{p['delta_cost_skip_minus_complete']:+.3f}"
        q='—' if p['delta_all_queries_complete_minus_skip'] is None else f"{p['delta_all_queries_complete_minus_skip']:+d}"
        lines.append(f"|{names[p['case_id']]}|{p['horizon']}|{p['actual_horizon_steps']}|{cost}|{q}|{p['label']}|")
    lines += ['', '一步、一小时、一天和剩余整周不当作四个独立样本。靠近周末的案例会按原终点截断，截断后的相同窗口不重复计证据。若短期省成本却后来提前失败，不能据短期成本给出“应取消”的整周标签。尤其过载例：取消当步成本少148.789，但实际下一状态ρ升到1.09796，原版为0.99635；当步风险指标已经有差异，不能只训练成本标签。', '',
        '## 4. 运行风险与外部事件', '',
        '|案例|评价时域|原/取消实际过载步数|原/取消负载峰值|实际攻击差异步数|', '|---|---|---:|---:|---:|']
    for p in primary:
        a,b=p['complete_branch'],p['skip_branch'];peak=lambda x:'—' if x is None else f'{x:.4f}'
        lines.append(f"|{names[p['case_id']]}|剩余整周|{a['overload_steps']}/{b['overload_steps']}|{peak(a['max_rho'])}/{peak(b['max_rho'])}|{len(p['external_attack_differences'])}|")
    lines += ['', '未达到共同终点的分支，其过载累计与成本均不可当作相同时域质量比较。实际攻击若不同，对应固定外部事件标签标记UNRESOLVED_EXTERNAL_DIFFERENCE，原始结果仍保留。', '',
        '## 5. 当前应保留什么样的标签', '',
        '第一层是可核对的数值结果：是否完成、存活步数、共同窗口运行成本差、候选与全部预测查询差、实际过载指标。', '',
        '第二层才是条件解释：保留补查是否避免提前失败；在同等存活下，是保留或省略在成本和计算上占优，还是需要权衡。不同维度不一致时保留多维标签，不强行拼成一个好坏分数。', '',
        '这些标签回答的是：在这个状态、这条外部过程和之后继续原FRAME的条件下，取消这一次补查会怎样。它们不直接等于“某类状态永远应该补查”，也不适用于所有未来门控策略。把标签用于未来学习之前，仍需确定优化目标、固定数据采集协议并按场景隔离检验。', '',
        '本阶段只有4个定向案例、2个已暴露日期；它们用于检验标签流程和解释矛盾，不是足够的训练集，更不证明跨电网泛化。没有新增模型、预算、阈值、排序或触发规则；没有独立AC重求解。', '',
        '## 复核入口', '',
        'design.json及code_freeze.json/data_freeze.json记录方案和源码/模型/输入；qualification.json记录两条原轨迹复现；各分支保存完整物理向量、逐步成本、候选反馈、干预见证与日志；labels.json/labels.csv为逐状态逐时域原始标签；case_results.json、review.json和decision.json记录判读。', '',
        '重现方法参考Grid2Op官方环境与代理文档，实际资格以本地轨迹核验为准：[Environment](https://grid2op.readthedocs.io/en/latest/user/environment.html)、[Agent](https://grid2op.readthedocs.io/en/latest/user/agent.html)。']
    (OUT/'GRID95_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    note='学习暂停。固定4状态、2条原轨迹重现及4条只取消一次补查的完整前缀对照已核验；干预前动作/观测/反馈与审阅数值记忆一致，同动作取消29查询后整周轨迹一致。主标签为剩余整周存活/成本/查询向量，短期窗口均保留，权衡不强制二分；不把整策略结果贴给单次调用。具体正负与存活结论见GRID95_report.md，无新拟合。'
    for n in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:append(REPO/n,'GRID95_LABELS_complete','## 2026-10-20 — GRID95_LABELS_complete\n\n'+note)
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines();p.write_text(lines[0]+'\n\n- **GRID95_LABELS_complete：** '+note+'\n\n'+'\n'.join(lines[1:]).lstrip()+'\n',encoding='utf-8')
    p=REPO/'docs/NEXT_STEPS.md';s=p.read_text(encoding='utf-8');lines=s.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('1. '):lines[i]='1. 用户要求学习暂停。GRID95单次干预标签流程已完成4例核验，先向用户汇报短期/整周标签、存活与成本计算权衡。下一步只研究标签目标与采集覆盖，不自动训练、批量扩样、改门控或换网。';break
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    append(REPO/'research_archive/power-grid-2026/README.md','GRID95_LABELS_complete','## GRID95_LABELS_complete\n\n'+note+'\n\n[报告](protocols/GRID95_labels/complete/GRID95_report.md)。')
    dest=REPO/'research_archive/power-grid-2026/protocols/GRID95_labels/complete';dest.mkdir(exist_ok=False)
    # Full numeric traces are zipped with an integrity manifest; no runtime or third-party source copied.
    import zipfile
    runfiles={}
    for path in sorted((OUT/'runs').rglob('*')):
        if path.is_file():runfiles[path.relative_to(OUT).as_posix()]=sha(path)
    save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save(OUT/'run_files.json',runfiles)
    archives={}
    for directory in sorted((OUT/'runs').iterdir()):
        members={rel:h for rel,h in runfiles.items() if rel.startswith('runs/'+directory.name+'/')}
        archive=dest/(directory.name+'.zip')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for rel in members:z.write(OUT/rel,rel)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for rel,h in members.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h
        archives[archive.name]=dict(sha256=sha(archive),size=archive.stat().st_size,members=len(members))
    for p in list(OUT.iterdir())+list(W.glob('*.py')):
        if p.is_file():shutil.copy2(p,dest/p.name);assert sha(p)==sha(dest/p.name)
    if (OUT/'interrupted_audits').exists():shutil.copytree(OUT/'interrupted_audits',dest/'interrupted_audits')
    collab=ROOT/'论文协作资料/FRAME_单次补查标签_20261020';collab.mkdir(exist_ok=True)
    for n in ['GRID95_report.md','标签定义与使用边界.md','labels.json','labels.csv','case_results.json','review.json','decision.json']:
        shutil.copy2(OUT/n,collab/n);assert sha(OUT/n)==sha(collab/n)
    complete=dict(completed=True,new_fits=0,physical_replays=6,cases=4,
        report_sha256=sha(OUT/'GRID95_report.md'),archive=str(dest),run_archives=archives,run_files=len(runfiles),
        collaboration_copy=str(collab),label_scope='Single omission under unchanged original continuation; no universal classifier labels or training deployment.')
    save(OUT/'complete.json',complete);shutil.copy2(OUT/'complete.json',dest/'complete.json');shutil.copy2(OUT/'complete.json',collab/'complete.json')
    print(json.dumps(complete,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
