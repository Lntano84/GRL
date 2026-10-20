"""Report the frozen experiment without selecting a model or deploying a gate."""
from pathlib import Path
from collections import Counter
import json,hashlib,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert read(OUT/'review.json')['passed_saved_data'] and read(OUT/'learning_review.json')['passed']
    labels=read(OUT/'labels.json');results=read(OUT/'learning_results.json');D=read(OUT/'design.json')
    decision=dict(label_collection_qualified=True,learning_diagnostic_completed=True,states=len(labels),primary_counts=dict(Counter(r['label'] for r in labels)),
        fixed_threshold_results=[dict(scheme=r['scheme'],model=r['model'],result=r['headline']['0.95']) for r in results if r['model'] in ['LOGISTIC','FOREST64']],
        no_model_or_threshold_selected=True,learned_gate_deployed=False,new_gated_closed_loop_runs=0,
        scope='Conditional individual omission values under original FRAME continuation. Never aggregate overlapping interventions as deployable policy savings. Same public grid and exposed dates; pilot diagnostics excluded headline.')
    save(OUT/'decision.json',decision)
    lines=['# GRID96：单次补查价值标签与简单模型学习','',
        '## 问题与协议','',
        '本轮研究能否仅凭补查前已揭示的反馈，判断取消这一次补查是否值得。固定抽取56个状态，覆盖21条已有场景记录、17个日历日期。每条记录取前、中、后位置，并保留补查前仍过载的状态；不按新反事实结果取样。21条事实重现和53条单次取消共74条新回放，另复用3条已核验试点。已知试点不计主要预测成绩。',
        '','原控制器、模型、候选顺序及128/64/2/0.9不变。每个分支只取消一次同站补查，此后恢复原FRAME；从初始状态完整重放前缀，核对实际状态、基础动作、数值记忆和已查询反馈。优化器依赖在本轮新回放前补入冻结。',
        '','## 标签','',
        '主时域为计划整周末（2017步），不是事实分支提前失败时刻。存活优先；两条分支都完整时，同时比较运行成本、全部预测查询、过载步数、最大负载率和累计过载量。取消分支各项不差且至少一项改善记SKIP，补查分支占优记KEEP；互有得失记MIXED。整条物理路径完全相同且取消后查询不多时记SKIP等价路径；双方不同路径均提前终止或共同外部攻击不一致时保留为UNRESOLVED。',
        '',f"全部标签：{dict(Counter(r['label'] for r in labels))}。未见新结果的主要评估样本标签：{dict(Counter(r['label'] for r in labels if not r['pilot_known']))}。"]
    for cohort in ['old13','G1','G2']:
        rr=[r for r in labels if r['cohort']==cohort];lines+=['',f"{cohort}：{len(rr)}状态，{dict(Counter(r['label'] for r in rr))}；两分支都完成整周{sum(r['completion_complete'] and r['skip_complete'] for r in rr)}例，取消导致未完成{sum(r['completion_complete'] and not r['skip_complete'] for r in rr)}例。"]
    lines+=['','## 学习与评价','',
        '使用15个补查前反馈特征。逻辑回归C=1；随机森林固定64棵树、最大深度4、叶节点至少3例。训练按日历日期等权；归一化只用训练折。按日历日期留一，G1/G2同日必须同时留出；另报告old13训练、G1/G2检验。固定P(SKIP)≥0.95且SKIP概率最大才建议取消，0.5仅作诊断。不搜索超参数或阈值；UNRESOLVED不参加拟合。',
        '','下列AUC只衡量SKIP识别；命中SKIP是相对本轮明确的条件标签，不意味着可连续部署。宏F1包含KEEP/SKIP/MIXED三类。']
    for r in results:
        h=r['headline'];q=h.get('0.95',{});p=h.get('0.5',{});lines+=['',f"- {r['scheme']} / {r['model']}：n={h['n']}；AUC={h.get('skip_auc')}；宏F1={h.get('macro_f1')}；平衡准确率={h.get('balanced_accuracy')}。0.95建议取消{q.get('proposed_skips',0)}例，正确SKIP {q.get('correct_skips',0)}，漏KEEP {q.get('missed_keep',0)}，MIXED {q.get('disputed_mixed',0)}；0.5建议取消{p.get('proposed_skips',0)}例，正确{p.get('correct_skips',0)}，漏KEEP {p.get('missed_keep',0)}，MIXED {p.get('disputed_mixed',0)}。"]
    lines+=['','## 固定高置信建议的逐例后果','']
    lookup={r['case_id']:r for r in labels};pred=read(OUT/'predictions.json')
    for r in pred:
        if r['model'] not in ['LOGISTIC','FOREST64'] or r['pilot_known'] or r['prediction']!='SKIP' or r['probabilities'][1]<.95:continue
        c=lookup[r['case_id']];lines+=[f"- {r['scheme']}/{r['model']} {c['case_id']} {c['week']} step{c['step']}：{c['label']}，整周成本差（取消−补查）{c['cost_skip_minus_completion']}，查询节省{c['queries_saved']}，两分支步数{c['completion_steps']}/{c['skip_steps']}。"]
    lines+=['','## 判读边界','',
        '这是固定小样本的可预测性检验。64棵浅树增加了表达能力，没有增加闭环部署证据。所有模型、阈值、差结果和权衡样本保留；不以一次分组检验的少量正确建议宣称可靠取消策略，不把各次单独取消的收益相加为整套策略收益。',
        '','17个日历日期来自同一公开电网且已暴露，生成种子与重复不是独立电网。标签依赖原FRAME后续策略及固定外部时序；相同路径的提前失败只支持取消的路径等价，不支持整周安全。未做独立AC重求解；数值/随机数见证不等于序列化所有不透明引擎对象。',
        '','数据与预测见labels.csv、labels.json、predictions.json、learning_results.json；审计见review.json、learning_review.json。']
    (OUT/'GRID96_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Lossless per-run archives: retain all physical evidence; no source data copies.
    dst=REPO/'research_archive/power-grid-2026/protocols/GRID96_value_learning/complete';dst.mkdir(parents=True,exist_ok=True)
    index=[]
    for folder in sorted((OUT/'runs').iterdir()):
        target=dst/(folder.name+'.zip')
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(folder.iterdir()):
                if p.is_file():z.write(p,p.name)
        members={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==members
        index.append(dict(run=folder.name,sha256=sha(target),bytes=target.stat().st_size,members=members))
    save(OUT/'archive_index.json',index)
    collab=ROOT/'论文协作资料/FRAME_补查价值学习_20261020';collab.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and (p.suffix in ['.json','.csv','.md','.log']):shutil.copy2(p,dst/p.name)
    for p in W.glob('*.py'):shutil.copy2(p,dst/p.name)
    for n in ['GRID96_report.md','labels.csv','learning_results.json','decision.json']:shutil.copy2(OUT/n,collab/n)
    save(OUT/'complete.json',dict(completed=True,new_replays=74,states=56,report='outputs/grid96_value_learning/GRID96_report.md',archive=dst.relative_to(REPO).as_posix(),collaboration=collab.relative_to(ROOT).as_posix(),remote_verification='Must compare local and remote SHA after pushing; this file alone does not assert publication'))
    shutil.copy2(OUT/'complete.json',dst/'complete.json')
    note='GRID96单次补查价值学习完成。'+str(decision['primary_counts'])+'；全部74条新回放及3复用分支保存数据核验，逻辑回归和64浅树固定日期隔离评估。见protocols/GRID96_value_learning/complete/GRID96_report.md；未部署学习门控，不相加单次收益。'
    for name in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md','docs/RESEARCH_STATE.md']:
        p=REPO/name;p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n\n## 2026-10-20 — GRID96_VALUE_LEARNING_complete\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/NEXT_STEPS.md';p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n\nGRID96已在用户重新授权下完成固定简单模型诊断；本轮授权已完成，不自动部署门控或追加状态/阈值。\n',encoding='utf-8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf-8').rstrip()+'\n\n- GRID96：单次补查价值标签与逻辑回归/64浅树固定评估，见`protocols/GRID96_value_learning/complete/GRID96_report.md`。\n',encoding='utf-8')
    print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
