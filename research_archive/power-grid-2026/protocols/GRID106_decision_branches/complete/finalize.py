"""Report all two-branch outcomes, preserve vectors locally, archive reproducible records."""
import shutil,zipfile
from common106 import ROOT,W,OUT,REPO,ARCHIVE,read,save,sha,verify,utc
def main():
    assert not (OUT/'complete.json').exists();verify()
    D=read(OUT/'design.json');R=read(OUT/'review.json');assert R['passed']
    cases=read(OUT/'case_results.json');labels=read(OUT/'labels.json');decision=read(OUT/'decision.json')
    lines=['# GRID106：两次关键补查的单次干预核查','',
        '本轮已完成一条FRAME原轨迹资格重现及两条单次取消回放。只研究已暴露的2037-03-02_100；不训练、不调参、不新增场景或最终测试。',
        '', '## 研究问题','',
        'GRID104中RF从第108步开始偏离FRAME，DQN种子0/1从第552步开始偏离。原整策略分别还有后续学习决定，不能把整周1.112%或1.758%的成本差直接贴给首次补查。本轮各分支只取消一个固定时刻的全部同站补查，后续都使用原FRAME。',
        '', '## 干预是否可比','',
        f"新FRAME的2017步动作、观测、逐步成本、查询数及已支付候选反馈精确重现GRID104。两分支分别逐步重现干预前107/551步；干预观测、基础动作、全局候选反馈及经审阅的控制器/优化数值与已设种对象随机状态一致。审计核对{R['checks']['steps']}个物理步、{R['checks']['vectors']}个保存向量哈希、{R['checks']['policy_replays']}次策略复算、{R['checks']['feedback']}项反馈合法性。",
        '', '重放完整前缀是状态重建方法，没有仅靠一条观测恢复环境，也没有声称克隆所有求解器内部对象。沿用GRID95已检查的五个未使用且未显式设种子模块RNG排除规则；根代理、环境和对手状态仍须一致。原求解器warm_start=False、logger属性工程修复、控制器和缓存均不变。',
        '', '## 单次取消的结果','']
    for c in cases:
        ps=[l for l in labels if l['case_id']==c['case_id']];p=next(x for x in ps if x['horizon']=='remaining_week')
        dc=p['delta_cost_skip_minus_frame'];dq=p['delta_all_queries_frame_minus_skip']
        lines += [f"### 第{c['step']}步",'',
            f"FRAME存活{c['frame_steps']}步，单次取消分支存活{c['skip_steps']}步；是否完成整周分别为{c['frame_complete']}、{c['skip_complete']}。实际执行动作首次差异在第{c['first_action_difference']}步。"]
        if dc is not None:
            lines += [f"原FRAME整周成本{c['frame_week_cost']:.6f}，单次取消为{c['skip_week_cost']:.6f}；取消−保留的剩余整周成本差为{dc:+.6f}，相对FRAME整周成本{dc/c['frame_week_cost']:+.4%}。保留−取消的实际全部预测查询差为{dq:+d}；正数表示取消更省查询。",'']
        else:lines+=['分支未到达共同整周终点，不能用其较短累计成本声称更省。','']
        for x in ps:
            delta=x['delta_cost_skip_minus_frame'];query=x['delta_all_queries_frame_minus_skip']
            cd='不可比' if delta is None else f'{delta:+.6f}'
            qd='不可比' if query is None else f'{query:+d}'
            lines.append(f"- {x['horizon']}（{x['horizon_steps']}步）：取消−保留成本{cd}，保留−取消全部查询{qd}；{x['label']}。实际过载步数FRAME/取消为{x['frame']['overload_steps']}/{x['skip']['overload_steps']}，负载峰值{x['frame']['max_rho']}/{x['skip']['max_rho']}。")
        lines+=['',f"剩余整周实际对手攻击差异步数：{len(p['external_opponent_difference_steps'])}。该标签只在共同FRAME后续策略下解释这一次补查；短窗口是同一干预的不同观察，不是独立样本。",'']
    lines+=['## 与学习目标的关系','',
        '第108步的原固定即时费用下，已保存候选反馈中没有正的分批停止点；第552步下一批13个候选有0.00693335的即时净收益。这些即时数字保持原值，没有为新结果更改价格或阈值。',
        '本轮单次干预结果应与上述即时标签逐项对照：若方向不同，说明即时目标不能替代该次决策的长期价值；若单次取消与整策略结果不同，则原来的整策略差额还包括后续门控决定及路径交互。两种情况都保留，不能靠挑窗口或挑种子把它们消掉。',
        '', '## 训练边界与资料','',
        '本轮模型拟合为0，最终测试未生成。本报告是开始下一轮训练前的通知材料；不自动启动训练或扩大规模。这两处已按失败结果选定，是开发解释案例，不能直接当成新增独立训练/测试数据。是否修改长期目标及如何安排同架构监督对照，应依据本轮结果另行固定方案。',
        '完整物理向量、搜索输入、日志及冻结方案保存在D盘归档；GitHub保存源码、方案、逐步记录、反馈、审计和全量文件哈希，较大的NPZ向量通过哈希定位本地ZIP。诊断时间包含状态见证和记录开销，不主张本轮提速，也不与历史时间混算。',
        'audit.py只核对保存向量、合法反馈、策略、状态前缀和成本账本，没有发预测，不是独立AC重求解。']
    (OUT/'GRID106_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    local=OUT/'archive';local.mkdir(exist_ok=False)
    files=[p for p in OUT.rglob('*') if p.is_file() and local not in p.parents]
    files+=list(W.glob('*.py'))
    mapping={p.relative_to(ROOT).as_posix():sha(p) for p in files};save(OUT/'full_file_manifest.json',mapping)
    archive=local/'GRID106_complete.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[OUT/'full_file_manifest.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    dest=ARCHIVE/'complete';dest.mkdir(exist_ok=False)
    for p in list(W.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file()]:shutil.copy2(p,dest/p.name)
    for folder in (OUT/'runs').iterdir():
        target=dest/'runs'/folder.name;target.mkdir(parents=True)
        for p in folder.iterdir():
            if p.suffix in ['.json','.gz']:shutil.copy2(p,target/p.name)
    save(dest/'local_archive_receipt.json',dict(path=str(archive),sha256=sha(archive),zip_crc_passed=True,files=len(files)+1,
        raw_npz_uploaded=False,reason='Large physical vectors remain in the verified D archive with all file hashes.'))
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    collab=ROOT/'论文协作资料/FRAME_两次补查长期价值_GRID106_20261022';collab.mkdir(exist_ok=False)
    for n in ['GRID106_report.md','review.json','decision.json','labels.json','case_results.json']:shutil.copy2(OUT/n,collab/n)
    for p,h in mapping.items():assert sha(ROOT/p)==h,p
    save(OUT/'complete.json',dict(passed=True,utc=utc(),runs=3,new_model_fits=0,final_test_materialized=False,
        training_not_started=True,notify_before_training=True,archive=str(archive),collaboration=str(collab),report_sha256=sha(OUT/'GRID106_report.md')))
    save(OUT/'pipeline_state.json',dict(phase='complete_no_training',completed=3,planned=3,utc=utc()))
    shutil.copy2(OUT/'complete.json',dest/'complete.json');shutil.copy2(OUT/'complete.json',collab/'complete.json')
    # Update exact archive manifests after the final receipt; the frozen protocol is unchanged.
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'})
    summary='; '.join(f"{c['case_id']} steps FRAME/skip{c['frame_steps']}/{c['skip_steps']}, whole-week cost FRAME/skip{c['frame_week_cost']:.6f}/{c['skip_week_cost']:.6f}" for c in cases)
    note='\n\n## 2026-10-22 — GRID106 two single-decision branches completed\n\nUser requested careful March108/552 diagnosis and notification before training. One factual qualification and two single-omission continuations under unchanged FRAME completed; all prefixes, saved vectors, paid feedback, policies and numeric/used RNG witnesses checked. '+summary+'. Full multihorizon labels retained. These are two selected exposed cases on one date, not new independent training/test data or scaling. No model fitting, parameter changes or final-test generation. Report archived under GRID106_decision_branches/complete; full vector ZIP remains on D with published hashes. Training must not start before notifying user.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID106核查完成，训练未启动\n\n1. 先向用户汇报两个单次干预的长期结果及其与即时奖励的关系；开始训练前通知用户，不自动拟合。\n2. 后续如做同输入/同隐藏层直接净收益回归，必须另行固定训练与评价方案，不能把当前两处验证错误加入训练。\n3. GRID104扩大资格仍为False；最终8测试场景未生成，不恢复旧自动化，不扩大日期/网络/电网。\n\n报告：research_archive/power-grid-2026/protocols/GRID106_decision_branches/complete/GRID106_report.md\n',encoding='utf8')
    print(read(OUT/'complete.json'),flush=True)
if __name__=='__main__':main()
