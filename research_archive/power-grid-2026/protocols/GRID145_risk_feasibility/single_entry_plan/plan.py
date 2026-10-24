"""Prospective single-entry proposal and proof; no scoring, fitting or policy replay."""
import json, shutil
from pathlib import Path
import run as G

def main():
    G.verify()
    O, R, REPO = G.O, G.R, G.REPO
    support = G.read(O/'risk_support_by_depth.json')
    fit128 = next(x for x in support if x['partition']=='fit' and x['visible_count']==128)
    fit256 = next(x for x in support if x['partition']=='fit' and x['visible_count']==256)
    assert (fit128['prefixes'],fit128['continue_prefixes'],len(fit128['continue_dates'])) == (153,35,11)
    assert (fit256['prefixes'],fit256['continue_prefixes'],len(fit256['continue_dates'])) == (42,4,3)
    text = '''# 下一步：单入口提前停止学习（方案，不是实验结果）

建议先只保留一次学习决策：原搜索完成128个候选后，满足原可停资格才调用模型。模型允许提前停止时，执行原同站补查并返回最佳已观察动作；模型要求继续时，原FRAME接管整个剩余搜索，不在192、256等位置再次调用学习停止器。无可接纳动作、原强制停止和候选耗尽仍按原规则处理。

## 为什么先做这个入口

先前模型把不同深度的可停前缀混合拟合，并在多个深度反复决策。新方案只问一个问题：在统一入口，现在结束能否保持原动作且少查？训练标签含义不变，仍与完整原FRAME加原补查比较，不改成一步奖励或较宽的质量容忍。

拟合数据在128处有153条状态、35条继续标签，覆盖11个继续日期；256处只有42条状态、4条继续标签，覆盖3个继续日期。选择128基于已有拟合支持和原前缀设计，不比较开发验证中哪个入口更赚钱。153条状态也不是153个独立电网场景。

## 单调性从结构上成立

对固定保存状态i，令s_i为只用前128反馈计算的固定停止分数，a_i^stop为此时停止并执行原补查后的动作，a_i^FRAME为原完整搜索动作，t为阈值。使用严格s_i>t才停止。没有可停资格的状态始终走原FRAME。

有资格时，一次完整搜索的动作不一致损失为：

L_i(t) = 1{s_i>t} × 1{a_i^stop ≠ a_i^FRAME}。

分数和两个动作在这个固定状态上不随t变化。因此t增大时L_i(t)只能不变或从1变0，不会从0变1。若把“提前结束但未严格减少查询”也记为不合格，只需将第二个指示项换成“动作不同或停止查询数不少于原查询数”，单调性同样成立。t=1时固定分数至多为1，全部交回原FRAME，动作一致性由构造保证。

这消除多学习出口引入的非单调误停；不是对模型准确率、计算优势或未来运行安全的证明。原损失标签与未来仅用于监督的结果不进入在线输入。

## 与风险校准的关系

该结构满足CRC基本定理所需的有界、右连续、随阈值非增的损失形式；仍须另行满足可交换抽样等条件。4个反复使用的校准日期不因此变成合格的新独立证据。校准只负责筛除不可靠停止，不替模型创造区分能力。

不能承诺它一定优于多出口：它放弃192等后续提前退出机会，可能节省更少。但任务更清楚、继续标签支持较多，能够分辨“入口判断学不好”与“重复退出结构/校准带来损失”。全部结果仍须与原FRAME、固定128/192/256及原学习模型在相同完整搜索口径比较。

## 下轮训练前必须固定

1. 只取原12拟合日期的128入口；原18项公开特征与缺失标记、标签、控制器、排序、奖励和补查保持。定义该入口的日期/调用权重，不把旧多前缀权重裁剪后冒称日期等权。
2. 仅一份简单分类器和一组事前参数，不同时扫描模型、类别系数、入口或阈值。拟合前通知并封存源码/数据/权重定义。
3. 原4校准日期用于经验阈值筛查和完整搜索重放，报告停止数、动作不合格和查询；不能写未来零风险保证。新模型/阈值在开发验证评分前封存。
4. 在原已暴露开发验证记录报告两类错误、完整搜索最终动作和查询，并保留全部简单基线。不把重复开发验证当最终测试，不以本轮结果放宽目标。
5. 只有搜索压缩确有收益，才考虑实际FRAME路径覆盖和真实控制时间。最终8测试仍封存，不在本轮采集新数据或扩大规模。

本文件只是下一步方案。没有新模型、分数、训练、阈值选择或单入口策略性能结果。
'''
    (O/'单入口学习方案.md').write_text(text,encoding='utf8')
    p=O/'GRID145_report.md'
    p.write_text(p.read_text(encoding='utf8')+'\n## 最终建议：先只学习128入口\n\n拟合支持核对后，下一方案进一步收窄为“128处只决定一次，继续则原FRAME接管剩余搜索”。128处35个继续样本覆盖11日期，256处4个只覆盖3日期。单入口的完整搜索误差等于停止指示与固定错误指示之积，随阈值非增，结构上消除第二学习出口带来的反转；不保证学得更好或节省更多，也不补足校准抽样条件。完整证明、代价和下轮冻结项见[单入口学习方案.md](单入口学习方案.md)。本轮未执行这个新策略或训练。\n',encoding='utf8')
    dec=G.read(O/'decision.json')
    dec['next']='Prospective single learned exit at128;continue delegates to the unchanged fullFRAME remainder. Fit support35continue/11dates versus4/3at256. Prove monotone fixed-state loss,then freeze one training protocol and notify before fitting. No new policy evaluated here.'
    G.save('decision.json',dec)
    c=R/'论文协作资料/FRAME_风险校准可行性_GRID145'
    for n in ['单入口学习方案.md','GRID145_report.md','decision.json']:shutil.copy2(O/n,c/n)
    addition='\n\nGRID145最终下一步建议：只在原128入口做一次学习停止，继续则交回原FRAME，无后续学习出口；单调完整搜索损失的指示乘积证明已给出。拟合支持128为153状态/35继续/11继续日期，256为42/4/3；不据开发验证选择入口。未执行或训练新策略，不承诺胜过CAP256；下轮冻结权重/模型/评价并训练前通知。\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+addition,encoding='utf8')
    p=REPO/'docs/NEXT_STEPS.md'
    s=p.read_text(encoding='utf8').replace('下一次训练先形成固定深度继续风险方案，依据risk_support_by_depth.json核对负例/日期支持；不预先选择好结果，不因样本少丢反例。','下一次训练先冻结128处单入口方案：继续则交回原FRAME，后续无学习退出；已有35继续/11日期支持，256仅4/3不先单训。入口取原设计及拟合支持，不看开发验证选入口；不因样本少丢反例。')
    p.write_text(s,encoding='utf8')
    G.verify()
    d=G.A/'single_entry_plan'
    d.mkdir()
    for n in ['单入口学习方案.md','GRID145_report.md','decision.json']:shutil.copy2(O/n,d/n)
    shutil.copy2(Path(__file__),d/'plan.py')
    (d/'archive_manifest.json').write_text(json.dumps({p.name:G.sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    count=0
    for phase in G.A.iterdir():
        for n,h in G.read(phase/'archive_manifest.json').items():assert G.sha(phase/n)==h;count+=1
    for p in c.iterdir():assert G.sha(p)==G.sha(O/p.name)
    G.save('final_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(list(c.iterdir())),
        no_new_fit_or_strategy_evaluation=True,original_sources_unchanged=True))
    d=G.A/'final_verification';d.mkdir()
    shutil.copy2(O/'final_verification.json',d/'final_verification.json')
    (d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':G.sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID145_SINGLE_ENTRY_PLAN_COMPLETE',count)

if __name__=='__main__':main()
