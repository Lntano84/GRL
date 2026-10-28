"""Hash-only final delivery and continuous research handoff."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R
def main():
    assert C.read(O/'complete.json')['passed'];C.scored();manifests=[]
    for root in [C.PLAN.A,A]:
        files=[root/'archive_manifest.json'] if (root/'archive_manifest.json').exists() else list(root.glob('*/archive_manifest.json'))
        for p in files:
            for rel,h in C.read(p).items():assert C.sha(p.parent/rel)==h
            manifests.append(str(p))
    assert len(manifests)==6
    dest=R/'论文协作资料/FRAME_补查强度对照_GRID230';copies=[]
    for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name)==C.sha(A/'audited'/p.name);copies.append(p.name)
    assert len(copies)==8
    handoff='''# GRL持续交接：GRID230完成，下一步GRID231只读

先读integration/docs/RESEARCH_STATE.md和NEXT_STEPS.md，以最新状态为准。用户允许自主训练推进直至喊停；每个可核验阶段简短汇报，没变化安静结束。无持续训练运行器，不重复完成拟合/评分/审计/归档。

225/226为历史数值未合格零评分；227完成固定混合经验对照；228完成610记录47变化诊断。均不要重做。228发现大损失及重复小节省同受保护步长放大作用，数值误差诊断未支持主要归因求解未收敛。

229仅训练内资格：11学习池10个alpha变大，但同一原STOP训练加权查询取消量只3池增加，不能统一压回旧alpha。原第一次合成资格用一ULP见证最大性会被舍入成零；原源码/设计/sourcefreeze和错误保存在229/interrupted_attempts及归档interrupted_attempt01，0真实拟合/外层读取。revision02只改合成见证到(bound,cap]中点，实际规则不变，通过101检查。只消费229revision02。

230固定训练内查询取消代价约束：预算=自身222在原STOP训练记录上的sum(结构权重*已知下一批候选数*取消指示)。在227原正例保护alpha以内，选最大满足训练取消代价<=预算的alpha；tiesKEEP，ULP只保数值可行。0端点/NN重训，11标量+1KEEP，.1秒内完成600秒窗口；权重全部不变，仅f0_s0/f1_s1/f3_s0改步长。全部模型/标量先发布，4681预测先发布，连接28536四臂保存序列后核验。90训练正例仍保留。它增加经验预算约束，不保证外层等查询预算。

230恢复Jan5_101call001种子0的大收益机会，保留两个状态在种子1/2重复少3/4无效查询。回本38回39/55。相对227新12处补查=1回本+11不回本，相关记录原始增加11有效候选查询和216无利查询；不能说等预算质量提升。净效用.095320317→.095926664、查询13.148416→13.171604；对旧222净效用.095926811基本持平略低、查询13.174668略少。对222三种子净效用-.000033714,+.000013724,+.000019551，所有种子保留，不宣称总体胜出。保留208主参考、222折中、230受控内部候选。

下一步231只读残余回本误停、Nov16恢复和重复节省，以及12处恢复中的11条无利补查。先确定哪些是修正强度、哪些是学习方向/拟合支持问题；不事后算一堆策略择优，不按已看外层选阈值、alpha、费用、惩罚或种子，不给Jan5设例外。42输入43参数的公共摘要+b修正器仍只是候选，尚未训练；若做新学习方案，必须依据自身外层训练支持先资格固定一次有限方案。

每个学习阶段最多22拟合/校准、单次600秒，不延长/盲扫/重做完整项，改变目标或经验约束必须披露。原费用标签、全种子日期、控制器、排序别名、锁站、128/64/停滞2/.9保留；旧控制复用。源码先发、模型先发、预测先发再连标签。所有开发数据已暴露，不称盲测；局部代理指标不等整周成本或实际控制时延。最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。

报告outputs/grid230_training_budget/GRID230_report.md，协作目录论文协作资料/FRAME_补查强度对照_GRID230。A从实际源码读取，不猜别名。状态/决定/日志/NEXT/归档首页已更新；归档、协作哈希和远端SHA核实后汇报。权限不足仅申请D工作区写与现有Git网络，拒绝不绕过。用户喊停则停新增并暂停grl，不恢复旧自动化、不归档聊天。
'''
    (O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
    C.save('delivery_receipt.json',dict(passed=True,archive_manifests_verified=manifests,eight_collaboration_copies=copies,all_source_hashes_unchanged=True,no_repeated_fits_or_scoring=True))
    d=A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'delivery_receipt.json',O/'CONTINUOUS_HANDOFF.md']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');print('GRID230_DELIVERY_VERIFIED',len(copies))
if __name__=='__main__':main()
