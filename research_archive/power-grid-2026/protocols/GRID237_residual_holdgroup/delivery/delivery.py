"""Verify the two-stage bounded probe and update the continuous handoff."""
from pathlib import Path
import importlib.util,json,shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R
def main():
    assert C.read(O/'complete.json')['passed'];C.scored();sp=importlib.util.spec_from_file_location('grid237_delivery236',R/'work/grid236_shared_prior_inputs/run.py');previous=importlib.util.module_from_spec(sp);sp.loader.exec_module(previous);manifests=[]
    for root in [previous.A,A]:
        for p in root.glob('*/archive_manifest.json'):
            for rel,h in C.read(p).items():assert C.sha(p.parent/rel)==h
            manifests.append(str(p))
    assert len(manifests)==9;copies=[]
    for folder,source,expected in [('FRAME_同源先验输入_GRID236',C.Q,6),('FRAME_修正层留组诊断_GRID237',O,8)]:
        files=list((R/'论文协作资料'/folder).iterdir());assert len(files)==expected
        for p in files:assert C.sha(p)==C.sha(source/p.name);copies.append(str(p))
    handoff='''# GRL持续交接：GRID236/237完成，下一步GRID238只读校准支持资格

先读integration/docs/RESEARCH_STATE.md和NEXT_STEPS.md，以最新状态为准。用户授权自主推进训练直到喊停，不逐轮确认；阶段实质结果简短报告，无变化安静。当前无持续运行器，不重复完成拟合/预测/审计/归档。

236完成同源公共先验再生成，复用24个已发布inner0森林头，22653公共行、0森林/NN新拟合/NNforward/物理；旧214 cal4公开预测与213矩阵hash均重现。fit8新筛选3747行：STOP2134/49正，辅助CONT1613/561正。全部公共预测先发布后仅加载inner1/2的fit8目标，未读取inner0cal4目标。fit8风险两域分别归一化结构质量×absNET后各半，原父222预算不继承。11模型有支持，f0_s1预定KEEP保持(新数据fit8虽1正，也未临时激活)。

237按原配置一次训练11个43参线性修正器(.0449秒)、1KEEP，0标量/NN/物理。offset同源b，原均值逻辑损失+全43单位L2含截距，原L-BFGS-B，全部success/rawgradinf<.001；只fit8。全部12权重先发布、3176个cal4原始未校准分数先发布，后才在本轮工作流解析inner0cal4标签。数据历史已暴露，不能称从未见过标签或盲测。

cal4 STOP1644/49回本，保留40/误停9，取消710无利/保留885无利；9误停涉及5调用8不同状态，包含跨池/前缀相关记录。种子误停6/1/2全部保留。辅助CONT1532/702正，误停140，单独看。相同未校准模型在fit8 STOP错停12/49，不把cal9vsfit12当泛化更好证据。也不把新cal9与旧两保护器871分歧0正例直接比较，两者人群/保护条件不同。9误停完整保存在cal4_STOP_positive_errors.json，全部3176分数/标签在cal4_error_rows.json。

这是固定上游outer12基模型、STOP样本总体条件下，修正层排除cal4直接拟合的诊断；上游标签祖先仍含cal4，不是整链独立OOF/全链嵌套/盲测，不能直接当可靠独立元选择器训练答案。本探针没有拟合alpha/预算/二级选择器，也没有新outer4策略分数/评估、整周仿真或时延。原230主候选、233Jan19/Jun1方向改善线索保留，不称竞争力提升。

下一步238仅只读：核查fit8/cal4的STOP正例约束分布、9误停的分组/先验/终点/祖先支持，明确若用4天做校准实际能估计什么。可以列单样本过零比例作描述，但本阶段不对真实数据求新极值步长/拟合alpha，不试多个组合再按结果择优，不给调用设例外。先资格固定唯一有限成对校准设计和必要性、总操作数、依赖祖先和对照，再另阶段执行。若后续要扩展新outer4开发策略评估，必须独立预注册并明确已经超出本237错误收集探针，不回改237；当前238不生成新outer4分数或策略评估。

参考源：237 run.py含固定模型/scoring/错误计数；236 fit_inputs/cal_public分角色；236future_cal_target_sources和235fixed_probe_pools给定分割。报告outputs/grid237_residual_holdgroup/GRID237_report.md，协作论文协作资料/FRAME_修正层留组诊断_GRID237；236报告/协作也已核验。9项既有归档清单、14协作副本已核对。归档必须读源码实际A常量，不猜别名。

新学习阶段最多22模型/标量拟合、一次600秒，不延长/盲扫，不为限制偷偷拆大搜索。费用/标签/全原日期种子/基础控制器/排序别名/锁站/128/64/停滞2/.9保持。最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。源码输入先发，所有模型标量先发，所有预测先发后连结果。Git helper必须poll到exit_code确认完成后才下一helper/发布屏障。旧235计数口径更正及233/225/226/229历史错误已封存，非当前故障。

按AGENTS更新docs状态/决定/日志/NEXT/首页、完整归档/D协作哈希、git_push.cmd推送并核对远端SHA。权限不足仅申请D工作区写和现有Git网络，拒绝不绕过。小而重复改善保留，不设全赢/5%新门槛，不将多查询质量改善冒充等预算优势；局部代理/错误计数不等整周成本/实际时延。用户喊停则停新增并暂停grl，不恢复旧自动化，不归档聊天。
'''
    (O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8');C.save('delivery_receipt.json',dict(passed=True,archive_manifests_verified=manifests,collaboration_copies=copies,total_probe_new_linear_models=11,total_probe_new_scalar_fits=0,no_repeated_fitting_scoring_or_fullarchive=True))
    d=A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'CONTINUOUS_HANDOFF.md',O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');print('GRID237_DELIVERY_VERIFIED',len(manifests),len(copies))
if __name__=='__main__':main()
