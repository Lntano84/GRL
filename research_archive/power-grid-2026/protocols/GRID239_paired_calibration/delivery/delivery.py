"""Verify the preregistration and pair result, then hand off one read-only stage."""
from pathlib import Path
import importlib.util,json,shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R
def main():
    assert C.read(O/'complete.json')['passed'];C.scored();sp=importlib.util.spec_from_file_location('grid239_delivery238',R/'work/grid238_calibration_support/run.py');prev=importlib.util.module_from_spec(sp);sp.loader.exec_module(prev);manifests=[]
    for root in [prev.A,A]:
        paths=[root/'archive_manifest.json'] if (root/'archive_manifest.json').exists() else list(root.glob('*/archive_manifest.json'))
        for p in paths:
            for rel,h in C.read(p).items():assert C.sha(p.parent/rel)==h
            manifests.append(str(p))
    assert len(manifests)==6;copies=[]
    for name,source,expected in [('FRAME_成对校准资格_GRID238',C.PLAN,6),('FRAME_成对校准对照_GRID239',O,8)]:
        files=list((R/'论文协作资料'/name).iterdir());assert len(files)==expected
        for p in files:assert C.sha(p)==C.sha(source/p.name);copies.append(str(p))
    handoff='''# GRL持续交接：GRID238/239完成，下一步GRID240只读共享错误和省查询区域

先读integration/docs/RESEARCH_STATE.md和NEXT_STEPS.md，以最新状态为准。用户允许自主推进训练直到喊停，不逐轮确认；实质阶段简短报告，无变化安静。当前无持续运行器，完成拟合/评分/审计/归档不重复。

238只读98正例资格完成，没求新步长/模型或生成外层分数；11学习池FIT8和CAL4均有原STOP正例，cal每池1—7、两池1，不能当稳定保证。独立预注册239新阶段，明确扩展到旧已暴露outer4离线开发，没回改237错误收集探针。

239复用237全部43参端点、同一inner8头和scalers，只改221最大正例保护的正例来源：FIT8 vsCAL4。22标量(.100s)/0新模型/2KEEP，24有效文件先发、24外层预测再发、后接结果；9362分数/21402保存轨迹/源正例约束/最大性/208控制均核对。原208CONT入口不变；b8<=0统一保持原父决定，131条支持域回退含3回本，两组相同，不能算新校准器学出的保留。没有新NN/森林拟合/物理。额外24头推理没做在线计时，不谈完整控制时间。

主对照结果：FIT8保留47/55回本，CAL4保留48/55；NET .094323443→.094299228，候选查询13.264926→13.283551。两臂32分歧，只有1回本恢复(Mar16种子2)，其余31不回本。CAL减FIT的种子NET为-.000055761、0、-.000016884，不支持CAL平均优越。保留全部日期种子，未试第三臂或挑参数。

值得研究的区域：相对共同GATE入口上下文，CAL4种子0/1即时增益和动作一致率完全相同，平均候选查询分别-.199885/-.077783，NET+.000288259/+.000112172。种子2却有共同大损失，不挑两个成功种子作为最终方法。两臂共同7回本误停全部种子2，4调用：Jan5_101call001三相关状态、Feb16_100call006两相关状态、Jun1_101call001一、Nov16_101call001一。FIT还额外漏Mar16一条，CAL恢复。原230仍主要内部候选；239与230/233训练日期/先验/约束不同，不能跨版本当纯校准消融。

下一步240只读：查看shared_profitable_misses.json、changed_choices/effect_decomposition及public_outer；分解共同种子2错误的b8/原208b、端点、两校准步长、训练/校准正例覆盖与支持域，保留种子0/1省查询区域和131回退来源。个体样本和同调用前缀相关，勿把7当独立失败。先查共同瓶颈是否来自端点或覆盖，而不是继续换alpha来源。0新第三臂、阈值、步长或模型，不试多组合择优，不按外层给Jan5例外/择种子。用自身训练标签和既有公开支持固定下一有限方案后再资格执行。

239报告outputs/grid239_paired_calibration/GRID239_report.md；协作论文协作资料/FRAME_成对校准对照_GRID239。238/239的6归档清单、14协作文件已核对。归档从源码实际A取，不猜别名。全部开发数据已暴露，239是离线代理比较，不是新盲测；上游outer12含内部cal4，条件层诊断不是整链独立OOF。原outer4标签祖先隔离仍保持。候选查询/即时增益/代理NET不等完整周质量/实际控制时延。

各后续学习阶段最多22模型/标量拟合、一次600秒，不延长/盲扫，不拆大搜索。费用/标签/全原日期种子/控制器/排序别名/锁站/128/64/停滞2/.9不变；最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。源码先发、模型标量先发、预测先发后连结果。Git helper必须poll到exit_code后才下一helper/发布屏障。旧235计数更正和233/225/226/229历史错误保留，非当前故障。

按AGENTS更新docs状态/决定/日志/NEXT/首页、归档/D协作哈希及git_push.cmd推送远端核对。权限不足仅申请D工作区写和现有Git网络，拒绝不绕过。小而重复改善保留，不设全赢/5%新门槛，不将多查询质量提升冒充等预算。用户喊停则停新增并暂停grl，不恢复旧自动化、不归档聊天。
'''
    (O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8');C.save('delivery_receipt.json',dict(passed=True,archive_manifests_verified=manifests,collaboration_copies=copies,total239_scalar_fits=22,total239_new_models=0,no_repeated_complete_work=True))
    d=A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'CONTINUOUS_HANDOFF.md',O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8');print('GRID239_DELIVERY_VERIFIED',len(manifests),len(copies))
if __name__=='__main__':main()
