"""Hash-only delivery receipt and next finite research handoff."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,R=C.O,C.A,C.R
def main():
    assert C.read(O/'complete.json')['passed'];C.scored();manifests=[]
    for p in A.glob('*/archive_manifest.json'):
        for rel,h in C.read(p).items():assert C.sha(p.parent/rel)==h
        manifests.append(str(p))
    assert len(manifests)==5
    dest=R/'论文协作资料/FRAME_公共修正器对照_GRID233';copies=[]
    for p in dest.iterdir():assert C.sha(p)==C.sha(O/p.name)==C.sha(A/'audited'/p.name);copies.append(p.name)
    assert len(copies)==8
    handoff='''# GRL持续交接：GRID233完成，下一步GRID234只读

先读integration/docs/RESEARCH_STATE.md、NEXT_STEPS.md，以最新状态为准。用户允许自主推进训练直至喊停，不必逐轮确认；阶段实质结果简短报告，没变化安静。当前无持续训练运行器，完成拟合/评分/审计/归档不重复。

233已按232固定方案训练11个43参数线性模型+11个联合标量=22，另f0_s1KEEP，.05785秒；原L-BFGS-B配置，全部梯度<1e-3(约1e-6)，数值资格通过。仅额外修正器输入原列83:125(公共41+b)，强208先验仍保留原信息。原224revision02半半风险、均值NLL+全43单位L2含截距，联合原90STOP正例保护与自身222训练取消加权查询代价上界。没有改费用/标签/模型规模/阈值/种子。所有模型和步长先发、4681预测先发、后接28536四臂保存序列并核验。全12模型、90正例/查询约束、预测/成本/控制/发布屏障通过。

结果：相对230恢复8条回本、误停6条，保留39回41/55；种子2恢复Jan19_101call005三个相关前缀、Jun1_101call001等机会，净效用+.000505086但查询+.128303。种子0净效用-.007654570、查询+.033325，主要Jan5_101call001四相关状态同批收益被误停，不当四独立失败；种子1净效用-.000039305、查询+.027189。总体NET.095926664→.093530401，查询13.171604→13.234543、即时增益下降；两个此前跨种子重复少3/4查询状态在233全部恢复无利补查。保留230主要内部候选，233保留局部恢复线索，不因均值差而丢掉，也不挑种子2替换。训练预算不保证外层相同查询；追加次数更少但每批候选不同，总查询反增。

下一步234仅只读分解8恢复/6误停、Jan5四相关状态、Jan19/Jun1恢复、新丢失重复节省：分别核对原强先验b、学习终点方向、联合步长限制与自身训练支持。可比较同一公开状态在125维与42维修正器的分歧，但不试多个新策略再按外层择优，不按外层选阈值/alpha/正则/费用/种子，不给调用设例外。用自身外层训练标签提出下一有限方案前不要新增拟合；各学习阶段<=22拟合/校准、600秒，不自动扩网络/日期。

工程记录：模型上传helper还没退出时过早检查远端被发布屏障阻止；score因缺receipt拒绝，0外层评分/0重训。并行误启动的第二helper因共享临时输出失败。已等待原helper确认6db443ca，保存interrupted_attempts/publication_wait_01/error.json，再首次评分并按78e655da封存预测。此后所有Git mutation必须严格串行等待helper完成，返回session_id要poll到exit_code后才调用发布屏障/另一个helper。不得把这当模型失败或当前故障。

旧225/226数值零评分失败、229合成舍入见证错误均已封存，230/231/232/233已完成，不重做。原控制器、费用标签、全种子日期、排序别名、初始锁站、128/64/停滞2/.9不变。最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。所有数据已暴露，只称开发比较，局部代理效用不等整周成本或实际时延。小而重复改善保留，不设全赢/5%新门槛，不把多查询质量提升冒充等预算优势。

报告outputs/grid233_public_correction/GRID233_report.md，协作论文协作资料/FRAME_公共修正器对照_GRID233，归档按源码实际A常量。更新docs状态/决定/日志/NEXT/首页、完整归档/协作哈希核对并git_push.cmd推送。权限不足只申请D工作区写和现有Git网络，拒绝不绕过。用户喊停则停新增并暂停grl，不恢复旧自动化，不归档聊天。
'''
    (O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8');C.save('delivery_receipt.json',dict(passed=True,five_archive_manifests_verified=manifests,eight_collaboration_copies=copies,source_and_model_score_hashes_verified=True,no_repeated_fits_or_scoring=True))
    d=A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),O/'CONTINUOUS_HANDOFF.md',O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    shutil.copytree(O/'interrupted_attempts',d/'interrupted_attempts');(d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():C.sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8');print('GRID233_DELIVERY_VERIFIED',len(copies))
if __name__=='__main__':main()
