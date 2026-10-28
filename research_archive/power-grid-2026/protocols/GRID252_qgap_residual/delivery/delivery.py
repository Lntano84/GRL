from pathlib import Path
import ast,json,hashlib,shutil
R=Path('D:/Codex/GRL研究');REPO=R/'work/GRL-results-integration';O=R/'outputs/grid252_qgap_residual'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arch(work):
 t=ast.parse((R/work/'run.py').read_text(encoding='utf8'))
 return next(eval(compile(ast.Expression(n.value),'<A>','eval'),{'REPO':REPO}) for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='A' for x in n.targets))
plan='''# GRID253：停止样本来源的只读资格问题

不要继续自动加第三个弱特征或单独换alpha。252已完成全部22拟合、177配对与守恒；原始query净-479，但6条小额恢复原NET合计.0795、5条损失2.0191，含同调用跨种子重复。保留230主候选、250低query/两种子小改善、252三调用恢复和省查询线索，不拼策略或择种子。

253仅只读，先复用GRID191 summary/报告和GRID235修正后祖先图结论，不把旧选择偏差诊断重跑当新发现。核对当前训练STOP源由原NET在见过日期上选择，当前头/校准内部祖先；在既有已发布结果上描述当前支持，但不新评分/拟合/调阈值。

盘点真正的baseNET(FACTOR)权重及fit/held日期，不把GRID214八天收益头当作八天baseNET。每条拟合记录参考NN若要排除该记录日期又保持当前outer4不参与标签学习，需要明确祖先集合；原4个12天基模型不能直接混用，别的外层模型可能含当前outer标签。若需两外层组同时排除，列所有日期集合、可复用权重、缺失模型和对应全3种子数量，不能事后选一个好fold或seed。

固定唯一有限数据/支持方案之前，完整算清必要NN拟合、收益头/修正器/保护标量、新forward与窗口，不隐去新增操作；记录后续是否仅修正STOP选择源，哪些父NN/收益先验仍带自身标签祖先，不能冒充全链独立OOF。

原7250updates的FACTOR历史单模型约70秒是已测参考，不保证本轮时间。若新数据基础确需18个NN，不能假设18个串行能塞进600秒；先明确可复用性、并发/资源与严格全局deadline，所有科学步骤计数和依赖公开。每学习阶段最多22模型/标量、单次600秒，不延长、不悄悄拆大搜索；若完整方案需超出当前授权则列具体决定，先不训练。不因预算不足换日期/模型目标/更新次数救结果。

仅用原教师记录/公开输入和自身训练标签；不生成或读最终8，不新物理/电网/论文/FRAME，不安装依赖。部署原控制器/排序/别名/锁站/128/64/停滞2/.9保持。原费用/标签/全种子日期保留；全部代码/输入、权重、预测按屏障发布，所有不利结果保留。
'''
(O/'NEXT_PROTOCOL_PLAN.md').write_text(plan,encoding='utf8');handoff='''# GRID252后持续交接

先读docs当前状态/NEXT，以最新为准，检查Python父子进程和Git，有运行器跟进，不重复完整项。251连续qgap公开资格/252固定支线训练已完成。qgap是原NET动作偏好，不是校准概率；用192/190缓存q列80/81及既有208parent83参考尺度，无新NETforward。252为41输入42参、11线性+11原步长(.052s)、1KEEP，85保护/4681分数/14268序列/177变化守恒核验；模型预测先发后接结果。

对250，252query13.165848→13.125506，NET .092891089→.092752918；回本29→30/55，恢复6损失5。种子0相对250 NET+.000094556/query-.089138，种子1-.000416026/query-.033470，种子2-.000093042/query+.001584。新加133无利合计293query中位1，取消33无利732query中位16，全保存记录query净-479；追加303→404不能当query增加。恢复6条在种子2三个调用(4/6三、2/16二、11/16一)，原始NET.0795；损失5原始NET2.0191(种子1Jan19三/种子2Jan19一/种子0Oct5一)，相关记录不独立。原Jan5主要漏查仍在。保留230主候选、250低query/两种子小改善、252恢复和省查询线索，252不能按均值替换250，不拼策略择种子。

下一253仅只读源选择/祖先与有限方案可行性，按NEXT_PROTOCOL_PLAN，先复用191/235旧结论，不重复命名旧诊断为新实验。盘点真正未见日期的baseNET(FACTOR)权重、fit集合与可复用预测；214inner8只是收益头，不能当STOP/NET隔离。另一outer模型可能含当前outer标签，不直接混用。列保持原4折全3种子/outer标签隔离的唯一数据支持方案、完整NN/收益头/修正器/标量/forward总操作和窗口；原7250updates单NN历史约70s，不假定18串行可在600s。只资格不训练，不能隐去操作或悄悄拆大搜索；确需超授权时列具体决定，不改更新次数/种子救预算。明确只是STOP选择源修正还是全链嵌套，父先验仍含自身标签时必须披露。不要继续自动加特征/alpha/风险比例，不给Jan19/Jan5例外。

每学习阶段<=22模型/标量、单次600秒不延长，不盲扫/重做完整拟合评分审计归档。原费用标签/全种子日期/控制器/排序别名/锁站/128/64/停滞2/.9保持；最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。开发已暴露/相关前缀不独立，局部代理不等整周/实际时延，不将多查询质量改善冒充等预算，不设全赢5%门槛。代码输入先发、全部权重步长先发、全部预测先发后连结果；工程错保留源/部分输出，仅不改协议窗口且不重复完整项才修复。Git严格等helper退出再下一helper/屏障，按AGENTS更新docs/归档首页、实际A归档、git_push.cmd推送，核对真实报告/D协作副本/清单/本地远端SHA。用户授权自主推进至喊停，更新grl起点，不恢复旧自动化、不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
for work,out,dest in [('work/grid251_qgap_plan','outputs/grid251_qgap_plan','论文协作资料/FRAME_连续动作偏好资格_GRID251'),('work/grid252_qgap_residual','outputs/grid252_qgap_residual','论文协作资料/FRAME_连续动作偏好学习_GRID252')]:
 oo=R/out;aa=arch(work);assert read(oo/'complete.json')['passed'] and read(oo/'independent_audit.json')['passed'];man=[];copies=[]
 for p in aa.glob('*/archive_manifest.json'):
  for name,h in read(p).items():assert sha(p.parent/name)==h,(p,name)
  man.append(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)))
 for p in (R/dest).iterdir():assert sha(p)==sha(oo/p.name);copies.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)))
 (oo/'delivery_check.json').write_text(json.dumps(dict(passed=True,archives=man,collaboration=copies,no_repeated_science=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8');d=aa/'delivery';d.mkdir(exist_ok=False);shutil.copy2(oo/'delivery_check.json',d/'delivery_check.json')
 if oo==O:
  for name in ['NEXT_PROTOCOL_PLAN.md','CONTINUOUS_HANDOFF.md']:shutil.copy2(oo/name,d/name)
  shutil.copy2(R/'work/grid252_qgap_residual/delivery.py',d/'delivery.py')
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
print('GRID251_252_DELIVERY_VERIFIED')
