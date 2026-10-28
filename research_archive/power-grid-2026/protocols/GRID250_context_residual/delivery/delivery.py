from pathlib import Path
import json,hashlib,shutil,ast
R=Path('D:/Codex/GRL研究');REPO=R/'work/GRL-results-integration';O=R/'outputs/grid250_context_residual'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arch(work):
 t=ast.parse((R/work/'run.py').read_text(encoding='utf8'))
 return next(eval(compile(ast.Expression(n.value),'<A>','eval'),{'REPO':REPO}) for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='A' for x in n.targets))
plan='''# GRID251固定资格问题

只资格、0拟合/新策略评分/外层标签。250比247三种子查询全降、两个种子代理NET提高，35变化额外取消32无利、回本恢复1/损失2；均值NET略降，230仍主候选。保留250低查询探索端点及全部反例，不设全赢门。

原NET停止/继续位仅给出决策类别。下一步核查同一已计算动作分数差q1-q0能否作为连续公开置信信息，避免直接换风险/alpha。所有特征选择由这一公开语义固定，不按Jan19调用选阈值或筛样本。

固定唯一下一候选：250公共40输入后仅加1列nativeqgap，41输入42参数。归一化完全复用各池原208parent mean83/scale83：gap=((q1-q0)-(mean81-mean80))/sqrt(scale80^2+scale81^2)。组合参考尺度不是新拟合的gap标准差，不拟合温度、权重或扫尺度。原NET、两动作分数和这些scalers均已发布，ownfit8/cal12/outer4均使用同一原NET/父scaler来源；上游outer12、内部cal4不独立的限制保留。

251核查原192x83与190cachedheld的q位置80/81、原NETSTOP位关系、全导出有限及各池自身fit8标签/风险支持，冻结全部输入，不训练模型，不看新outer结果，不设置经验效果筛选门槛。若数据接口或公开可得性不通过则保留故障停止，不换输入救结果。

资格通过并发布后下一252：11个42参线性修正器+11原221保护步长、1KEEP、单次600秒，原219meanNLL+全参数单位L2、原L-BFGS-B/零42初始化，全部success且gradinf<.001后评分。行/风险/标签/b12/85保护/原父CONT入口不变。主匹配250，其余旧候选仅历史参照，明确多1维/参数的函数类变更，不称同容量。

源码输入先发、全部模型步长先发、全部预测先发后接旧开发结果。全种子日期反例保留，不改费用/阈值/风险比例，不给Jan19或Jan5例外，不自动扩数据/NN/物理/电网/FRAME/论文/最终8，也不安装依赖。小而重复改善保留，局部代理和query不是整周质量或控制时延。
'''
(O/'NEXT_PROTOCOL_PLAN.md').write_text(plan,encoding='utf8');handoff='''# GRID250后持续交接

先读docs状态/NEXT，以最新为准，已有运行器跟进，完整项不重做。248自身fit8只读完成，5702公共邻居先发后接2851自身标签；STOP41正例仅8调用，同域最近邻正例1/跨域13，辅助不能整体删，STOP风险冲突active11跨域41.60%/同域6.65%，不是因果证明。

249资格确认nativeSTOP公开可得，192q0/q1与2851训练域一致、85保护/4681外层入口全STOP，0新NET。250在247公共39后加一位上下文，40输入41参，11线性+11原步长(.065s)、1KEEP完成，85保护/4681分数/14268序列/35变化守恒核验。模型和所有预测先发后连结果。

250对247三种子query全降，NET差+.000062816/+.000022562/-.000124279；35变化多取消32无利、恢复1回本/损失2，回本30→29/55。query13.199858→13.165848、NET .092904056→.092891089。18变化种子日期17正1负不独立；种子2Jan19call005新失约.493NET、种子0Oct5新失约.055，原Jan5主要漏查未修复，反例全部保留。250保留低query研究候选/两种子小改善，不替换230主候选、不择种子。

下一251仅资格0拟合/新策略/外层标签，按NEXT_PROTOCOL_PLAN.md固定原NET连续qgap公开输入，原192raw83和190held q索引80/81，复用既有208parent mean/scale83组合尺度，不拟合温度或扫尺度。只自身fit8标签风险支持、接口/有限/公开性核验，全部输入封存先发；不按外层损失设阈值/例外。唯一后续252候选公共40+gap=41输入42参，11线性+11原步长/600秒、1KEEP，原行/标签/风险/b12/85保护不变。资格发布后另阶段执行，不预设有效/同容量，不同时改网络、风险比例、alpha或标签。

每学习阶段<=22模型/标量、单次600秒，不延长/盲扫/拆大搜索，不重做完整拟合/评分/审计/归档。代码输入先发、全部模型步长先发、全部预测先发后连结果。原费用/种子日期/控制器/排序别名/锁站/128/64/停滞2/.9保持；最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。全旧开发已暴露，基础NET与父模型outer12，内部cal4非全链独立，局部代理不等整周/真实时延，不将多查询质量提升冒充等预算，不设全赢5%门槛。工程错保留源/部分输出，不改协议窗口且不重复完整项才修复。Git严格等helper退出后下一helper/屏障，按AGENTS更新docs/归档首页，实际A归档，git_push.cmd推送核对报告/协作副本/清单/远端SHA。用户允许自主推进至喊停，更新grl起点，不恢复旧自动化、不归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
for work,out,dest in [('work/grid248_auxiliary_support','outputs/grid248_auxiliary_support','论文协作资料/FRAME_辅助域支持_GRID248'),('work/grid249_context_bit_plan','outputs/grid249_context_bit_plan','论文协作资料/FRAME_原决策上下文资格_GRID249'),('work/grid250_context_residual','outputs/grid250_context_residual','论文协作资料/FRAME_决策上下文学习_GRID250')]:
 oo=R/out;aa=arch(work);assert read(oo/'complete.json')['passed'] and read(oo/'independent_audit.json')['passed'];man=[];copies=[]
 for p in aa.glob('*/archive_manifest.json'):
  for name,h in read(p).items():assert sha(p.parent/name)==h,(p,name)
  man.append(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)))
 for p in (R/dest).iterdir():assert sha(p)==sha(oo/p.name);copies.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)))
 (oo/'delivery_check.json').write_text(json.dumps(dict(passed=True,archive_checks=man,collaboration=copies,no_scientific_reruns=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8');d=aa/'delivery';d.mkdir(exist_ok=False);shutil.copy2(oo/'delivery_check.json',d/'delivery_check.json')
 if oo==O:
  for name in ['CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.md']:shutil.copy2(oo/name,d/name)
  shutil.copy2(R/'work/grid250_context_residual/delivery.py',d/'delivery.py')
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
print('GRID248_249_250_DELIVERY_VERIFIED')
