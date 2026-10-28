from pathlib import Path
import json,shutil,hashlib,ast
R=Path('D:/Codex/GRL研究');REPO=R/'work/GRL-results-integration';O=R/'outputs/grid247_canonical_residual'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def archive_const(work):
 t=ast.parse((R/work/'run.py').read_text(encoding='utf8'))
 return next(eval(compile(ast.Expression(n.value),'<A>','eval'),{'REPO':REPO}) for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='A' for x in n.targets))
plan='''# GRID248：下一步固定只读问题

247已完成全部13变化方向/步长代数分解，不再重算新策略或调alpha。三个新增回本误停，方向和步长两部分均更负，不能套用旧“方向已改善只是alpha放大”的结论。主要Jan5损失未解决，去重复对照不支持重复列是主因。

下一步248转向辅助域对实际停止域的训练支持：246原目标中STOP回本风险份额11池均值4.54%，辅助CONT为98.26%，两域风险各半。这是原目标定义，有学习取消倾向的合理依据，不直接改比例或标签。先只在各池自身fit8核对公共39向量下的标签/风险和跨域支持，0新模型/标量/外层预测/策略。

固定只读比较全部自身fit8 STOP和CONT，使用247同一39维标准化公共输入的欧氏距离，不拟合度量、不扫k。对每条训练记录，以公开输入和训练内物理调用组排除规则，找跨域最近1条及同域最近1条，排除自身与同一physical_state_group，避免同调用相关前缀自证。先封存所有邻居配对和距离，再接各池原标签/风险；报告全部种子/池的标签分歧、距离比与原风险权重。若某域无不同调用候选，显式记无支持，不换规则补样本。此为训练数据几何支持诊断，不是新监督模型或独立效能证据。

只在自己的训练标签与公开支持基础上固定下一唯一有限学习方案；不按已暴露外层误停挑特征/风险比例/阈值/alpha，不给Jan5设例外，不试多个方案选外层最好。不要再做一轮纯alpha更换或盲加网络。
'''
(O/'NEXT_PROTOCOL_PLAN.md').write_text(plan,encoding='utf8');handoff='''# GRID247后持续交接

先读docs RESEARCH_STATE/NEXT_STEPS，以最新状态为准，已有运行器跟进，完整项不重做。246资格完成：原MaxRhoReward=2-rho，三对奖励/负载表达fit误差<=1.2e-7；原STOP风险正例份额11池均值4.54%，辅助CONT98.26%，两域风险各半。247唯一候选去奖励列14/18/19，其余列、标签、权重、父b12、85保护集合不变，38公共+b12=39输入40参数，单位L2功能方向惩罚变化已披露。

247已完成11线性+11原规则步长(.055s)、1KEEP，全部success/grad资格、85保护、4681分数和14268序列核验，模型先发、预测先发后连结果。主匹配243：回本33→30/55，不回本取消243→249；NET .092921661→.092904056、查询13.203561→13.199858。仅13决定变化，种子NET变化-.000000842/-.000060863/+.000008893；种子2小折中、原种子1相对GATE小改善及所有反例保留，不替换230/243。原Jan5主要损失未修复，不把重复特征当主要原因。

13变化的方向/步长代数分解已保存，不重复做新策略。新增3回本误停的方向和步长均更负，不能泛套旧的方向改善/alpha放大解释。下一248只读，按NEXT_PROTOCOL_PLAN.md预定全自身fit8公共39几何支持：全部STOP/CONT、固定跨域/同域最近1，排除自身/同physicalcall，先封存公共配对/距离后接自身原标签/风险。无候选记缺失，不改规则；0度量拟合/新模型/标量/外层预测/第三策略。查辅助CONT如何支持真实STOP，不能直接按外层结果改半半风险或标签；不要盲换alpha/增网络。仅据自身训练和公开支持资格固定唯一下一有限方案后执行。

每学习阶段最多22模型/标量、单次600秒不延长/盲扫/悄悄拆搜索，原费用标签/种子日期/控制器/排序别名/锁站/128/64/停滞2/.9保持。最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。已暴露开发/相关前缀不称盲测，内部12天保护不独立，局部代理不等整周/真实时延，不将多查询质量改善冒充等预算。代码输入、模型步长、全部预测分阶段先发。工程错误保留源/部分输出，不改协议窗口且不重复完整项才修复；247准备旧路径错误0拟合/评分已留原源并纠正，不是当前故障。Git严格等helper退出再下一helper/屏障，按AGENTS更新docs/归档首页，实际源码A归档，用git_push.cmd推送并核对协作副本/清单/远端SHA。用户授权自主推进至喊停，同步grl跟进起点，不恢复旧自动化或归档聊天。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID247后\n\n1. 246/247资格训练评分审计完成，不重做。247不替换230/243，保留种子2相对243小折中及既有有效取消区域。\n2. GRID248仅训练内跨域支持只读核查，按247 NEXT_PROTOCOL_PLAN：全部自身fit8的39公共向量，固定最近1跨域/同域，排除自身及同physicalcall，先公共配对封存再接原标签风险；缺支持显式保留。0新模型/度量拟合/alpha/外层策略。\n3. 247的13变化代数分解已完成，新增3回本误停方向/步长均更负，不重做。STOP回本风险平均4.54%、辅助CONT98.26%，核查辅助迁移支持，不直接按外层改风险比例、扫特征/阈值或设Jan5例外；证据后固定唯一有限方案。\n4. 每学习阶段<=22模型/标量，单次600秒不延长，不扫参/重做。代码输入、模型步长、全部预测分阶段先发；原费用标签/种子日期/控制器/排序别名/锁站/128/64/停滞2/.9不变；最终8/物理/电网/论文/FRAME不动，不安装依赖。Git严格串行等helper退出。\n',encoding='utf8')
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID247后固定的下一问题\n\n13变化已分解，新增3回本误停方向与步长均更负，不重做alpha策略。下一248按封存计划只读自身fit8 STOP/CONT跨域几何支持，固定最近1/排除同调用、先公共配对后接自身标签。STOP与辅助CONT风险正例份额4.54%/98.26%的差别先解释，不直接换风险比例或增加网络。\n',encoding='utf8',newline='\n')
for work,out,dest in [('work/grid246_canonical_plan','outputs/grid246_canonical_plan','论文协作资料/FRAME_公共表示资格_GRID246'),('work/grid247_canonical_residual','outputs/grid247_canonical_residual','论文协作资料/FRAME_去重复表示训练_GRID247')]:
 oo=R/out;aa=archive_const(work);assert read(oo/'complete.json')['passed'];man=[];copies=[]
 for p in aa.glob('*/archive_manifest.json'):
  for f,h in read(p).items():assert sha(p.parent/f)==h,(p,f)
  man.append(dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)))
 for p in (R/dest).iterdir():assert sha(p)==sha(oo/p.name);copies.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)))
 receipt=dict(passed=True,archive_checks=man,collaboration_checks=copies,no_repeated_fit_or_archive=True)
 (oo/'delivery_check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8');d=aa/'delivery';d.mkdir(exist_ok=False);shutil.copy2(oo/'delivery_check.json',d/'delivery_check.json')
 if oo==O:
  for n in ['CONTINUOUS_HANDOFF.md','NEXT_PROTOCOL_PLAN.md']:shutil.copy2(oo/n,d/n)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
print('GRID246_247_DELIVERY_VERIFIED')
