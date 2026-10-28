from pathlib import Path
import json,hashlib,shutil,ast
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid243_parent_coherent'; REPO=R/'work/GRL-results-integration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def archive_const(work):
 t=ast.parse((R/work/'run.py').read_text(encoding='utf8'))
 # Match actual source constant without importing experiment runners.
 for n in ast.walk(t):
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='A' for x in n.targets):
   return eval(compile(ast.Expression(n.value),'<archive constant>','eval'),{'REPO':REPO})
 raise AssertionError(work)
checks=[]; copies=[]
for work,out,dest in [('work/grid242_parent_prior_inputs','outputs/grid242_parent_prior_inputs','论文协作资料/FRAME_父先验一致输入_GRID242'),('work/grid243_parent_coherent','outputs/grid243_parent_coherent','论文协作资料/FRAME_父先验一致对照_GRID243')]:
 a=archive_const(work)
 for p in sorted(a.glob('*/archive_manifest.json')):
  m=read(p)
  for f,h in m.items():assert sha(p.parent/f)==h,(p,f)
  checks.append({'manifest':p.relative_to(REPO).as_posix(),'files':len(m),'sha256':sha(p)})
 for p in sorted((R/dest).iterdir()):
  assert p.is_file() and sha(p)==sha(R/out/p.name)
  copies.append({'path':p.relative_to(R).as_posix(),'sha256':sha(p)})
 assert read(R/out/'complete.json')['passed']
assert len(checks)==9 and len(copies)==14
for name in ['source_freeze.json','trained_freeze.json','scored_freeze.json']:
 for p,h in read(O/name).items():assert sha((R if name=='source_freeze.json' else O)/p)==h
handoff='''# GRID243后持续交接

242输入支持和243固定11线性+11步长已完成，所有权重、4681预测先发后连接标签，14268保存序列和85保护正例核验通过，不能重跑拟合/评分/完整归档。243回本保留33/55，取消243不回本追加，查询13.440525降至13.203561，代理净效用.096092524降至.092921661。种子1小折中保留，种子0/2损失全保留，不替换230。

下一步GRID244仅只读：同ID对齐242父12天b12与236固定inner8先验b8，在fit8/cal4、原STOP/CONT、NET正负分别比较分布/符号/资格。核对243的22漏查与85保护正例、239共同7漏查。先查父12天已参与标签学习的内部读出和外层未参与读出的支持差别；比较还含训练样本量差别，不称单一因果。0拟合/标量/新模型forward/物理，不尝试新阈值/alpha/第三策略/调用例外。

仅根据自身训练支持固定下一有限方案。各学习阶段至多22模型/标量和一次600秒，不延长、不扫参；代码输入先发，模型步长先发，全部预测先发后接结果。原费用标签、种子日期、控制器/排序/别名/锁站/128/64/停滞2/.9不变。最终8、物理、电网、论文、FRAME不动，不安装依赖。数据已暴露，只称开发比较，局部代理不等整周质量/实际时延，不将多查询质量提升冒充等预算优势。Git严格串行等helper退出再下一helper/屏障。用户授权自主推进至喊停，更新状态文档、交接及grl跟进起点，不恢复旧自动化。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
rec=dict(passed=True,archive_manifests=checks,collaboration_files=copies,no_repeated_fit_or_archive=True)
(O/'delivery_check.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=archive_const('work/grid243_parent_coherent');d=a/'delivery';d.mkdir(exist_ok=False)
for p in [O/'CONTINUOUS_HANDOFF.md',O/'delivery_check.json',R/'work/grid243_parent_coherent/delivery.py']:shutil.copy2(p,d/p.name)
(d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
print('DELIVERY_VERIFIED',len(checks),len(copies))
