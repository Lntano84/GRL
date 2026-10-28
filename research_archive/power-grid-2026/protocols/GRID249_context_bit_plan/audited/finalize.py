from pathlib import Path
import sys,json,shutil
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO;C.verify();q=C.read(O/'qualification.json');assert q['passed'] and C.read(O/'complete.json')['actual_fits']==0
for r in q['pools']:
 pool=r['pool']
 for folder in ['fit_inputs','calibration_inputs']:
  old=np.load(C.Q/folder/f'{pool}.npz');new=np.load(O/folder/f'{pool}.npz');assert np.array_equal(new['X'][:,:39],old['X'])
  for k in old.files:
   if k!='X':assert np.array_equal(old[k],new[k])
  assert np.array_equal(new['X'][:,39],old['domain_STOP'].astype(float)) if folder=='fit_inputs' else np.all(new['X'][:,39]==1)
 p=np.load(C.PUBLIC/'outer_parent_public'/f'{pool}.npz');n=np.load(O/'outer_parent_public'/f'{pool}.npz');assert np.array_equal(p['X'][:,C.KEEP],n['X'][:,:39]) and (n['X'][:,39]==1).all() and np.array_equal(p['ids'],n['ids']) and np.array_equal(p['b'],n['b']) and np.array_equal(p['base_CONT'],n['base_CONT'])
C.save('independent_audit.json',dict(passed=True,all36input_exports_checked=True,native_context_scores_verified_noNETforward=True,allcanonical39_andlabels_weights_parentprior_85constraints_unchanged=True,all41gradientchecks_pass=True,newfits=0,newheldscores=0,heldoutcomes=0));lines=['# GRID249：原模型决策上下文的可得性资格','','下一轮只增加一个已知上下文：原NET在当前前缀打算继续还是停止。它由原模型已经算出的两个动作分数比较得到，不依赖买下后续候选的反馈，也不读取未来收益标签。','','已用192保存的原分数核对全部训练域，和2851条自身fit8输入的STOP/CONT标记完全一致；自身12天保护输入均为STOP。又用190旧外层缓存核对4681个部署入口均为STOP，额外位恒为1。没有运行新NET或读取外层结果。','','下一轮250保留247的39维公共输入，追加未标准化的0/1上下文位，得到40输入、41参数。共享原特征系数，允许一个受单位L2约束的域偏置；所有参数包括截距均受惩罚。这增加一个参数和条件化能力，明确不是严格同容量。','','标签、各域一半风险、训练行、父b12、标准化和85条保护正例逐项不变。11个线性模型加11个原221保护步长，单次600秒、1KEEP，原L-BFGS-B设置不变，全部success且40?原始41维梯度小于.001后才评分；不加迭代或重试救结果。','', '主要匹配247与共享GATE，230保留历史参考。只做这一个候选，不同时改域风险比例、标签、特征子集或步长规则。不预设上下文位一定有效，也不能用它弥补稀少STOP正例/8调用的独立支持不足。','','36份输入导出全部核对，41维梯度有限差分通过。所有源码输入先发布，全部模型步长先发布、全部预测先发布后连接开发结果。当前拟合0、外层评分0。没有新增物理仿真、最终8、电网、论文或FRAME比较；基础NET仍outer12，内部12天保护不独立。'];text='\n'.join(lines).replace('40?原始41维','原始41维');(O/'GRID249_plan.md').write_text(text+'\n',encoding='utf8');C.archive('audited');dest=R/'论文协作资料/FRAME_原决策上下文资格_GRID249';dest.mkdir(exist_ok=False)
for n in ['GRID249_plan.md','qualification.json','independent_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
msg='GRID249公开上下文资格完成，0拟合/新NET/外层评分或标签。原q0/q1重建与2851训练域一致，85保护原STOP、4681外层入口全STOP；36导出/41维有限差分通过。固定250唯一支线在247公共39后加nativeSTOP位，40输入41参、共享斜率+正则化域偏置，原标签/权重/父b/85保护不变。11线性+11标量/600秒，主匹配247；明确多1参/函数类变化，不设有效预期，不改半半风险或标签。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID249原决策上下文资格\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
 else:t+=entry
 p.write_text(t,encoding='utf8',newline='\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID249后\n\n1. 248只读与249资格完成，下一250只消费249qualified_freeze输入，40输入41参，11线性+11保护步长/600秒、1KEEP，不重做旧模型。\n2. 原219均值NLL+单位完整L2、221原规则、原L-BFGS-B和零41初始化；全部success且原始41维gradinf<.001后才评分，失败保留停止，不救结果。主匹配247/共享GATE，230历史参考，所有种子反例保留。\n3. 源码输入先发，全部12权重步长先发，全部12预测先发后连结果。上下文来自原NET已知动作，0新NETforward；标签/权重/父b/85保护不变。不可同时改风险比例/标签或扫域模型。\n4. 每阶段最多22拟合/600秒不延长，原费用/控制器/排序别名/锁站/128/64/停滞2/.9不变；最终8/物理/电网/论文/FRAME不动，不安装依赖。Git严格串行等helper退出。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID249原决策上下文资格\n\n[方案](protocols/'+A.name+'/audited/GRID249_plan.md)；公开可得性通过，下一轮仅增加一个上下文参数。\n\n').encode()+p.read_bytes());print('GRID249_AUDITED')
