from pathlib import Path
import sys,json,shutil
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO;C.verify();q=C.read(O/'qualification.json');assert q['passed'] and C.read(O/'complete.json')['actualfits']==0
for r in q['pools']:
 pool=r['pool'];raw=np.load(C.FULL/'fit_features'/f'{pool}.npz');lookup={int(i):j for j,i in enumerate(raw['ids'])};sc=np.load(C.BASE/'models'/f'{pool}.npz');center=float(sc['mean'][81]-sc['mean'][80]);denom=float(np.hypot(sc['scale'][80],sc['scale'][81]));assert center==r['gap_center'] and denom==r['reference_scale']
 for folder in ['fit_inputs','calibration_inputs']:
  old=np.load(C.Q/folder/f'{pool}.npz');new=np.load(O/folder/f'{pool}.npz');assert np.array_equal(new['X'][:,:40],old['X']);ix=np.array([lookup[int(i)] for i in old['ids']]);assert np.array_equal(new['X'][:,40],(raw['x'][ix,81]-raw['x'][ix,80]-center)/denom)
  for k in old.files:
   if k!='X':assert np.array_equal(old[k],new[k])
 old=np.load(C.Q/'outer_parent_public'/f'{pool}.npz');new=np.load(O/'outer_parent_public'/f'{pool}.npz');h=np.load(C.HELD/f'held_{pool}.npz');assert np.array_equal(new['X'][:,:40],old['X']) and np.array_equal(new['X'][:,40],(h['features'][:,81]-h['features'][:,80]-center)/denom);assert np.array_equal(new['ids'],old['ids']) and np.array_equal(new['b'],old['b']) and np.array_equal(new['base_CONT'],old['base_CONT'])
C.save('independent_audit.json',dict(passed=True,all36input_transforms_reproduced=True,allpublished_parent83_scalers_unchanged=True,old40inputs_labels_weights_nativebit_b12_85constraints_unchanged=True,all42finite_difference_checks=True,newfits=0,newheldscores=0,newNETforward=0,heldoutcomesread=0));lines=['# GRID251：原模型连续动作分数差的资格','','本轮只增加调用前已经算出的原NET两动作分数差。它表示原模型的相对动作偏好强度，不是经过校准的回本概率或不确定性保证。没有运行新NET、拟合温度或筛选外层阈值。','','固定公式为：gap=((q_continue−q_stop)−(parent_mean81−parent_mean80))/sqrt(parent_scale80²+parent_scale81²)。两均值和两尺度完全复用原208已发布的83维scaler，组合参考尺度不是新拟合的gap标准差。12份scaler的原发布哈希逐项通过。','','250的40维输入后追加这一列，形成41输入、42参数。全部36份训练/保护/公开外层输入逐项重现，原40维输入、标签、风险权重、原STOP位、父b12、85条保护与原父CONT入口不变；2851自身fit8及4681公开外层向量均有限，原NET接口一致。资格过程没有读取外层结果。','','下一252只做这一支线：11个42参数线性模型加11个原221最大正例保护步长，1个固定KEEP，单次600秒；原219均值NLL+全参数单位L2、原L-BFGS-B和零42初始化。全部success且原始42维梯度<.001后才评分，失败保留停止，不增迭代或择种子。主要匹配250，其余历史候选保留。','','这增加一个输入和一个参数，明确改变函数类与惩罚中心的可调整方向，不称严格同容量，不预设会改善；不同时改风险比例、标签、alpha规则或网络。父scaler/NET仍来自outer12，内部12天保护不独立；这些记录均为已暴露开发数据。','','42维梯度有限差分与输入独立重现已通过。代码输入先发布、所有模型步长先发布、所有预测先发布后连接旧开发标签。当前拟合0、外层评分0，没有新增物理仿真、最终8、电网、论文或FRAME比较。']
(O/'GRID251_plan.md').write_text('\n'.join(lines)+'\n',encoding='utf8');C.archive('audited');dest=R/'论文协作资料/FRAME_连续动作偏好资格_GRID251';dest.mkdir(exist_ok=False)
for n in ['GRID251_plan.md','qualification.json','training_gap_support.json','independent_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
msg='GRID251仅资格0拟合/外层评分或结果完成。原192/190缓存q1-q0及12原208parent83scaler组合参考尺度公开可得，36导出/42维梯度通过；原40维、行标签权重/STOP位/b12/85保护不变。固定252唯一额外连续动作偏好输入，41输入42参，11线性+11步长/600秒、1KEEP，主匹配250。不是校准概率/不确定性保证，不改风险/alpha/阈值/网络。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID251连续动作偏好资格\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
 else:t+=entry
 p.write_text(t,encoding='utf8',newline='\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID251后\n\n1. 251资格完成，下一252只消费251qualified_freeze和输入，41输入42参，11线性+11原保护步长/600秒、1KEEP，不重做250。\n2. 原219meanNLL+完整单位L2、原L-BFGS-B零42初始化，全部success/原始gradinf<.001才评分，失败保留不救结果。主匹配250/共享GATE，全种子日期反例保留，不改风险/标签/阈值/alpha规则或同时加网络。\n3. 源码输入先发，所有12权重步长先发，所有12预测先发后连旧开发结果。原连续qgap是动作偏好分数，不是校准置信概率；复用原208尺度，无新NETforward。\n4. 原控制器/费用/排序别名/锁站/128/64/停滞2/.9保持；最终8/物理/电网/论文/FRAME不动，不安装依赖；Git严格串行等helper退出。\n',encoding='utf8');p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID251原NET连续动作偏好资格\n\n[方案](protocols/'+A.name+'/audited/GRID251_plan.md)；仅资格，固定一个连续公开输入后再训练。\n\n').encode()+p.read_bytes());print('GRID251_AUDITED')
