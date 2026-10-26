"""Interpret diagnosis without editing prior models,thresholds or reports."""
from pathlib import Path
import shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
from sklearn.metrics import roc_auc_score
def main():
    C.verify();assert C.read(C.O/'review.json')['passed'];assert not (C.O/'complete.json').exists();s=C.read(C.O/'summary.json');pm=C.read(C.O/'pool_metrics.json');acc=C.read(C.O/'opportunity_accounting.json');cal=C.read(C.O/'calibration.json');old=C.read(C.R/'outputs/grid190_extension_gate/gate_predictions.json');checks=0
    for p in pm:
        f=p['fold'];seed=p['seed'];z=np.load(C.P/'fit_features'/f'f{f}_s{seed}.npz');cp=np.load(C.P/'models'/f'f{f}_s{seed}.npz');logit=z['standard']@cp['coef'][0]+cp['intercept'][0];y=z['y'];w=z['mass'];assert abs(roc_auc_score(y,logit,sample_weight=w)-p['fit']['weighted_AUC'])<1e-12
        stable=np.maximum(logit,0)+np.log1p(np.exp(-abs(logit)))-y*logit;assert abs(w@stable-p['fit']['weighted_BCE'])<1e-12
        h=np.load(C.P/f'held_f{f}_s{seed}.npz');rr=[v for v in old if v['fold']==f and v['seed']==seed];hy=np.array([v['actual_next_change'] for v in rr]);hw=C.weights(rr);assert abs(roc_auc_score(hy,h['logit'],sample_weight=hw)-p['held']['weighted_AUC'])<1e-12;checks+=2
    for part in ['fit','held']:assert abs(sum(v['pool_equal_mass'] for v in cal if v['partition']==part)-1)<1e-12
    a=s['fit_held']['fit'];b=s['fit_held']['held'];waste=acc['categories']['WASTE']['paid_query_mean'];hitq=acc['categories']['HIT']['paid_query_mean'];ratio=b['weighted_next_gain']/a['weighted_next_gain'];support=C.read(C.O/'conditional_support.json')
    lines=['# GRID191：单批门控为什么仍会白查和漏查','',
        '判断：保留GRID190跨种子的局部小改善。当前优先问题是训练侧与留出侧的区分能力、停止点分布存在明显差异，不是简单把线性门控换成更大网络。下一次可检验训练覆盖这一因素，先保持模型和读出不变。','',
        '## 拟合与日期留出差异','',
        '两侧都按日期→物理调用组→起点等权，再对12个模型平均。下表与上轮84/278的原始计数口径不同，没有改写上轮结果。','',
        '|指标|拟合侧|留出侧|','|---|---:|---:|']
    for name,key in [('动作变化正例比例','weighted_prevalence'),('追加命中率','weighted_precision'),('机会召回率','weighted_recall'),('排序AUC','weighted_AUC'),('概率BCE','weighted_BCE'),('拟合侧先验常数的BCE','constant_fitprior_BCE')]:lines.append(f"|{name}|{a[key]:.6f}|{b[key]:.6f}|")
    lines+=['','拟合侧排序和识别较强，留出侧明显减弱；留出概率BCE甚至高于使用拟合侧正例比例的常数预测。不能只凭拟合能力说模型已经学稳，也不能单凭这一差距断定增加容量能修复。','',
        f"停止点中的下一批净收益为正比例，拟合侧约{100*a['weighted_profitable_fraction']:.2f}%，留出侧约{100*b['weighted_profitable_fraction']:.2f}%；下一批平均奖励增量约相差{ratio:.1f}倍。这里比较的是两个条件停止集合，不是全部控制状态的分布；原NET在拟合日期和留出日期的错误本身，也会改变门控看到什么。因而不能把全部落差叫作门控单独过拟合。",'',
        '## 停止点筛选排除了什么','',
        f"每折全部可达拟合状态中，下一批NET为正的有{s['all_fit_profitable_range'][0]}–{s['all_fit_profitable_range'][1]}个；门控只取原NET首次STOP后，保留{s['conditional_profitable_range'][0]}–{s['conditional_profitable_range'][1]}个。排除多数这类状态并非数据处理错误：原NET在训练侧通常已经选择继续，因此它们本来就不会进入门控。",'',
        '但部署到留出日期时，NET可能在这些较大收益机会面前误停，门控需要识别的状态便不同。训练侧把原模型已会处理的过程全部排除，可能削弱门控对此类机会的覆盖。这是值得单独检验的支持变化假设，尚不是因果结论。','',
        '## 额外查询具体花在哪里','',
        '|类别|配对数|追加查询的日期宏平均贡献|可获得奖励增量的日期宏平均贡献|','|---|---:|---:|---:|']
    for key,label in [('HIT','追加且改选'),('WASTE','追加但未改选'),('MISS','未追加、存在改选机会'),('CORRECT_SKIP','未追加且没有改选')]:
        z=acc['categories'][key];lines.append(f"|{label}|{z['records']}|{z['paid_query_mean']:.6f}|{z['potential_gain_mean']:.9f}|")
    lines+=['',f"额外候选查询中约{100*waste/(waste+hitq):.2f}%花在没有改选的批次。仍有528个遗漏的改选机会，来自{acc['categories']['MISS']['physical_groups']}个物理调用组；其中前三组贡献约{100*acc['top3_missed_gain_fraction']:.2f}%的遗漏奖励增量。这里未追加的收益是保存教师反馈中的事后机会，不是已经实现的收益。",'',
        '## 下一次最小对照','',
        '保留当前STOP域样本，并补入原NET实际停止前的继续状态；补充集合按原模型动作确定，必须包含其中正、负两类，不按将来收益挑选。原NET及其83维表示、84参数门控、C=1、阈值0.5均保持。只检验覆盖是否有帮助，不同时换损失或扩网络。','',
        '候选训练分布固定为一半原STOP域、一半停止前CONTINUE域；各域内部沿用日期/调用组/起点均衡，继续域再平分该起点的可达继续步。继续保留原STOP侧标准化，训练总权重与旧门控相同，以避免改变特征尺度和L2正则强度。这个混合会改变训练分布，输出不能未经校准就解释为STOP域的条件概率。','',
        '全部原NET决定仍保留，门控只在首次STOP后可选追加一批。候选比较要保留原门控和无条件ADD1；还应在各日期/种子按较小的额外查询预算，对双方追加独立随机稀释，计算同预期查询量下的收益和动作一致。概率只由查询量确定，不用奖励选概率；这是数学期望比较，不是实际硬预算或在线时间匹配。','',
        '没有充分理由直接重做普通共享辅助头或盲目改加权分类。本轮也未训练上述候选；需要另行冻结具体对照并在训练前通知。','',
        '## 核验与工程记录','',
        f"16565条拟合分数从已保存83维表示与冻结系数回读，分数极值逐项重现原训练记录；4681条留出分数直接复用。{checks}组加权AUC由独立库复算，BCE用稳定等价公式复核，概率分箱质量与全部查询/机会分解守恒。没有神经网络前向、参数更新、新电网预测或新的门槛。",'',
        '常数先验BCE的首次计算因NumPy不支持对布尔数组直接取负而停止，使用等价浮点系数公式恢复；原冻结源保留，模型、分数、样本、标签、权重和口径未改。默认沙箱首次创建D盘输出被拒绝，确认未生成目录后以授权路径权限执行原准备步骤；均不作为方法失败。工程收据见engineering_recovery01.json。']
    (C.O/'GRID191_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('decision.json',dict(priority='Test conditional support augmentation with frozen capacity/value baseline',do_not_prioritize='larger gate or changed threshold solely from these diagnostics',fit_vs_held_AUC=[a['weighted_AUC'],b['weighted_AUC']],support_hypothesis_not_causal_proof=True,no_new_fits_this_stage=True))
    C.save('publication_audit.json',dict(passed=True,independent_weighted_AUC_checks=checks,BCE_and_calibration_conservation=True,original_source_models_thresholds_unchanged=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.verify();C.save('complete.json',dict(passed=True,fit_linear_readbacks=16565,held_cached_rows=4681,new_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=C.R/'论文协作资料/FRAME_门控迁移与训练覆盖_GRID191';dest.mkdir(exist_ok=False)
    for n in ['GRID191_report.md','summary.json','decision.json','pool_metrics.json','conditional_support.json','opportunity_accounting.json','publication_audit.json','complete.json']:shutil.copy2(C.O/n,dest/n);assert C.sha(C.O/n)==C.sha(dest/n)
    msg='GRID191已有门控诊断：拟合/held加权AUC .944/.629，命中83.10%/28.28%，heldBCE .709劣于fit先验常数.388。条件停止点中正NET比例1.32%→5.63%、平均增量约25倍；原每折130–174个正NET拟合状态筛STOP后仅9–23。保留GRID190小改善，优先单变量测试STOP+停止前CONT覆盖，84参数/原标准化/总权重/阈值不改，不先扩模型。只是支持变化假设，当前零新拟合/NET前向/物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-26 — GRID191门控迁移与覆盖诊断\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a0,b0=t.split('\n',1);t=a0+'\n'+e+'\n'+b0
        else:t+=e
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID191后\n\n1. 保留GRID190重复小改善；冻结一个STOP与停止前CONT域覆盖对照：不按收益选补充样本，一半各域、原83维/84参数/STOP标准化/总权重/C1/.5不变，具体训练前通知。\n2. 先核对补充样本只来自相应拟合日期和原NET实际继续路径。双方额外查询若不同，按较小预算独立稀释比较数学期望，不用奖励定概率；不能声称硬预算/在线时间匹配。\n3. 原NET冻结，全部12端点先发布再held，不扩物理/数据/最终8/论文。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID191门控覆盖诊断\r\n\r\n[报告](protocols/GRID191_gate_diagnosis/complete/GRID191_report.md)；拟合与held落差明显，优先检验覆盖，保留已有小改善。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID191_COMPLETE')
if __name__=='__main__':main()
