from pathlib import Path
import csv, hashlib, json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fa02_end'
a=json.loads((OUT/'FA02_analysis.json').read_text());audit=json.loads((OUT/'FA02_audit.json').read_text())
state=json.loads((OUT/'RUN_STATE.json').read_text());rows=a['results']
verdict={'ACCEPT_CHEAP_BASELINE':'接受便宜基线，停止当前配置的低预算主动选例主张',
         'RETAIN_LOW_BUDGET_HYPOTHESIS':'保留低预算主动选例假设；下一步需要独立封存数据与成熟强对手确认',
         'UNDETERMINED':'不确定；不追加种子、预算、模型或批量搜索'}[a['verdict']]
text=['# FA02-END：统一末端拟合检查','',f'裁决：**{a["verdict"]}**。{verdict}。','',
'这是已看测试折上的开发检查。0.25B来自事后前缀诊断；不是独立确认，不改变FA00/FA01既有裁决，未建立论文新意或GRL优势。','',
'## 主要结果','', '| 方法 | ALL seed 7 | ALL seed 42 | ALL seed 99 | ALL 均值 | COMPLETE 均值 |',
'|---|---:|---:|---:|---:|---:|']
for arm in ['FIXED-100','RANDOM-100','CHEAP-100']:
    vals=[next(r['par10'] for r in rows if r['arm']==arm and r['mode']=='ALL' and r['seed']==s) for s in [7,42,99]]
    text.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in vals)+f' | {a["means"]["ALL"][arm]:.6f} | {a["means"]["COMPLETE"][arm]:.6f} |')
text+=['','PAR-10越低越好。相对改善=(对照均值−原方法均值)/对照均值；不把三次种子重放当作三套独立场景。','',
'| 模式／对照 | 主动相对改善 | 主动更好的种子数 |','|---|---:|---:|']
for key,v in a['comparisons'].items():text.append(f'| {key} | {100*v["relative_improvement"]:.4f}% | {v["better_seed_count"]}/3 |')
text+=['','## COMPLETE到ALL：只能解释组内补入观测的影响','',
'| 方法 | seed | COMPLETE | ALL | ALL−COMPLETE | 追加付款观测 | 尾部已付款秒数 |','|---|---:|---:|---:|---:|---:|---:|']
for arm in ['FIXED-100','RANDOM-100','CHEAP-100']:
    for seed in [7,42,99]:
        c=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['mode']=='COMPLETE')
        r=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['mode']=='ALL')
        text.append(f'| {arm} | {seed} | {c["par10"]:.6f} | {r["par10"]:.6f} | {r["par10"]-c["par10"]:+.6f} | {r["paid_count"]-c["paid_count"]} | {r["acquisition_cost_s"]-c["acquisition_cost_s"]:.6f} |')
text+=['','原保存模型前缀均值：FIXED 723.463369、RANDOM 823.910901、CHEAP 851.159444。它们使用各臂不同轮数的模型随机种子，只是探索参照。COMPLETE和ALL统一使用keyed_seed(seed,"pair",0,pair_index)，成本回归器同样采用round=0；二者组内差值隔离补入观测变化。原历史前缀到COMPLETE的变化含重新设定模型随机状态，不能全部归为尾部数据作用。','',
'## 协议与审计','',
'三方法×三既有种子×COMPLETE/ALL=18次完整选择器拟合。每次沿用100棵树、55个成对分类器和11个成本回归器的原模型规格。未重新选候选、未新增求解器采集、未修改采样、批量、截止或数据划分。全部终端预测落盘后才读取测试成绩评分。','',
'ALL只保留0.25B=86460模拟CPU秒内已付款结果；跨边界观测转为剩余预算下界。COMPLETE复原最近完整拟合使用的付费输入。首个成对标签保留，再从合法末端缓存补齐可推导标签。初始化计费；缓存不重复计费。','',
'拟合前预检发现并修正浮点累加边界问题：约1e-12秒的账目差可能把已付款完成误转为删失。重建采用1e-8秒账目容差；修正发生在正式冻结与任何终端拟合之前。没有作废或挑选终端模型。','',
f'审计使用此前已审计的FA00_data.npz，通过独立的收费和标签计算路径重算 {audit["raw_matrix_paid_items"]:,} 项付费输入及 {audit["independently_derived_labels"]:,} 条成对标签，验证18个模型参数、随机种子与训练输入；核对 {audit["scored_predictions"]:,} 个测试预测。本轮没有重新解析ARFF，也没有独立重新拟合18个模型。54次检查是同一已拟合模型和共享Selector预测路径的重复一致性检查，不是三套不同适配器的独立验证。最大收费误差 {audit["max_charge_error_s"]:.3g} 秒，原输入及冻结代码哈希未变。预检还通过九组隐藏后缀扰动和完整批次标签数检查。','',
f'18次拟合与前缀重建实测 {state["wall_s"]:.2f} 秒，独立审计 {audit["wall_s"]:.2f} 秒。纯拟合合计 {sum(r["fit_s"] for r in rows):.2f} 秒；首次正式冻结位于FA02_freeze.json。','',
'18个模型、可见缓存、合法付费前缀、标签、完整预测分别保存于models/。FA02_per_fit.csv为18行；FA02_test_predictions.csv为18×129=2322行。','',
'## 边界与研究意义','',
'该检查使用同一ASP-POTASSCO场景和已经查看过的测试折。三种子是初始化与算法随机性重放。所有结果只支持当前适配与预算，不等同完整FrugalAS复现或整个主动学习领域的结论。','',
'这是现有轨迹前缀重建后新增末端拟合的交付规则检查，不是重新执行的独立低预算采集策略。末端拟合费用在获取预算之外统一单列；没有把矩阵模拟CPU秒与本机拟合墙钟换算成生产加速。特征费用仍存在722项缺失，不宣称端到端成本最优。','',
'无新采样器、神经网络或GRL。即使判为保留，也只是当前基线的投资资格，尚需独立场景、成熟强基线、可辨识信息和相对现有工作的贡献证明。','',
'来源：继承FA00/FA01冻结数据和适配。原始代码：[FrugalAS固定版本](https://github.com/stacs-cp/JAIR2026-FrugalAS/tree/7a5727651a92fd2fa4960dcbbd7f6dab94130028)。']
(OUT/'FA02_report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
manifest=[]
for f in sorted(OUT.rglob('*')):
    if f.is_file() and f.name!='FA02_delivery_manifest.json':
        manifest.append(dict(path=str(f.relative_to(OUT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
(OUT/'FA02_delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(verdict)
