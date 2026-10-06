from common import *
import csv

def main():
    analysis=json.loads((OUT/'FA03B_analysis.json').read_text())
    audit=json.loads((OUT/'FA03B_audit.json').read_text())
    assert audit['passed']
    results=analysis['results']
    table=['| 方法 | seed 7 | seed 42 | seed 99 | 平均 PAR-10 |','|---|---:|---:|---:|---:|']
    for arm,mean in analysis['means'].items():
        vals={r['seed']:r['par10'] for r in results if r['arm']==arm}
        table.append(f'| {arm} | {vals[7]:.4f} | {vals[42]:.4f} | {vals[99]:.4f} | {mean:.4f} |')
    text=f"""# FA03B：ASP 低预算强对照补齐

**冻结裁决：{analysis['verdict']}。** 六条新增轨迹完成，独立反馈、费用、排序、标签与末端交付审计通过。

这是同一 ASP-POTASSCO 已看测试折上的开发筛查，不是 FA03/QBF 的跨场景确认；不改变历史 FA01/FA02 裁决，不构成新算法贡献。

## 主表

{chr(10).join(table)}

主动方法为 FIXED-100 的原 Uncertainty+PredCost Pareto 规则。前三臂直接引用 FA02-END 的 ALL 末端模型并重新核对评分，没有重跑采集或拟合。

## 判据

"""
    for arm,c in analysis['comparisons'].items():
        text+=f"- 相对 {arm}：主动均值改善 {100*c['improvement']:.4f}%；主动更好的种子 {c['better_seeds']}/3；对照进入102%收口范围：{c['close']}。\n"
    formal=[r for r in results if not r.get('inherited')]
    text+=f"""

继续要求主动相对全部四个对照平均至少改善5%，各至少2/3种子同向；任一新增对照均值不高于主动的102%则接受强基线，其余记不确定。这些是投资筛查尺度，不是等价或显著性保证。

## 成本与诊断

六条新增轨迹记录的主动执行墙钟合计 {sum(r['wall_s'] for r in formal):.1f} 秒；模型拟合 {sum(r['fit_s'] for r in formal):.1f} 秒；审计 {audit['wall_s']:.1f} 秒。墙钟包含模型与控制，不是运行真实ASP求解器的时间。终端文件压缩与哈希归档在逐运行结果计时点之后，单列为归档开销，不宣称端到端加速。

每轨迹上限86,460模拟CPU秒，原生截止600秒、采集截止100秒，初始化费用计入。特征费用仍含未知项；没有把评分罚分当收费。

| 新臂 | 平均实际执行数 | 平均成对标签 | 平均有效标签 | 平均完整已观察行 | 平均拟合秒 |
|---|---:|---:|---:|---:|---:|
"""
    for arm in ARMS:
        rs=[r for r in formal if r['arm']==arm]
        text+=f"| {arm} | {np.mean([r['executions'] for r in rs]):.1f} | {np.mean([r['labels'] for r in rs]):.1f} | {np.mean([r['informative_labels'] for r in rs]):.1f} | {np.mean([r['full_rows'] for r in rs]):.1f} | {np.mean([r['fit_s'] for r in rs]):.1f} |\n"
    lex=[r for r in formal if r['arm']==ARMS[0]]
    text+=f"""

LEX 在自身各轮合法状态下，与 CHEAP 假想选择集合的平均重合率（先各轨迹平均再三种子平均）为 {np.mean([r['cheap_overlap_mean'] for r in lex]):.4f}。有 {sum(r['secondary_keys_change_batches'] for r in lex)}/{sum(r['batch_count'] for r in lex)} 批执行顺序或集合受后两键影响。不是把它当成完整FrugalAS复现，也没有因同选调整优先级。

随机整实例臂免费推导所有合法配对标签；查询粒度和拟合频率均不同，这是完整基线策略，不能把胜负单独归因于粒度。标签多不自动表示信息多；删失权重10×下界是合法观测近似，不是真实PAR-10后悔。

## 审计边界与执行记录

独立从作者ARFF重读真实记录，与继承NPZ核对；按逐动作账目重算费用与反馈、可见池、LEX排序和行内顺序、首次标签与权重、最终缓存。所有最终预测冻结后才评分。原三臂九个末端预测也逐例复核。

核对 {audit['counts']['observations']:,} 项观测记录、{audit['counts']['label_events']:,} 个标签事件、{audit['counts']['batches']} 批、{audit['counts']['test_predictions']:,} 个测试预测；最大收费误差 {audit['max_fee_error_s']:.3g} 秒。末端模型重预测与保存值一致；没有独立重拟合每轮在线模型，因此不把这项审计写成所有预测器的独立复现。

预检用两个玩具执行检验断点续跑，原100树参数保留；这不是正式轨迹或神经网络训练。预检发现作者排序表的索引数组为只读，以及公共模块的导入路径优先级导致测试导入旧runner；两项已修正，未启动或作废任何正式轨迹。

QBF-2011资格阻断仍保留。此次没有重新选场景、调预算/种子/模型、训练GRL，或根据测试结果挑checkpoint。通过也只保留开发假设；收口只针对当前主动优势主张。

## 文件

- FA03B_analysis.json：逐运行成绩与冻结裁决。
- FA03B_audit.json：独立复算范围及误差。
- FA03B_per_run.csv、FA03B_test_predictions.csv：逐轨迹与逐例结果。
- FA03B_execution_freeze.json、FA03B_prediction_freeze.json：源代码与预测封存。
- runs/：全部动作、每轮候选、检查点、终端模型/缓存/标签与预测。

源码位于 work/fa03b/。执行：run.py；审核：audit.py；报告：report.py。原fa00/fa01/fa02代码和结果未改。
"""
    (OUT/'FA03B_report.md').write_text(text,encoding='utf8')
    fields=['arm','seed','par10','inherited','acquisition_s','wall_s','fit_s','labels','informative_labels','executions','full_rows','batch_count','cheap_overlap_mean']
    with (OUT/'FA03B_per_run.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    save(OUT/'FA03B_delivery_manifest.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='FA03B_delivery_manifest.json'})
    print(str(OUT/'FA03B_report.md'))

if __name__=='__main__':main()
