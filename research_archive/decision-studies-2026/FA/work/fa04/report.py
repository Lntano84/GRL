from common import *
import csv

def main():
    a=json.loads((OUT/'FA04_analysis.json').read_text());audit=json.loads((OUT/'FA04_audit.json').read_text())
    assert audit['passed']
    rows=a['results'];new=[r for r in rows if not r.get('inherited')]
    table=['| 方法 | seed7 | seed42 | seed99 | 平均 PAR-10 |','|---|---:|---:|---:|---:|']
    for arm,mean in a['means'].items():
        vals={r['seed']:r['par10'] for r in rows if r['arm']==arm}
        table.append(f'| {arm} | {vals[7]:.4f} | {vals[42]:.4f} | {vals[99]:.4f} | {mean:.4f} |')
    primary=a['comparisons']['RANDOM-ROW'];secondary=a['comparisons']['FIXED-100']
    text=f'''# FA04：粗粒度主动PC强基线

**主裁决：{a['verdict']}。** 三条新增轨迹完成并通过独立账目、反馈、标签、平均不确定性和排序复算。

## 成绩

{chr(10).join(table)}

主比较COARSE-PC对随机整实例：平均改善{100*primary['improvement']:.4f}%，{primary['better_seeds']}/3种子更好。次要比较对既有Pareto：平均改善{100*secondary['improvement']:.4f}%，{secondary['better_seeds']}/3种子更好。COARSE-PC处于Pareto均值的102%以内：{a['secondary_competitive_vs_pareto']}。

主判读先检查平均改善≥5%且至少2/3种子同向；否则平均改善≤2%为未达到实用平均收益尺度；其余记不确定。这不是等价检验或统计显著性保证。FA03B原联合裁决保持UNDETERMINED，不能凭本轮更换对照而授予它通过。

## 冻结方法与成本

20实例初始化、11算法、55个成对分类器、原RF参数与缓存语义不变；预算86,460模拟CPU秒（含初始化），固定采集截止100秒，原生截止600秒。每批平均55个模型的不确定性，取11个实例，查询全部仍合法算法。无有效模型项贡献0.5；已知标签对应模型仍参加平均。按原row_acquire/row_execute/row_alg随机流处理并列和执行顺序。最终用全部已付款观测统一末端拟合。

旧RANDOM-ROW和Pareto只引用并复核六份末端预测，不新增采集或拟合。

三条新增轨迹记录的主动执行墙钟合计{sum(r['wall_s'] for r in new):.1f}秒，其中模型拟合{sum(r['fit_s'] for r in new):.1f}秒；独立审计{audit['wall_s']:.1f}秒。这不是实际运行ASP求解器的时间。各运行墙钟覆盖控制、候选记录、拟合与预测，终端日志压缩和归档哈希在该计时点之后。新臂selection_s只记录一次完整规划区间，没有继承LEX的内部候选重复计时。费用与墙钟单列，特征成本仍有未知项，不宣称端到端加速。

| seed | 付费执行 | 缓存命中 | 全部成对标签 | 有效标签 | 观察全部算法的行 | 原生测试超时 | wall秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in sorted(new,key=lambda x:x['seed']):
        text+=f"| {r['seed']} | {r['executions']} | {r['cache_hits']} | {r['labels']} | {r['informative_labels']} | {r['full_rows']} | {r['test_timeouts']} | {r['wall_s']:.1f} |\n"
    text+=f'''
“观察全部算法的行”包含删失反馈，不等于所有完成时间已知。超时惩罚是评分而非支付费用；成对权重使用已见删失下界，是合法近似，不是真实PAR-10后悔。

## 审计与边界

预检包含55项手算（含无模型项）、同分随机次序、660项历史初始化反馈、两个实际隐藏世界相同前缀的动作不变、玩具100树模型的实际断点续跑和预算尾部。玩具拟合不计作新增ASP正式轨迹，已单列记录。

从原ARFF独立重读并核对数据。复算{audit['counts']['paid_observations']:,}项付费执行、{audit['counts']['observations']:,}项总观测记录、{audit['counts']['label_events']:,}个标签事件、{audit['counts']['batches']}批次与{audit['counts']['test_predictions']:,}个测试预测，最大收费误差{audit['max_fee_error_s']:.3g}秒。

对每轮封存的合法池与概率表，用逐列累计重算55项均值，再用独立一遍lexsort核对并列顺序，验证其输入缓存/标签哈希。该项没有独立重拟合或重新预测所有在线RF，不能声称每轮预测器独立复现。最终模型重新预测与封存文件一致；全部三份最终预测封存后才评分，不按checkpoint选优、不删尾部实例。

本方法依CP2026论文公式在既有合法协议下改编，尚未复核作者CP封存代码，不称忠实完整复现。没有加入超时预测或动态截止。来源：https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2026.38

这是同一已看ASP测试折的开发基线定位，三种子不是三个独立场景。无论裁决怎样，都不是新算法贡献，不启动GRL，不在此折追加预算、种子或采样器寻找赢家。下一次算法开发需先明确新假设和未用于开发的评价协议。

## 文件

- FA04_analysis.json、FA04_per_run.csv：冻结判读、逐运行结果。
- FA04_test_predictions.csv：全部129实例与三种子的逐例成绩，包含六份旧预测。
- FA04_audit.json：独立复算范围和误差。
- FA04_preflight.json、FA04_execution_freeze.json、FA04_prediction_freeze.json：预检与封存。
- runs/：全部反馈、概率表、检查点、最终模型/缓存/标签/预测与文件哈希。

执行：work/fa04/run.py；审计：work/fa04/audit.py；报告：work/fa04/report.py。旧代码和输出不改。
'''
    (OUT/'FA04_report.md').write_text(text,encoding='utf8')
    fields=['arm','seed','par10','test_timeouts','inherited','acquisition_s','executions','cache_hits','labels','informative_labels','full_rows','wall_s','fit_s','selection_s']
    with (OUT/'FA04_per_run.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    save(OUT/'FA04_delivery_manifest.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='FA04_delivery_manifest.json'})
    print(str(OUT/'FA04_report.md'))

if __name__=='__main__':main()
