"""Post-score independent Decimal reconciliation and transparent packaging."""
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fa06'
DATA=ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO'
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    d=read(ROOT/'outputs/fa06_design/FA06_design_freeze.json');scores=read(OUT/'FA06_scores.json')
    audit=read(OUT/'FA06_training_audit.json');repair=read(OUT/'FA06_logging_repair.json')
    assert audit['status']=='passed' and repair['status']=='verified'
    by_id={s:i for i,s in enumerate(d['ids'])};by_algo={s:a for a,s in enumerate(d['algorithms'])};test=set(d['test'])
    expected={}
    with (DATA/'algorithm_runs.arff').open() as f:
        for line in f:
            if line.strip().lower()=='@data':break
        for row in csv.reader(f):
            if not row or int(row[1])!=1 or by_id[row[0]] not in test:continue
            assert row[4] in ['ok','timeout']
            expected[(by_id[row[0]],by_algo[row[2]])]=Decimal(row[3]) if row[4]=='ok' else Decimal(6000)
    totals={};witnesses=0
    with (OUT/'FA06_per_instance.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            group=(row['arm'],int(row['seed']));i,a=int(row['row']),int(row['algorithm'])
            pred=np.load(OUT/'fits'/f'{group[0]}_s{group[1]}'/'predictions.npy')
            assert int(pred[d['test'].index(i)])==a
            value=expected[(i,a)]
            assert math.isclose(float(row['native_par10_s']),float(value),rel_tol=1e-12,abs_tol=1e-12)
            totals[group]=totals.get(group,Decimal(0))+value;witnesses+=1
    for r in scores['per_run']:
        value=float(totals[(r['arm'],r['seed'])]/Decimal(129))
        assert math.isclose(value,r['par10_s'],rel_tol=1e-12,abs_tol=1e-12)
    means={a:sum(totals[(a,s)]/Decimal(129) for s in d['seeds'])/Decimal(3) for a in d['weights']}
    gains={a:1-means[a]/means['ORIGINAL'] for a in ['UNIT','NATIVE-UPPER']}
    directions={a:sum(totals[(a,s)]<totals[('ORIGINAL',s)] for s in d['seeds']) for a in gains}
    signal=[a for a in gains if gains[a]>=Decimal('.05') and directions[a]>=2]
    robust=all(1-means['ORIGINAL']/means[a]>=Decimal('.05') and
               sum(totals[('ORIGINAL',s)]<totals[(a,s)] for s in d['seeds'])>=2 for a in gains)
    close=all(abs(means[a]/means['ORIGINAL']-1)<=Decimal('.02') for a in gains)
    verdict='simple_rule_development_signal' if signal else 'original_robust' if robust else 'close_in_this_protocol' if close else 'undetermined'
    assert verdict==scores['verdict']
    save('FA06_score_reconciliation.json',dict(status='passed',decimal_witnesses=witnesses,
        means={a:float(v) for a,v in means.items()},improvements={a:float(v) for a,v in gains.items()},
        directions=directions,verdict=verdict,independent_rf_refits=0))
    feature_cost={k:dict(known_seconds=0.,missing_cells=0,instances=0) for k in ['train','test','reserved_unused']}
    membership={i:k for k in feature_cost for i in d[k]}
    with (DATA/'feature_costs.arff').open() as f:
        for line in f:
            if line.strip().lower()=='@data':break
        for row in csv.reader(f):
            if not row or int(row[1])!=1:continue
            entry=feature_cost[membership[by_id[row[0]]]];entry['instances']+=1
            for v in row[2:]:
                if v=='?':entry['missing_cells']+=1
                else:entry['known_seconds']+=float(v)
    save('FA06_feature_costs.json',dict(by_split=feature_cost,
        interpretation='Provided trace feature costs, separately counted; unknown cells are not zero; reserved_unused feature processing is not algorithm-feedback acquisition.'))
    distributions=[]
    for seed in d['seeds']:
        tables=read(OUT/'collections'/f's{seed}'/'labels.json')
        for arm in d['weights']:
            fractions=[];label_fractions=[]
            for table in tables:
                censor=[r for r in table if (r['ka'],r['kb'])!=(2,2)]
                fractions.append(math.fsum(r[arm] for r in censor)/math.fsum(r[arm] for r in table))
                label_fractions.append(len(censor)/len(table))
            distributions.append(dict(seed=seed,arm=arm,
                censor_raw_weight_fraction_model_median=float(np.median(fractions)),
                censor_label_fraction_model_median=float(np.median(label_fractions))))
    save('FA06_weight_descriptions.json',dict(rows=distributions,
        interpretation='Model-wise raw weight composition only, not RF influence or actual loss mechanism.'))
    rows=[]
    for a,label in [('ORIGINAL','原代理'),('UNIT','单位权重'),('NATIVE-UPPER','原生上界代理')]:
        rs=[r for r in scores['per_run'] if r['arm']==a]
        cells=[f"{next(r['par10_s'] for r in rs if r['seed']==s):.2f}" for s in d['seeds']]
        improvement='—' if a=='ORIGINAL' else f"{scores['improvements_vs_original'][a]*100:+.2f}%"
        rows.append('| '+' | '.join([label,*cells,f"{scores['means'][a]:.2f}",improvement,'—' if a=='ORIGINAL' else f"{scores['better_seeds'][a]}/3"])+ ' |')
    first=read(OUT/'FA06_predictions_seal.INVALID_logging.json')
    corrected=read(OUT/'FA06_predictions_seal.json')
    fit_total=math.fsum(r['fit_s'] for r in scores['per_run'])
    report='''# FA06：末端权重处理机制检验

完成日期：2026-10-05。九次PC末端拟合完成，独立费用/反馈/投票审计与Decimal成绩复算通过。

**冻结裁决：UNDETERMINED（不确定）。** 单位权重平均与原代理接近；原生代价上界代理更差。没有任何替代规则通过“平均至少改善5%、至少2/3种子同向”。总体不能进入“接近”档，因为上界代理超出±2%。不追加权重、种子、折或预算；不启动GRL或付费解除删失策略。

## 1. 主结果：原生测试PAR-10，越低越好

| 方法 | seed7 | seed42 | seed99 | 均值 | 相对原代理改善 | 更好的种子 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

改善为 `1−mean(P_method)/mean(P_original)`，不是平均逐种子百分比。均值先按每个种子的129个评价实例计算，再对三个种子等权。

冻结四档逐项核对：

- 简单规则信号：UNIT为−0.1703%、1/3同向；NATIVE-UPPER为−4.8033%、0/3同向，均未通过。
- 原代理稳健档：原代理对UNIT的平均优势仅约0.17%，对上界代理约4.58%（以替代臂为分母），未达到对二者均至少5%。
- 接近档：UNIT均值在±2%内，上界代理不在；“两替代均接近”不成立。
- 因而落入其余情况：不确定。没有因UNIT接近而临时改为全局收口，也没有因上界3/3更差而跳过原判据。

## 2. 权重改变了动作，但没有建立实用平均质量优势

相对原代理，UNIT分别改变39/40/37个评价实例的算法选择，NATIVE-UPPER改变16/23/19个；每个种子评价集合均为129个实例。权重处理确实影响最后选择，不能说“模型完全不受权重影响”。

原代理原生超时次数为14/13/12，UNIT为13/13/13，上界为15/13/13。UNIT在seed7更好、seed99更差，三种子超时总数相同；均值接近不意味着逐实例效果相同，也不是等价检验。

第三臂提前删失时采用6000−v，是合法信息下的悲观替代，并没有恢复真实尾部时长。其成绩不佳既不能否定成熟生存模型，也不能证明追加真实后续反馈无用。原始权重份额描述另存，不用它推断随机森林影响或失败原因。

## 3. 输入、预算与评价边界

官方ASP-POTASSCO repetition1/fold2评价129行；新train1048行，reserved117行不用。初始化和全部反馈只来自新train。预处理重新在新train拟合，独立从源特征复算缺失列筛选、列中位数和标准化参数。

三条正式随机整实例缓存：seed7/42/99分别2143/2386/1999次付费执行、8068/9385/7249条正权重非平局偏好；每条费用86460模拟CPU秒。合计6528次付费执行、24702条共享偏好。三臂在每个种子内共享全部标签、支持、特征、RF参数和逐对随机种子；末端共九次拟合，不拟合采集成本回归器。

预算来自矩阵回放，不是实际运行ASP求解器的费用。采集无中间拟合、模型不参与选例，因此这不是对历史Pareto或粗粒度PC采样器的重排名，也不直接与FA02成绩作方法比较。

三种子共享同一个评价集合，不是三个独立数据集。fold2有120行属于旧训练集合、9行属于旧验证集合；本次采集与拟合排除它们，但这是同一旧场景的新切分开发检验，**不是新数据或未见实例的独立确认**。

所有预测封存、训练审计通过之后，独立过程才转换评价行的原生成绩用于一次评分；保留全部129行，不按难度/超时/可解性过滤。保留集未用于成绩评价。训练加载器只转换train的运行数值和状态；其他行使用环境占位值，隐藏评价值扰动通过真实采集runner检验。

## 4. 记录器缺陷与修复（必须随结果披露）

首次批次的新动作日志没有写盘：`open('ab')`误放在“文件已存在”的分支内。此前预检覆盖了已存在日志的恢复，却遗漏首次文件创建；独立审计在评分前因找不到日志中止。**评价成绩当时尚未读取。**

旧三条无日志缓存、旧运行冻结与预测封存保留为 `INVALID_logging`；修正只改变日志创建与持久化，补入真实“首次创建并逐条落盘”守卫后，从空缓存重放三条轨迹。

新旧三份缓存和标签共六个文件SHA-256逐项完全相同，费用、执行数、完成/删失数和训练输入签名也相同。因此九份已完成模型被保留并复用，没有再次训练九份RF。独立审计逐项验证新标签、训练行、标签方向、三种权重、模型随机种子/参数，加载全部495个成对模型重新预测并复核投票。修复证书见 `FA06_logging_repair.json`。

这不是把无日志批次直接当作有效轨迹使用：最终收费与合法性证据来自修正后的完整日志。模型复用成立于训练输入完全相同；本轮没有按成绩改策略或挑选重放结果。

另有一次启动时的新审计工具模块同名导致循环导入，在正式采集前修复；随机整实例函数复制后经AST核对与冻结源函数一致。该工程修复没有改变采样规则。

## 5. 验收与成本

八项最终预检通过：权重手算、反向胜负、平局支持、预算尾部、相等下界类型、破并列、仅训练成绩加载、真实隐藏评价扰动、恢复与首次日志创建的相关检查合并为八个记录项。

独立审计覆盖6528次付费执行、24702条共同偏好、九份完整拟合输入、495个已训练模型的复预测与投票；1161个“权重×种子×评价实例”评分用源运行表和Decimal算术另行对账。**没有独立重新拟合RF**，也不把1161次评分当作独立样本。

1810个历史交付文件哈希未变，冻结设计未变。正式有效采集3条；另有初次无日志批次3条、前后预检共7条诊断采集调用。实际PC末端拟合9次；日志修复未新增模型拟合。诊断调用不作为独立数据或正式比较。

'''+f'''九份RF拟合时间合计{fit_total:.2f}秒。初批采集/拟合进程{first['wall_s']:.2f}秒（含哈希与保存），修正重放/复用进程{corrected['wall_s']:.2f}秒；独立训练核对与评分{scores['wall_s']:.2f}秒。不能把修正进程时间单独说成整个实验耗时，也不能将矩阵回放墙钟当成实际求解器获取费用。

'''+'''特征成本按作者提供字段单列于 `FA06_feature_costs.json`：区分train/test/reserved，缺失单元不当作零；属于公共特征计量，未计入主获取预算，不宣称端到端省时。

## 6. 研究含义与收口

FA05-Q的信息不确定性事实仍成立。FA06补上的是条件机制检验：在同一批合法反馈下，权重会改变选择，但当前三个简单处理没有产生达到冻结尺度的平均收益。

本轮不足以支持付费延长运行、学习采集控制或GRL，也不证明真实代价标签没有价值。UNIT与原代理均值接近、上界更差，只限这个随机整实例输入和评价协议；不外推到Pareto、其他模型或场景。

按预案停止扩展本配置：不调惩罚、换折、加种子或延长预算。FA03B“不确定”、FA04和FA05-Q的原结论保持不变。研究领域仍可保留，但“已知偏好的未知原生代价”目前只有动机与信息事实，尚未形成实用缺口或新算法贡献。

## 7. 交付与运行

- `FA06_per_run.csv`、`FA06_per_instance.csv`：九份成绩与1161条评分见证。
- `FA06_scores.json`、`FA06_score_reconciliation.json`：冻结裁决及独立Decimal复算。
- `FA06_training_audit.json`、`FA06_preflight.json`：反馈/训练/投票审计与守卫。
- `FA06_predictions_seal.json`、`FA06_evaluator_freeze.json`：评分前封存与评价程序版本。
- `FA06_logging_repair.json`：新旧训练文件身份与模型复用证书。
- `collections/s{seed}/`、`fits/{arm}_s{seed}/`：有效完整反馈、标签、缓存、模型与预测。

历史输出不改动。首次执行顺序为preflight→run→audit→report；已揭示成绩后run拒绝重跑，audit保留原评分。检查点用于真正的中断恢复，不按结果重新采集。

```powershell
& work/fa00/.venv/Scripts/python.exe work/fa06/preflight.py
& work/fa00/.venv/Scripts/python.exe work/fa06/run.py
& work/fa00/.venv/Scripts/python.exe work/fa06/audit.py
& work/fa00/.venv/Scripts/python.exe work/fa06/report.py
```
'''
    (OUT/'FA06_report.md').write_text(report,encoding='utf-8')
    save('RUN_STATE.json',dict(status='complete',active_jobs=False,actual_terminal_fits=9,
        formal_valid_acquisitions=3,invalid_logging_acquisitions=3,verdict=verdict,grl_started=False))
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='FA06_delivery_manifest.json']
    paths+=list((ROOT/'work/fa06').glob('*.py'))
    save('FA06_delivery_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        design_sha256=sha(ROOT/'outputs/fa06_design/FA06_design_freeze.json')))
    print(json.dumps(dict(verdict=verdict,means={a:float(v) for a,v in means.items()},
                         reconciled_scores=witnesses,fit_s=fit_total,feature_costs=feature_cost)))

if __name__=='__main__':main()
