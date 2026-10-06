"""Freeze FA06 public metadata and design. No outcome loading or model fitting."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/fa06_design'
CV=ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO/cv.arff'
PROTOCOL=ROOT/'outputs/fa00/FA00_protocol.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads(PROTOCOL.read_text())
    ids=previous['ids'];algorithms=previous['algorithms']
    text=CV.read_text(encoding='utf-8')
    data=text[text.lower().index('@data')+5:]
    cv={}
    for row in csv.reader(data.splitlines()):
        if not row or row[0].lstrip().startswith('%'):continue
        name,rep,fold=row
        assert int(rep)==1 and name not in cv
        cv[name]=int(fold)
    assert set(cv)==set(ids) and len(ids)==1294 and set(cv.values())==set(range(1,11))
    test=np.array([i for i,x in enumerate(ids) if cv[x]==2],dtype=int)
    pool=np.array([i for i,x in enumerate(ids) if cv[x]!=2],dtype=int)
    order=np.random.default_rng(20261001).permutation(pool)
    n_valid=math.ceil(.1*len(pool))
    reserve=np.sort(order[:n_valid]);train=np.sort(order[n_valid:])
    assert not(set(train)&set(reserve) or set(train)&set(test) or set(reserve)&set(test))
    assert len(test)==129 and len(train)==1048 and len(reserve)==117
    overlap={k:len(set(previous[k])&set(test)) for k in ['train','validation','test']}
    seeds=[7,42,99]
    init={str(s):np.random.default_rng(s).choice(train,20,replace=False).tolist() for s in seeds}
    assert all(not(set(v)&set(test)) for v in init.values())
    files=[PROTOCOL,CV,ROOT/'work/fa00/core.py',ROOT/'work/fa00/prepare.py',
           ROOT/'work/fa03b/policy.py',ROOT/'work/fa00/upstream/ActiveRFModel.py',
           ROOT/'outputs/fa05_q/FA05_report.md',ROOT/'outputs/fa05_q/FA05_independent_audit.json',
           ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO/feature_values.arff',
           ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO/algorithm_runs.arff']
    inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    computed_budget=.25*.05*len(train)*len(algorithms)*600
    assert math.isclose(computed_budget,86460,rel_tol=0,abs_tol=1e-8)
    budget=86460.0  # canonical exact decimal boundary, avoiding formula roundoff
    design=dict(stage='FA06',date='2026-10-05',status='design_frozen_not_executed',
       question='Conditional effect of terminal pairwise sample weights under a shared legal-feedback acquisition cache',
       dataset='ASP-POTASSCO',source_commit=previous['source_commit'],
       ids=ids,algorithms=algorithms,official_repetition=1,test_fold=2,
       train=train.tolist(),reserved_unused=reserve.tolist(),test=test.tolist(),
       split_seed=20261001,initial_rows=init,seeds=seeds,
       test_overlap_with_old_splits=overlap,
       evaluation_scope='New split development mechanism test; not new data, unseen instances, or independent external confirmation',
       acquisition='Three fresh RANDOM-ROW caches; no intermediate model fit or model-driven acquisition',
       initial='20 random training instances, all algorithms in canonical order; charged',
       batch_instances=11,cap_s=100,native_cutoff_s=600,native_timeout_score=6000,
       acquisition_budget_s=budget,budget_formula='.25 * .05 * N_train * N_algorithms * 600',
       stop='budget exhausted or fixed-cap legal pool exhausted; truncate final dispatch before revealing feedback',
       execution='restart; max lower bound; no repeat at same/smaller cap; cached outcomes cost zero',
       labels='all legally identifiable terminal pair labels; common positive original-weight support; ties excluded for all arms',
       weights={
           'ORIGINAL':'both complete: abs(va-vb); one complete v, premature censor l: abs(10*l-v)',
           'UNIT':'1 on the same common positive-weight, non-tie support',
           'NATIVE-UPPER':'both complete: abs(va-vb); one complete v, premature censor l: 6000-v; native timeout: exact 6000-v'},
       native_upper_interpretation='Maximum compatible native regret, pessimistic surrogate; premature censor not asserted to be native timeout',
       model=dict(n_estimators=100,max_features='sqrt',max_depth=2**31,min_samples_split=2,bootstrap=True,n_jobs=1),
       model_rng="per pair keyed_seed(seed,'pair','FA06-terminal',pair_index), identical across weights",
       preprocessing='Rebuild train-only non-all-missing feature selection, median imputation and scaling; never reuse old preprocessing',
       selection='55 pairwise hard classifiers; same canonical pair order and original first-encounter vote tie break; no cost regressor needed for terminal vote',
       fits=9,weight_rules=3,acquisition_caches=3,
       test_scoring='once, after all caches/labels/model-inputs/predictions are sealed; all 129 evaluation rows retained',
       primary='ratio of arithmetic seed means of native test PAR10; per seed directions also reported',
       diagnostics=['test algorithm-choice disagreements','native timeout counts','per-pair weight class composition','actual collection and fit wall time'],
       no_sweeps=True,no_new_learning_architecture=True,no_grl=True,
       gates=dict(simple_rule_signal='UNIT or NATIVE-UPPER mean >=5% below ORIGINAL and improves >=2/3 seeds',
          original_robust='ORIGINAL mean >=5% below both alternatives and improves >=2/3 seeds vs each',
          close='both alternatives lie within +/-2% of ORIGINAL mean',
          otherwise='undetermined; no extra folds/seeds/rules/budgets'),
       gate_scope='Investment screening only, not significance/equivalence/novel algorithm evidence',
       priority='protocol validity first, then simple-rule signal, then original robust, then close, otherwise undetermined',
       wall_limit_s=7200,engineering_limit='half day; retain checkpoints, no silent downscaling or replacing partial outcomes',
       input_hashes=inputs,
       semantic_reads=[PROTOCOL.relative_to(ROOT).as_posix(),CV.relative_to(ROOT).as_posix()],
       raw_outcome_handling='algorithm_runs.arff only hashed as bytes at design stage; not parsed; no collection/fits/scoring started')
    save('FA06_design_freeze.json',design)
    with (OUT/'FA06_split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['row','instance_id','split','old_split'])
        old={i:k for k in ['train','validation','test'] for i in previous[k]}
        tr=set(train);te=set(test)
        for i,name in enumerate(ids):w.writerow([i,name,'train' if i in tr else 'test' if i in te else 'reserved_unused',old[i]])
    spec='''# FA06：相同付费反馈下，末端权重是否影响算法选择

日期：2026-10-05。状态：规格与公开划分已冻结；未启动采集、拟合或评分。

## 研究问题与边界

本轮只问：**同一预算取得的合法偏好标签，使用原代理、单位权重或原生代价上界代理时，最终选择成绩是否有实用差异？**

FA05-Q确认了“胜负已知、原生代价未知”，没有确认权重近似造成决策损失。这里检验一种廉价处理的条件效果，不研究新采样器、预算控制、生存模型或GRL，不把权重替代称为新算法贡献。

## 1. 为什么需重新采集

旧官方fold1测试已多次用于开发。本轮固定官方repetition1的fold2作评价，共129行；其余1165行以20261001固定随机种子留出117行且完全不用，1048行训练。总获取预算仍为86460模拟CPU秒。

fold2中有120行属于旧训练集合、9行属于旧验证集合；没有核实它们是否全部被实际查询。**这不是新数据或从未见过的实例，不能称独立确认。** 它只是预先冻结的新切分上的开发性机制检验。旧fold1结果已经影响选题设计，这一边界同样保留。

不能把旧训练日志删掉fold2的最终标签后复用：其反馈或费用曾进入采样模型、后续选择和预算前缀，删标签不会逆转这种影响。因此本轮从空缓存重新采集；fold2与117个保留实例均不参与采集、预处理拟合、标签推导或模型拟合。初始化重新抽取，预处理重新按新训练集合计算，旧归一化矩阵不能直接复用。

## 2. 三份共享采集缓存

- 种子固定7、42、99，每种子一份新缓存。
- 20个训练实例初始化，运行全部11个算法，初始费用计入预算。
- 随后沿用随机整实例规则：每批11个仍可查询实例，查询其全部合法算法；不按模型分数选例。
- 截止100秒，从头重跑语义，预算尾部先截短再生成反馈。完成判定使用ok且runtime≤实际截止，未完成只给下界。
- 下界保持最大值；完成可缓存；不重复查询相同或更小下界。预算用尽或合法池耗尽即结束，不人为补满样本。
- 不做中间模型拟合。随机行选择不依赖模型；本轮质量比较只需要末端训练，不能把省去中间拟合解释为新采样器的端到端优势。

三权重共享**该种子新取得的缓存和标签**，不是沿用历史已付费记录。各权重的离线采集成本相同，一份缓存只计费一次。原生成绩只由回放环境读取以产生反馈；训练器和策略只收到合法反馈。

## 3. 仅三种权重，共九次末端拟合

各臂使用相同正权重非平局支持、标签、训练特征、RF参数、逐成对模型的随机种子和投票规则。胜负已经合法确定的标签不被重新赋方向。双方完成相等的零权重平局对所有臂都排除。

| 权重臂 | 双方完成 | 一方完成v、另一方提前删失于l |
|---|---|---|
| ORIGINAL | `abs(va−vb)` | `abs(10*l−v)` |
| UNIT | 1 | 1 |
| NATIVE-UPPER | `abs(va−vb)` | `6000−v` |

NATIVE-UPPER采用公开T=600的原生PAR10代价可能集合最大值，**是悲观替代权重，不是已知真实代价，也不声称删失方真的超时**。若确认原生超时，该式才是精确代价；100秒采集通常不会给出这种确认。例v20/l100，三臂分别980、1、5980。

不加入下界或更多惩罚常数，不扫描倍率。NATIVE-UPPER仍是点替代，不能代替成熟生存分布或区间损失方法。

RF保持100树、max_features=sqrt、max_depth=2**31、min_samples_split=2、bootstrap=True、单线程。每对模型用 `keyed_seed(seed,'pair','FA06-terminal',pair_index)`，三个权重臂相同。55个成对分类器按固定顺序硬投票，沿用首次出现票的破并列规则。末端预测不使用成本回归器，因此不额外拟合11个采集成本模型。

这里移除了中间拟合、只比较末端权重，是一个新的受控机制协议；不能直接拿成绩与FA02/FA04作方法排序。它也不回答在Pareto或粗粒度PC输入上是否有同样效果。

## 4. 评价与计量

所有缓存、标签、权重、训练输入、模型与129行预测先落盘并封存，再由独立评价过程一次读取原生评价成绩。保留所有评价行，不按难度、超时或可解性删除实例。不读取保留验证成绩，不挑checkpoint。

主指标为各权重臂三个种子的原生平均测试PAR10；改善按种子均值之比计算。例如UNIT相对ORIGINAL为 `1−mean(P_UNIT)/mean(P_ORIGINAL)`。另列每种子方向、实际选择分歧、原生超时次数及差值。选择分歧只是诊断，不当作质量收益。

三种子共用一个评价集合，不是三个独立数据集；不据此给显著性或泛化保证。完整采集与拟合计算耗时单列，特征获取成本依旧单列；主预算是求解器反馈获取CPU秒，不能直接声称端到端省时。

## 5. 投资筛查规则（结果前冻结）

先检查协议资格，再按以下顺序判断：

1. UNIT或NATIVE-UPPER相对ORIGINAL均值至少好5%，且至少2/3种子同向：**接受该廉价规则的开发信号**。还没有新算法贡献，更不能跳到付费解除删失。
2. ORIGINAL相对两个替代臂均值均至少好5%，各至少2/3种子同向：当前代理在本机制协议中有稳健性信号，降低围绕权重近似开发复杂方法的优先级。
3. 两个替代臂与ORIGINAL的均值差均在±2%内：本协议未呈现实用平均敏感性，收口这个权重处理探针；**这不是等价检验**。
4. 其他结果记不确定，不追加折、种子、权重或预算找赢家。

如果存在协议不合格、未完成拟合或账目错误，先报告未判定，不套用成绩门槛。若两个廉价规则都达到第一条，分别报告，不事后挑一个宣称胜过所有方法。

即使权重显著改变选择，也可能不改变PAR10；即使PAR10改善，也仅证明一个便宜替代在本协议中有效。现有未知尾部可被多个真实运行分布解释，不能把NATIVE-UPPER当作恢复真实尾部标签的证据。

## 6. 必须通过的检查

- 新train、reserved、test互不相交；初始化、每次反馈和训练行全部属于新train。
- 新预处理只在新train拟合；输入签名不使用旧预处理、模型或预测文件。
- 单个v20/l100手算权重为980/1/5980；双方完成和平局、预算尾部和方向相反的例子也检查。
- 每种子三臂支持/标签/特征/模型随机种子一致，差别只在sample_weight；完整支持摘要哈希落盘。
- 替换fold2隐藏运行时长与状态，真实采集runner应得到相同的付费轨迹和训练输入。预测阶段输入签名也须相同；不要用手写常量冒充该检查。
- 独立从反馈重算收费、标签资格、权重与末端投票、原生评分；披露是否独立重新拟合RF，不默认宣称已经重拟合。
- 所有历史交付文件保持不变；新输出放 `outputs/fa06/`。

实际计算上限两小时，工程接入半天。逐完成执行保存检查点，禁止中断后丢弃已付费前缀重开；超上限保留未完成状态，不缩模型或换数据。

## 7. 文献定位

删失建模和按决策损失选择已有Run2Survive；用区间训练已有Superset Learning；预算内付费解除删失也已有直接工作。本轮三个点权重只是便宜对照，不是对这些成熟方法的完整比较。

- [Run2Survive](https://proceedings.mlr.press/v129/tornede20a.html)
- [Superset Learning](https://epub.ub.uni-muenchen.de/91670/1/_PAKDD__Superset_Learning_for_Algorithm_Selection_with_Right_Censored_Data.pdf)
- [预算受限解除删失预印本](https://arxiv.org/abs/2510.12144)

若第一档出现，后续才单独冻结成熟删失对照和新的确认数据；不能把换惩罚常数包装为CCF B贡献。若其他档出现，同样不启动GRL。

## 本轮交付状态

`FA06_design_freeze.json`保存完整划分、初始化、权重、预算、判据与来源哈希；`FA06_split_manifest.csv`保存每行的新旧归属。设计阶段语义读取仅为公开划分和旧协议；原生成绩文件只作字节哈希，未解析。

**当前仅冻结规格，尚未编写或启动正式采集、九次拟合和评分。**
'''
    (OUT/'FA06_next_step.md').write_text(spec,encoding='utf-8')
    save('FA06_design_checks.json',dict(status='passed',train=len(train),reserved=len(reserve),test=len(test),
        native_cutoff_s=600,budget_s=budget,old_test_overlap=overlap,new_acquisitions=0,new_fits=0,
        semantics='metadata qualification only; future collector and training runner not yet implemented'))
    print(json.dumps(dict(train=len(train),reserved=len(reserve),test=len(test),budget=budget,overlap=overlap,new_acquisitions=0,new_fits=0)))

if __name__=='__main__':main()
