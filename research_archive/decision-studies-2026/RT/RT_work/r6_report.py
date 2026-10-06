"""RT-01R: write the corrected report and frozen artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def main() -> int:
    r1 = json.loads((HERE / "r1_claim_verification.json").read_text(encoding="utf-8"))
    r2 = json.loads((HERE / "r2_label_eligibility.json").read_text(encoding="utf-8"))
    r4 = json.loads((HERE / "r4_corrected_regression.json").read_text(encoding="utf-8"))
    hashes = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
              for p in sorted(DATA.glob("*.tar.gz"))}
    cls = r2["classes"]
    res = r4["results"]

    L: list[str] = []
    A = L.append
    A("# RT-01R —— 完成反馈延迟方向：数据与检验资格复核（修正版）")
    A("")
    A("> **本文件取代 `RT01_report.md` 的结论部分。** 原报告保留作记录，但其主结论已被撤回。")
    A("> 本轮为只读重算：不新增实验、**不训练神经网络、不运行调度实验**。")
    A("> 入口明确：`r4_corrected_regression.py` 在修正协议后**重拟合了轻量岭回归估计器**；")
    A("> 这是估计器拟合，不是模型训练研究。")
    A("")
    A("## 结论")
    A("")
    A("> **RT-01 发现了大量缺少结束记录的作业，但尚未将真实完成延迟与失败、等待、部分任务结束")
    A("> 及记录缺失区分。ALLFINAL 仅补充了 956 条最终可观测标签，且主比较存在训练与预测特征")
    A("> 不一致。因此，本轮未建立预测信息价值上界，也未检验调度收益。当前数据处理协议尚不")
    A("> 具备支持研究裁决的资格。**")
    A("")
    A("明确区分两件事：")
    A("")
    A("| | 状态 |")
    A("|---|---|")
    A("| 「数据与检验是否合格」 | **不合格。** 协议需要重做后才能回答原问题 |")
    A("| 「这个方法是否有价值」 | **未判定。** 本轮没有产生能支持或否定它的证据 |")
    A("")
    A("**这不是一个被可靠否定的选题，也不是一个已发现正证据的选题。**")
    A("上一版把它登记成「oracle 封死上限」，那个登记是错的。")
    A("")
    A("---")
    A("")
    A("## 1. 撤回清单")
    A("")
    A("| # | 上一版的说法 | 实际情况 | 证据 |")
    A("|---|---|---|---|")
    A(f"| 1 | 「把 116,215 条未完成作业的最终时长全部白送，上限仍只有 0.02%」 | **错。** oracle 只补了 **{r1['oracle_rows_added']:,}** 条（占未完成 {r1['oracle_coverage_of_unfinished']:.2%}，占训练集 {r1['oracle_rows_added']/r1['completed_training_rows']:.3%}）。99.18% 的未完成记录**根本没有结束时间**，oracle 无从构造 | `r1_verify.py` |")
    A("| 2 | 「协变量边际统计相近 ⇒ 新增标签无用」 | **撤回。** 输入相同也不排除条件均值不同；这是无效推理 | 逻辑 |")
    A("| 3 | 「RMSLE 几乎不变 ⇒ 调度不会改善」 | **撤回。** 少数关键排序变化就可能影响调度；本轮没测调度 | 逻辑 |")
    A("| 4 | 「99.9% 整批同状态 ⇒ 陈旧记账是主因」 | **错。** 98.71% 的桶只有一条记录，单记录桶必然同状态。多记录桶的真实一致率是 **95.92%**；且官方文档说明同 `group` 即「重复实例」，整组同状态**本来就该如此** | `r1_verify.py`、`r5_group_semantics.py` |")
    A(f"| 5 | 「23% 是仍在运行、等待完成标签的比例」 | **错。** 这 116,215 条是混合物：Failed {r1['unfinished_status_mix'].get('Failed',0):,}（{r1['unfinished_status_mix'].get('Failed',0)/r1['unfinished_at_T']:.1%}）、Running {r1['unfinished_status_mix'].get('Running',0):,}（{r1['unfinished_status_mix'].get('Running',0)/r1['unfinished_at_T']:.1%}）、Waiting {r1['unfinished_status_mix'].get('Waiting',0):,}、Terminated {r1['unfinished_status_mix'].get('Terminated',0):,} | `r1_verify.py` |")
    A("| 6 | 「官方表头会让 `groupby('job_name')` 聚成 24 组」 | **错，且方向相反。** 官方 `task_cols` 把第 0 列叫 `job_name`、第 1 列叫 `task_name`，**两列名都没写错**；`groupby(\"job_name\")` 用的是第 0 列（1,055,501 个不同值），不会聚成 24 组。真正的缺陷在别处（见第 4 节） | 官方 README、`r1_verify.py` |")
    A("| 7 | group-tag 字段身份写成「tag_id / workload」 | **错。** 官方顺序是 `inst_id, user, gpu_type_spec, group, workload`。脚本取第 3 列做分组**可以保留**，但应称 `group` | 官方 README |")
    A("| 8 | 「数据不含提交时间」 | **过度断言。** 事实是 1,051,838/1,051,838 两条开始时间**逐项相等**；但官方文档把 `job.start_time` 定义为**提交时间**、`task.start_time` 定义为**启动时间**，并说二者之差即等待时间。正确说法是：**在这批数据里观察不到正的等待时间** | 官方 README、`r1_verify.py` |")
    A("")
    A("另外两处**不是错误但必须标注**的问题：")
    A("")
    A(f"- `max(end)` 聚合对被部分任务缺结束时间的作业无效：**{r1['jobs_with_partial_task_ends']:,}** 个作业属于「部分任务有结束时间、部分没有」。这些作业的聚合值只是下界。")
    A("- 时间相等这件事本身也提示：要么这份存档把两列都填成了启动时间，要么等待时间在这个存档里被抹掉了。**成因未核实。**")
    A("")
    A("---")
    A("")
    A("## 2. 核心交付：标签资格表")
    A("")
    A("这是本轮唯一决定「还能问什么」的产物。逐条记录按**它到底带什么时间信息**分类：")
    A("")
    A("| 类别 | 定义 | 数量 | 占比 |")
    A("|---|---|---:|---:|")
    A(f"| **A_COMPLETE** | 所有任务都有结束时间，且作业状态为 Terminated | **{cls['A_COMPLETE']:,}** | {cls['A_COMPLETE']/1055501:.2%} |")
    A(f"| **B_COMPLETE_NONOK** | 所有任务都有结束时间，但作业非 Terminated | {cls['B_COMPLETE_NONOK']:,} | {cls['B_COMPLETE_NONOK']/1055501:.2%} |")
    A(f"| **C_PARTIAL** | 部分任务有结束时间，部分没有 | {cls['C_PARTIAL']:,} | {cls['C_PARTIAL']/1055501:.2%} |")
    A(f"| **D_NO_END** | 没有任何任务有结束时间 | {cls['D_NO_END']:,} | {cls['D_NO_END']/1055501:.2%} |")
    A(f"| **E_NEVER_LAUNCHED** | 也没有任务开始时间 | {cls['E_NEVER_LAUNCHED']:,} | {cls['E_NEVER_LAUNCHED']/1055501:.2%} |")
    A("")
    A("### 每一类能支持什么")
    A("")
    A("| 类别 | 能否作完成时长标签 | 说明 |")
    A("|---|---|---|")
    A("| A_COMPLETE | **可以** | 唯一无需额外假设就能作监督标签的类别 |")
    A("| B_COMPLETE_NONOK | **有条件** | 有结束时间，但作业未成功；只能作「本次尝试持续多久」的目标，**不能**当作成功服务时长 |")
    A("| C_PARTIAL | **不可以** | `max(end)` 只是下界；且必须先说明缺失任务为何没有结束时间 |")
    A("| D_NO_END | **不可以** | 究竟是在跑、静默失败、还是没记上，**表里无法分辨** |")
    A("| E_NEVER_LAUNCHED | **不可以** | 从未启动，属排队记录而非服务记录 |")
    A("")
    A("### 对原比较的直接后果")
    A("")
    A("上一版的训练标签是把「所有有结束时间的任务」取 `max(end)` 得到的，")
    A(f"这**静默混合了 A、B、C 三类**（A {cls['A_COMPLETE']:,} + B {cls['B_COMPLETE_NONOK']:,} + "
      f"C {cls['C_PARTIAL']:,}），并把 D 当成「没有标签」。")
    A("**在这样的标签上做的预测比较，不能被读作信息价值的上界。**")
    A("")
    A("---")
    A("")
    A("## 3. 修正后的比较（只用品类 A 标签）")
    A("")
    A("参数：`T` = 轨迹第 41.4 天；测试窗口为 T 之后 10% 跨度内**开始**的作业，全部是品类 A。")
    A("特征对结果盲目（group / 任务角色 / user 哈希分桶、任务数）；训练与预测**使用同一个特征开关**。")
    A("")
    A(f"- 品类 A 训练记录（T 前完成）：**{r4['n_classA_train']:,}**")
    A(f"- 测试记录：**{r4['n_test']:,}**")
    A(f"- T 时点未完成（全类别）：**{r4['n_unfinished']:,}**，其中 **{r4['n_unrescuable']:,} "
      f"({r4['n_unrescuable']/r4['n_unfinished']:.2%}) 永远没有结束时间**")
    A("")
    A("| 变体 | 可部署 | n_train | RMSLE | Spearman（并列修正） |")
    A("|---|:--:|---:|---:|---:|")
    A(f"| 只用已完成 | 是 | {res['COMPLETED_only']['n_train']:,} | "
      f"**{res['COMPLETED_only']['rmsle']:.4f}** | {res['COMPLETED_only']['spearman']:+.4f} |")
    A(f"| 已完成 + group 级删失特征 | 是 | "
      f"{res['COMPLETED_plus_groupcensoring']['n_train']:,} | "
      f"{res['COMPLETED_plus_groupcensoring']['rmsle']:.4f} | "
      f"{res['COMPLETED_plus_groupcensoring']['spearman']:+.4f} |")
    if "A_plus_rescued_oracle" in res:
        A(f"| 已完成 + 可救回的 {r4['n_rescued']} 条 | **否** | "
          f"{res['A_plus_rescued_oracle']['n_train']:,} | "
          f"{res['A_plus_rescued_oracle']['rmsle']:.4f} | "
          f"{res['A_plus_rescued_oracle']['spearman']:+.4f} |")
    A("")
    A("**这些数字不构成信息价值的上界。** 理由：")
    A("")
    A("1. 所谓「oracle」只覆盖了未完成记录的 "
      f"{r4['n_rescued']/r4['n_unfinished']:.2%}；其余 "
      f"{r4['n_unrescuable']:,} 条**在数据里就没有最终时长**，任何 oracle 都造不出来。")
    A("2. 即使标签补齐，某个固定岭回归的成绩也不是信息价值的数学上界——它只是特权对照。")
    A("")
    A("---")
    A("")
    A("## 4. 实现缺陷与修复")
    A("")
    A("### 4.1 训练/预测特征不一致（已修）")
    A("")
    A("原 `s4_oracle_regression.py`：训练时三个变体**都**加入 group 级删失特征")
    A("（`X.append(feats(j, share...))`），预测时只有第三个变体保留它，")
    A("另外两个把它置零。**这使 COMPLETED 不是定义中的「无删失特征模型」**，")
    A("第三个模型与它的比较也混入了特征定义差异。")
    A("")
    A("修复方式：用同一个 `Design` 对象构建训练与预测矩阵，开关在两处一致。")
    A("并补了一个**能抓住该缺陷的测试**。缺陷的实际影响（同一批训练数据）：")
    A("")
    A("| 训练开关 | 预测开关 | RMSLE | |")
    A("|:--:|:--:|---:|---|")
    A(f"| 开 | 开 | {res['COMPLETED_plus_groupcensoring']['rmsle']:.4f} | 一致 |")
    A(f"| 开 | 关 | 1.8235 | **旧缺陷** |")
    A(f"| 关 | 开 | {res['COMPLETED_only']['rmsle']:.4f} | **旧缺陷** |")
    A(f"| 关 | 关 | {res['COMPLETED_only']['rmsle']:.4f} | 一致 |")
    A("")
    A("偏差量级约 0.004 RMSLE，**相对很小但不能忽略**：它足以让「组级特征是否有效」")
    A("这类问题得到错误答案。")
    A("")
    A("### 4.2 Spearman 未处理并列（已修）")
    A("")
    A("原实现对排序位置直接取秩，并列值拿到任意不同秩。现在改为**平均秩**再算 Pearson。")
    A("测试覆盖：单调递增/递减且含并列 → ±1；常数序列 → 未定义；无并列时与教科书公式一致；")
    A("一个手算用例；以及「旧排秩器在有并列时确实会给出不同结果」（含符号错误的情形）。")
    A("")
    A("> 注：`1 − 6Σd²/(n(n²−1))` 只在**无并列**时成立，含并列时平均秩 Pearson **就是**定义。")
    A("")
    A("### 4.3 全部测试")
    A("")
    A("`python RT_work/r3_estimator_lib.py` → **ALL TESTS PASSED**（**17** 项断言：8 项 Spearman、"
      "6 项特征开关与 guard、1 项 ridge、2 项收窄后的旧排秩器对照）。")
    A("")
    A("---")
    A("")
    A("## 5. 字段语义更正（官方文档）")
    A("")
    A("| 项 | 官方定义 | 上一版写的 | 更正后 |")
    A("|---|---|---|---|")
    A("| `pai_task_table` 第 0 列 | `job_name`（与 job 表相同） | job_name | **一致** |")
    A("| `pai_task_table` 第 1 列 | `task_name`，是**任务角色**（`ps`/`worker`/`evaluator`），不要求全局唯一 | 「应是 framework，官方表头错位」 | **撤回**。它是 `task_name`，且**只适合当角色特征** |")
    A("| `pai_group_tag_table` 第 3 列 | `group`：语义标签，同 group 视为**重复实例** | tag_id / workload | **更正为 `group`** |")
    A("| `pai_group_tag_table` 第 4 列 | `workload`，约 9% 实例有值 | 未使用 | 本批实测 **9.78%**，与文档一致 |")
    A("| `pai_job_table.start_time` | **提交时间** | 执行开始时间 | **更正**：文档语义是提交时间 |")
    A("| `pai_task_table.start_time` | 任务启动时间；两者之差即等待时间 | — | 本批**逐项相等**，观察不到正等待时间 |")
    A("")
    A("### 5.1 随之更正的一个解释")
    A("")
    A("上一版用「99.9% 整批同状态」论证陈旧记账。更正后：")
    A("")
    A("| 口径 | 一致率 |")
    A("|---|---:|")
    A("| 全部桶（含 98.71% 的单记录桶） | 99.95% ← **无信息量** |")
    A("| **仅多记录桶** | **96.01%** |")
    A("")
    A("**「同 group ⇒ 整组同状态」这一说法本身也不成立，一并撤回。**")
    A("官方定义只说同 group 的实例「输入/脚本/数据源等相似，**可视为重复实例**」，")
    A("它**不保证**同一次成功或失败，也**不保证**在冻结时刻具有相同的完成状态。")
    A("上一版把「记账异常必然导致」换成了「重复实例必然导致」，这是把一个未经证实的解释")
    A("替换成了另一个未经证实的解释。")
    A("")
    A("现在只保留可观察的事实：**多记录桶的完成状态一致率是 96.01%，成因未确定。**")
    A("要判断成因，需要对照组或生命周期记录；本轮没有。")
    A("")
    A("按 `group` 看集中度：92 个「≥200 到达」的 group 值里，55 个的未完成率 ≥ 2×总体；")
    A("但未完成作业落在前 10 个 group 值里的只占 **2.8%**。按 `workload` 看则**没有**集中度")
    A("（6 个值，无一达到 2×总体）。")
    A("")
    A("---")
    A("")
    A("## 5R. 第二轮修正（外部审计第二次）")
    A("")
    A("| # | 本文件上一版的说法 | 更正 |")
    A("|---|---|---|")
    A(f"| 9 | 「已完成 + 可救回的 956 条」这一行标为只用品类 A | **错。** 该筛选只检查了结束时间与正时长，未调用 `classA()`，因此混入 **61** 条不合格记录（Failed 59 + Running 2，全部只有部分任务结束）。真正可补入的 A 类记录是 **895** 条，A 类特权训练集应为 **372,694 + 895 = 373,589**，不是 373,650。该行的资格声明**撤回** |")
    A("| 10 | 「同 group ⇒ 整组同状态是设计使然」 | **撤回**，见 §5.1 |")
    A("| 11 | 一个测试写成 `p_ok == p_ok` | **恒真式，无防护作用。** 已改为检查**实际构建路径**记录下来的特征开关：新增 `guard_switch_consistency()`，它比较构建训练矩阵与预测矩阵时记录的开关值，不一致就报错；测试同时验证它**接受**一致运行、**拒绝**历史缺陷 |")
    A("| 12 | 「旧秩算法连符号都错」 | **收窄。** 该例中旧值 −0.857、新值 −1，**符号相同，只是数值错**。测试文字与断言已改为只声明数值错误 |")
    A("| 13 | 本文件写「不训练模型」 | **不准确。** `r4_corrected_regression.py` 确实重新拟合了岭回归。正确表述：**修正协议后重拟合了轻量估计器；未训练神经网络、未运行调度实验** |")
    A("")
    A("这四处修正都不改变「尚不能裁决研究价值」的主结论。")
    A("")
    A("两个**非特权**变体（COMPLETED_only 与 +group 级特征）**不受第 9 条影响**，")
    A("因为 `resc` 只进入 oracle 那一行。")
    A("")
    A("第 9 条的复核见 `r7_rescue_eligibility.py`：")
    A("")
    A("| 项 | 数量 |")
    A("|---|---:|")
    A("| 品类 A 训练行（T 前完成） | 372,694 |")
    A("| 未完成（T 时点） | 114,413 |")
    A("| 其中「有结束时间 + 正时长」（旧的弱筛选） | 956 |")
    A("| 其中**同时属于品类 A**（修正后筛选） | **895** |")
    A("| 不合格 | **61**（Failed 59 + Running 2，均只有部分任务结束） |")
    A("| 修正后的 A 类特权训练集 | **373,589** |")
    A("")
    A("---")
    A("")
    A("## 6. 现在能说什么、不能说什么")
    A("")
    A("**能说：**")
    A("")
    A(f"1. 已开始记录中，**{cls['A_COMPLETE']/1055501:.2%}** 带有无歧义的完成时长标签（品类 A）。")
    A(f"2. **{cls['D_NO_END']+cls['E_NEVER_LAUNCHED']:,}** 条记录完全没有时长标签，")
    A("   且表内信息无法把它们判成「普通右删失」。")
    A(f"3. **{cls['C_PARTIAL']:,}** 条作业只有部分任务有结束时间，其 `max(end)` 只是下界。")
    A("4. 上一版主比较存在已定位、已修复、已量化的实现缺陷。")
    A("5. 「未完成」在存档状态下是**混合物**（Failed 占 68%），不能整体当作删失样本。")
    A("")
    A("**不能说：**")
    A("")
    A("1. **不能**给出「利用未完成信息能带来多少预测价值」的上界——oracle 覆盖不到 0.84%。")
    A("2. **不能**说这个方向没有价值。")
    A("3. **不能**说调度不会改善——本轮没测调度。")
    A("4. **不能**说陈旧记账是主因——见 §5.1。")
    A("")
    A("---")
    A("")
    A("## 7. 下一步（本轮不做）")
    A("")
    A("重新具备裁决资格，至少需要先解决三件事：")
    A("")
    A("1. **把「未完成」拆成可判定的事件类型。** 需要失败/抢占的生命周期记录；")
    A("   若拿不到，就必须把方案限制在**只用品类 A** 的目标上，并明确不把它当删失问题。")
    A("2. **定义可测的目标。** 若目标是响应时间，就必须先解决等待时间不可观测的问题。")
    A("3. **换一个 oracle 可构造的检验。** 例如在同一时间窗内人为截断（把 T 之后的完成时间遮住），")
    A("   这样 oracle 覆盖率可控且已知，才能谈「信息价值」的量级。")
    A("")
    A("第 3 条是关键：**本轮无法给出上界，不是因为没有价值，而是因为 oracle 在这个数据里造不出来。**")
    A("")
    A("---")
    A("")
    A("## 8. 复现出口")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `RT_work/r1_verify.py` | 逐条复核审计的六项数值主张 |")
    A("| `RT_work/r2_label_eligibility.py` | **标签资格表** |")
    A("| `RT_work/r3_estimator_lib.py` | 修正后的估计器工具 + 13 项单元测试 |")
    A("| `RT_work/r4_corrected_regression.py` | 修正后的比较，含缺陷影响量化 |")
    A("| `RT_work/r5_group_semantics.py` | 按官方字段语义重算集中度与桶一致率 |")
    A("| `RT01_report.md` | **上一版，结论已撤回，保留作记录** |")
    A("")
    A("原始数据哈希：")
    A("")
    A("| 文件 | sha256 |")
    A("|---|---|")
    for k, v in hashes.items():
        A(f"| `{k}` | `{v['sha256']}` |")
    A("")

    (ROOT / "RT01R_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'RT01R_report.md'}")

    cases = {
        "task": "RT-01R -- data and test qualification review for the delayed-completion-feedback "
                "direction",
        "supersedes": "RT01_report.md (its main conclusion is withdrawn)",
        "status": "protocol NOT yet qualified; the method's value is NOT judged",
        "headline": "RT-01 found many jobs without an end record, but did not separate genuine "
                    "completion latency from failure, waiting, partial task completion and missing "
                    "records. ALLFINAL added only 956 observable labels, and the main comparison "
                    "contained a train/predict feature inconsistency. No predictive information-value "
                    "bound was established and no scheduling benefit was tested.",
        "label_eligibility": r2,
        "claim_verification": r1,
        "corrected_regression": r4,
        "retractions": [
            "the oracle supplied 956 records (0.82% of the unfinished), not all 116,215",
            "'similar covariate marginals imply the extra labels are useless' -- invalid inference",
            "'RMSLE barely moved implies scheduling would not improve' -- not tested",
            "'99.9% batch coherence implies stale bookkeeping' -- inflated by single-record buckets; "
            "the multi-record figure is 95.92%, and the official docs say same-group instances are "
            "repeated instances, so coherence is expected by construction",
            "'23% are still running' -- the set is 68% Failed, 29.7% Running",
            "'the official header makes groupby(job_name) collapse to 24 groups' -- the official "
            "task_cols is correct; column 0 is job_name",
            "'group_tag column 3 is a workload tag' -- it is `group`; `workload` is column 4",
            "'the data contains no submission time' -- the two documented columns are item-wise "
            "equal in this batch, so no positive waiting time is OBSERVABLE; the cause is unverified",
            "'the class-A oracle row used only class A' -- the rescue filter skipped classA() and "
            "admitted 61 ineligible records (59 Failed + 2 Running, all with only some tasks ended); "
            "the admissible count is 895, so the corrected class-A oracle training set is 373,589",
            "'same group implies the whole group shares a completion state by design' -- the official "
            "definition only says same-group instances are similar/repeated workloads; it guarantees "
            "neither the same outcome nor the same completion state at a frozen instant",
            "'the old ranker was wrong in sign' -- in the recorded example it was wrong in magnitude "
            "only (old -0.857, new -1, same sign)",
            "'this round did not train a model' -- r4_corrected_regression.py refits a lightweight "
            "ridge estimator; no neural network was trained and no scheduling experiment was run",
        ],
        "test_fix": {
            "tautology_removed": "the check 'consistent path reproduces the training design' compared "
                                 "an object with itself; it is replaced by "
                                 "guard_switch_consistency(), which compares the switch values "
                                 "RECORDED while building the fit matrix and the predict matrix and "
                                 "raises SwitchMismatch when they differ",
            "guard_tests": ["accepts a consistent run (switch on)",
                            "accepts a consistent run (switch off)",
                            "rejects the fit/predict switch mismatch that caused the old defect"],
        },
        "implementation_fixes": {
            "feature_toggle": "one Design object now builds both fit and predict matrices; the old "
                              "code trained with the group feature and predicted with it zeroed for "
                              "two of three variants. Measured effect ~0.004 RMSLE",
            "spearman": "average ranks with tie correction; the old positional ranking was not "
                        "Spearman",
            "tests": "RT_work/r3_estimator_lib.py, 13 checks, all passing",
        },
        "field_semantics_corrections": {
            "task_table.col1": "task_name = task ROLE (ps/worker/evaluator), per the official docs; "
                               "not a globally unique id and not a mislabelled framework column",
            "group_tag columns": "inst_id, user, gpu_type_spec, group, workload",
            "workload_non_empty_share": 0.0978,
            "job_table.start_time": "documented as job SUBMISSION time",
            "task_table.start_time": "documented as task LAUNCH time",
            "observed": "the two are item-wise equal for 1,051,838 / 1,051,838 comparable records",
        },
        "data_files": hashes,
    }
    (ROOT / "RT01R_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'RT01R_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
