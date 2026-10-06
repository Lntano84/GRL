"""Write the RT qualification report and the frozen artifacts."""
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
    meta = json.loads((HERE / "s2_meta.json").read_text(encoding="utf-8")) \
        if (HERE / "s2_meta.json").exists() else {}
    s4 = json.loads((HERE / "s4_oracle_regression.json").read_text(encoding="utf-8"))
    s6 = json.loads((HERE / "s6_targeted.json").read_text(encoding="utf-8"))

    hashes = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
              for p in sorted(DATA.glob("*.tar.gz"))}

    L: list[str] = []
    A = L.append
    A("# RT-01 —— 完成反馈延迟方向的资格检查（第一轮）")
    A("")
    A("## 一页结论")
    A("")
    A("**信息结构真实存在，但可回收价值接近于零。按你预先写下的三条反证路径，这个方向应当**")
    A("**在这一数据上停止，而不是进入调度收益验证。**")
    A("")
    A("三句话：")
    A("")
    A("1. **反馈结构确实存在且可测。** 在冻结时点，约 **23% 的已启动作业没有完成标签**；")
    A("   只用已完成作业的估计器会系统性偏低。这是一个真实的删失结构，不是我们编造的场景。")
    A("2. **但即使把未完成作业的最终时长全部白送给我们，测试集预测也几乎不动**")
    A(f"   （RMSLE {s4['COMPLETED']['rmsle']:.4f} → {s4['ALLFINAL_oracle']['rmsle']:.4f}，"
      f"**提升 0.02%**）。也就是说：**这块信息里几乎没有可回收的预测价值。**")
    A("3. **未完成这一类的主导性质不是「运行得久」，而是「成批地从未收到结束记录」。**")
    A("   (tag, user, 起始分钟) 批次里 **99.9% 是「整批完成」或「整批未完成」**，")
    A("   中位已运行年龄 **13.7 天**。这更像 trace 的陈旧记账记录，而不是一次真实的运行时漂移。")
    A("")
    A("按你给的三条反证路径逐条对照：")
    A("")
    A("| 反证路径 | 本数据上的结果 |")
    A("|---|---|")
    A("| 1. 反馈延迟可能不重要（历史样本充足时简单统计已经够好） | **部分成立**：全量历史统计的 RMSLE 已经很低，且删失信息加不上去 |")
    A("| 2. 有偏差也可能不影响调度（预测误差降低但次序不变） | **未测**（本轮不接调度器），但预测层面已先失败 |")
    A("| 3. 简单删失估计可能就够 | **更强的情况：连 oracle 都不够。** 这是本轮的主要发现 |")
    A("")
    A("**结论：在 ATLAS/PAI 上，这个细分点不是「值得投入半年的候选缺口」，")
    A("而是一个被数据本身的记账噪声解释掉的现象。**")
    A("")
    A("---")
    A("")
    A("## 1. 数据入口（本轮实际拿到的东西）")
    A("")
    A("你给的仓库地址 `zhiyunjiang0810/non-clairvoyant-with-predictions` **返回 404**；")
    A("该账号公开仓库数为 1，且不是这个数据集。同名仓库实际在 **`gsadra/non-clairvoyant-with-predictions`**，")
    A("其中包含三张表与两个 notebook。数据本身来自公开的 Alibaba PAI 2020 GPU trace。")
    A("")
    A("| 文件 | 大小 | sha256 |")
    A("|---|---:|---|")
    for name, h in hashes.items():
        A(f"| `{name}` | {h['bytes']:,} | `{h['sha256']}` |")
    A("")
    A("解压后的三张表：")
    A("")
    A("| 表 | 行数 | 字段 |")
    A("|---|---:|---|")
    A("| `pai_job_table.csv` | 1,055,501 | job_name, inst_id, user, status, start_time, end_time |")
    A("| `pai_task_table.csv` | 1,261,050 | job_name, framework, inst_num, status, start_time, end_time, plan_cpu, plan_mem, plan_gpu, gpu_type |")
    A("| `pai_group_tag_table.csv` | 1,055,032 | inst_id, user_hash, group_id, tag_id, gpu_type_spec |")
    A("")
    A("### 1.1 连接关系（逐一验证过，不是照抄 README）")
    A("")
    A("| 连接 | 重叠 | 结论 |")
    A("|---|---|---|")
    A("| `task.col0` = `job.col0` | **1,055,501 / 1,055,501** | 任务表第 0 列**就是** job_name，可连 |")
    A("| `group_tag.col0` = `job.col1` | 1,053,971 | inst_id 可连 |")
    A("| `group_tag.col1` = `job.col2` | 1,430 | user 可连 |")
    A("| `task.col0` = `job.col1` | 0 | 不可连 |")
    A("")
    A("**必须记录的一处坑：** 官方 notebook 的表头把任务表写成 "
      "`[\"job_name\",\"task_name\",...]`，但它自己 `read_csv(header=None)` 的第 1 列实际是")
    A("**framework**（tensorflow / worker / ps / …，只有 24 个取值），而第 0 列才是 job_name。")
    A("若照抄该表头，`task.groupby(\"job_name\")` 会聚成 24 组，job→task 连接会静默失败。")
    A("")
    A("---")
    A("")
    A("## 2. 时间语义核实（这一条限制了整个选题）")
    A("")
    A("| 检查 | 结果 |")
    A("|---|---|")
    A(f"| `job.start_time` vs `min(task.start_time)` | **1,051,838 / 1,051,838 完全相等** |")
    A("| group-tag 表是否含提交时间列 | **否**。5 列分别为 inst_id / user_hash / group_id / tag_id / gpu_type_spec，**无任何时间戳** |")
    A("")
    A("**含义：这条 trace 不区分「提交/入队」与「开始执行」。** 数据集里的 `submit_time`")
    A("（ATLAS 自己的文档也这么叫）实际就是执行开始时间。因此：")
    A("")
    A("- 无法从这份数据重建**排队延迟**；")
    A("- 无法测量**端到端响应时间**；")
    A("- 能测的只是**执行时长**的反馈延迟。")
    A("")
    A("这与你要求的「响应时间」目标有实质差距，必须放在限制里，不能含糊过去。")
    A("")
    A("---")
    A("")
    A("## 3. 反馈结构：确实存在，且量级不小")
    A("")
    A(f"| 状态 | 行数 | 已启动 | 有结束时间 | 从未收到结束 |")
    A("|---|---:|---:|---:|---:|")
    for st, n, lau, he in (("Terminated", 732355, 732355, 732355),
                           ("Failed", 256555, 256555, 25620),
                           ("Running", 62928, 62928, 1061),
                           ("Waiting", 3663, 3089, 45)):
        A(f"| {st} | {n:,} | {lau:,} | {he:,} | {n-he:,} |")
    A("")
    A("注意：**ATLAS 的基准代码把两张表都过滤成 `status == 'Terminated'`**，")
    A("恰好丢掉全部未完成记录（job 表 296,420 行）。所以「未完成信息」不但存在，")
    A("而且是**被官方流程主动丢弃**的——这正是你设想的那个信息缺口。")
    A("")
    A("### 3.1 时间冻结更新点的删失比例")
    A("")
    A("| T/span | T (天) | 已到达 | 已完成 | 未完成(全部) | 占比 | 未完成(status=Running) | 占比 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for frac, day, arr, comp, ua, ur in (
            (0.20, 13.8, 95971, 75830, 20141, 6399),
            (0.40, 27.6, 281948, 223577, 58371, 19753),
            (0.60, 41.4, 503707, 387492, 116215, 34504),
            (0.80, 55.2, 738560, 565614, 172946, 49869),
            (0.95, 65.5, 1002720, 714341, 288379, 60190)):
        A(f"| {frac:.2f} | {day} | {arr:,} | {comp:,} | {ua:,} | {ua/arr:.2%} | {ur:,} | "
          f"{ur/arr:.2%} |")
    A("")
    A("**未完成比例稳定在 21–29%**（严格读法 6–7%），不随窗口位置剧烈变化。")
    A("")
    A("---")
    A("")
    A("## 4. 偏差确实存在：只用已完成作业会严重低估")
    A("")
    A("在 T/span = 0.6 这个冻结时点：")
    A("")
    A("| 群体 | n | 均值 | p50 | p90 |")
    A("|---|---:|---:|---:|---:|")
    A("| 已完成作业的真实时长 | 387,492 | 5,755 s | 620 s | 13,570 s |")
    A("| 未完成作业**已经跑了多久** | 116,215 | 1,377,920 s | 1,183,379 s | 2,579,422 s |")
    A("")
    A("**99.7% 的未完成作业已经超过了已完成作业的均值。** 只看已完成记录，")
    A("会得到严重偏低的时长分布——这正是你假设的机制。")
    A("")
    A("按作业类型看，偏差也确实**集中**（T/span = 0.6，按框架，取自 `s3_estimators.json`）：")
    A("")
    s3 = json.loads((HERE / "s3_estimators.json").read_text(encoding="utf-8"))
    fw_rows = [r for r in s3["0.6"] if r["kind"] == "framework"]
    fw_rows.sort(key=lambda r: -r["A_mean"])
    A("| 框架 | 已到达 | 已完成 | 未完成 | 只用已完成的均值 | KM 中位 | oracle 中位 |")
    A("|---|---:|---:|---:|---:|---|---|")
    for r in fw_rows[:6]:
        km = f"{r['C_km_median']:,.0f} s" if r["C_km_median"] is not None else "从未降到 0.5"
        orc = f"{r['oracle_p50']:,.0f} s" if r["oracle_p50"] is not None else "—"
        A(f"| {r['group']} | {r['n_arrived']:,} | {r['n_complete']:,} | {r['n_unfinished']:,} | "
          f"{r['A_mean']:,.0f} s | {km} | {orc} |")
    A("")
    A("而且**次序会变**：只用已完成的排序把 `JupyterTask` 排第一，censor-aware 排序把")
    A("`PyTorchWorker` 排第一；Kendall τ 只有 **+0.53 ~ +0.85**，")
    A("按 tag 分组时 **95% 以上的组名次发生变化**。")
    A("")
    A("**如果故事到此为止，这个方向就成立了。但下一节否掉了它。**")
    A("")
    A("---")
    A("")
    A("## 5. 决定性的检验：oracle 回归")
    A("")
    A("设计（严格避免自身泄漏）：")
    A("")
    A("- 在 `T_split` **之前**开始的作业上拟合；在 `[T_split, T_split+10%span)` 内**开始**的作业上评估；")
    A("- 测试集只保留已完成作业，标签真实；目标作业自己的结果在被观测前绝不进入训练；")
    A("- 特征是**对结果盲目**的：workload tag / framework / user（哈希分桶）、任务数；")
    A("  **不含已运行时长、不含任何编码结果的特征**；")
    A("- 标签为 `log1p(最终时长)`；岭回归，全部变体用同一套特征与超参。")
    A("")
    A("| 训练集变体 | 可部署 | n_train | RMSLE | Spearman |")
    A("|---|:--:|---:|---:|---:|")
    A(f"| COMPLETED（只用已完成） | 是 | {s4['COMPLETED']['n_train']:,} | "
      f"**{s4['COMPLETED']['rmsle']:.4f}** | {s4['COMPLETED']['spearman']:+.4f} |")
    A(f"| ALLFINAL（未完成作业的**最终**时长，oracle） | **否** | "
      f"{s4['ALLFINAL_oracle']['n_train']:,} | **{s4['ALLFINAL_oracle']['rmsle']:.4f}** | "
      f"{s4['ALLFINAL_oracle']['spearman']:+.4f} |")
    A(f"| COMPLETED + 组级删失特征 | 是 | "
      f"{s4['COMPLETED_plus_groupcensoring']['n_train']:,} | "
      f"{s4['COMPLETED_plus_groupcensoring']['rmsle']:.4f} | "
      f"{s4['COMPLETED_plus_groupcensoring']['spearman']:+.4f} |")
    A("")
    A(f"**可回收上限 = {(s4['COMPLETED']['rmsle']-s4['ALLFINAL_oracle']['rmsle'])/s4['COMPLETED']['rmsle']:.2%}**")
    A("（把 116,215 条未完成作业的最终时长全部白送，测试集 RMSLE 只降这么多）。")
    A("")
    A("### 5.1 这个数字为什么这么小")
    A("")
    A("未完成作业只占训练窗口的 23%，而且他们的**可观测协变量分布与已完成作业几乎一样**")
    A("（任务数 p50 都是 1、p90 都是 2；均值 1.14 vs 1.24）。")
    A("所以加进这批样本没有改变模型对**测试作业**的映射——")
    A("他们的标签虽不同，但他们长得和已有样本一样。")
    A("")
    A("这与你的第 2 条反证路径是同一种失败：**信息不同，但决策不需要它。**")
    A("")
    A("---")
    A("")
    A("## 6. 「未完成」到底是什么：陈旧记账，不是运行时漂移")
    A("")
    A("| 检查 | 结果 |")
    A("|---|---|")
    A("| 未完成记录的起始年龄 p50 | **13.7 天**（p95 = 32.7 天） |")
    A("| 30 天前就启动、至今无结束 | 11,353 条（占未完成 9.8%） |")
    A("| (tag, user, 起始分钟) 批次是否整批同状态 | **99.9%**（114,861 整批未完成 / 376,792 整批完成 / 仅 259 混合） |")
    A("| 未完成作业的任务数 vs 已完成 | 几乎相同（p50 = 1 vs 1） |")
    A("")
    A("**成批同状态 + 天文数字的运行年龄**，说明这一类的主导成因不是「作业变长了」，")
    A("而是**某些作业从未收到结束记录**（抢占、作业级故障、记账缺失）。")
    A("")
    A("只有 0.8% 的未完成作业后来真的收到了结束时间，且其中位总时长是已完成作业中位数的 **29 倍**。")
    A("")
    A("### 6.1 年龄匹配后的对照（这才是反馈延迟故事该有的样子）")
    A("")
    A("把范围限制在 T 前 2 天内启动的作业：")
    A("")
    A("| 类别 | n | 中位时间 | p90 |")
    A("|---|---:|---:|---:|")
    A("| 未完成（已获得的服务量） | 5,861 | **16.4 h** | — |")
    A("| 已完成（真实时长） | 20,909 | 0.2 h | 2.8 h |")
    A("")
    A("**85.8% 的未完成作业已经超过已完成作业的 p90。** 年龄匹配之后，删失效应是真实且强的。")
    A("所以问题不是「没有信号」，而是「**这个信号对预测未来作业没有用**」（第 5 节），")
    A("且它在总体中与陈旧记录混在一起（99.9% 成批同状态）。")
    A("")
    A("---")
    A("")
    A("## 7. 组级删失特征：有信号，但覆盖面太小")
    A("")
    A("| 检验 | 结果 |")
    A("|---|---|")
    A("| 组内前后半段删失比例的一致性（≥50 样本的 611 个组） | 中位绝对差 **0.061**，66.1% 在 0.10 以内 |")
    A("| 判定 | 中等稳定：可当特征，但不是干净的组属性 |")
    A("")
    A("按组的训练期删失比例分桶，看只用已完成模型的偏差：")
    A("")
    A("| 删失比例桶 | 测试 n | 真实 p50 | 预测 p50 | log 偏差 | RMSLE | 加删失特征后 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for b in s6["bins"]:
        A(f"| {b['bin']} | {b['n']:,} | {b['median_true']:,.0f} | "
          f"{b['median_pred_completed']:,.0f} | {b['log_ratio']:+.3f} | "
          f"{b['rmsle_completed']:.4f} | {b['rmsle_with_share']:.4f} |")
    A(f"| **总体** | 81,361 | | | | {s6['overall_completed']:.4f} | "
      f"{s6['overall_with_share']:.4f} |")
    A("")
    A("**两个问题：**")
    A("")
    A("1. 训练期删失比例 > 0.30 的组只覆盖测试集的 **2.18%**。")
    A("2. 偏差方向在最高桶**翻转**（`>0.60` 桶里模型反而**高估**，中位比为 0.45x）。")
    A("   一个方向不稳定的修正量不能作为特征使用。")
    A("")
    A("所以 `d_robust` 式的修正即使做出来，也只能作用在约 2% 的作业上，且方向不可靠。")
    A("")
    A("---")
    A("")
    A("## 8. 判读")
    A("")
    A("按你预先固定的三条反证路径：")
    A("")
    A("| # | 路径 | 判定 |")
    A("|---|---|---|")
    A("| 1 | 反馈延迟不重要 | **部分命中**：全量历史统计已足够，删失信息加不上去 |")
    A("| 2 | 有偏差但不影响决策 | **未测**（本轮不接调度器） |")
    A("| 3 | 简单删失估计就够 | **更强**：连 oracle 都不够（0.02% 上限） |")
    A("")
    A("**建议：不要在 ATLAS/PAI 上继续这个细分点。**")
    A("理由不是「没测到效应」，而是**效应的可回收上限被 oracle 直接封死在 0.02%**，")
    A("且主导成因与陈旧记账混同。继续做下去只会得到更多「未完成占比很高」的描述性结果。")
    A("")
    A("这也正是你要求的前置检查应该起的作用：**在训练任何模型之前就否掉它。**")
    A("")
    A("### 8.1 仍然值得保留的东西")
    A("")
    A("1. **反馈/标签权限的审计方法**是可复用的：")
    A("   逐时点冻结、只算已产生记录、oracle 对照明确标注不可部署。")
    A("2. **数据本身的可用性结论**有独立价值：")
    A("   ATLAS/PAI 的表头存在 job_name/framework 错位、三表连接关系与 README 不同、")
    A("   且**不含提交时间**。想要这三个结论的人不必再踩一遍。")
    A("3. **负面结论本身**：如果后续有人想在作业 trace 上做「利用未完成信息」，")
    A("   本报告给出的检验路径（oracle 回归 + 成批同状态检验 + 年龄匹配）可以直接复用。")
    A("")
    A("---")
    A("")
    A("## 9. 限制（必须与结论一起读）")
    A("")
    A("1. **无法测量排队延迟与响应时间。** 数据不含提交时间（§2），")
    A("   所以本轮结论只覆盖**执行时长**的反馈延迟，")
    A("   不能外推到「平均响应时间」这个目标。")
    A("2. **未接调度器。** 本轮不测 SRPT/FIFO 等固定规则下的实际收益，")
    A("   因此不能声称「有收益」或「没有收益」——只能说预测层面先失败。")
    A("3. **合成/真实混合。** 用的是真实生产 trace，但只做了抽象单资源时长的分析；")
    A("   没有做 GPU 放置、抢占或多机调度。")
    A("4. **oracle 上限依赖特征集。** 用的是对结果盲目的粗特征；")
    A("   更强的特征集可能让上限略有变化，但 0.02% 与「值得投入」之间的差距过大，")
    A("   量级结论不会因此翻转。")
    A("5. **官方 notebook 未运行。** 环境无 lightgbm/scipy，本轮用标准库自实现岭回归；")
    A("   没有复现 ATLAS 的预测基线，因此不能声称与它们的数字可比。")
    A("6. **单一数据集。** 结论只针对这份 trace。")
    A("")
    A("---")
    A("")
    A("## 10. 复现出口")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `RT_work/s1_schema.py` | 三张表的状态/时间戳分布扫描 |")
    A("| `RT_work/s1b_join.py` | 连接能力审计（含 job_name/task_name 错位证据） |")
    A("| `RT_work/s2_build.py` | 派生作业表构建 + 时间语义审计 → `rt_jobs.csv` |")
    A("| `RT_work/s2b_audit.py` | 精确时间语义与删失审计 |")
    A("| `RT_work/s2c_censoring.py` | 严格/宽松两种未完成读法的对照 |")
    A("| `RT_work/s3_estimators.py` | 已完成 / 未完成当数据 / KM / oracle 四估计器与次序比较 |")
    A("| `RT_work/s4_oracle_regression.py` | **决定性的 oracle 回归** |")
    A("| `RT_work/s5_stale.py` | 陈旧记录与成批同状态检验 |")
    A("| `RT_work/s6_targeted.py` | 组级删失特征的稳定性与定向偏差 |")
    A("")

    (ROOT / "RT01_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'RT01_report.md'}")

    cases = {
        "task": "RT-01 qualification check -- delayed completion feedback for runtime prediction",
        "verdict": "stop on this dataset",
        "verdict_reason": "the recoverable predictive value of the unfinished records is bounded at "
                          "0.02% of completed-only RMSLE by an oracle regression, and the unfinished "
                          "class is dominated by stale batch-completion records rather than feedback "
                          "latency",
        "data": {"repo": "gsadra/non-clairvoyant-with-predictions",
                 "note": "the repo named in the task (zhiyunjiang0810/...) returns 404; the dataset "
                         "lives under a different account",
                 "source": "Alibaba PAI 2020 GPU trace via the ATLAS release",
                 "files": hashes},
        "join_map": {
            "task.col0 == job.col0 (job_name)": "1,055,501 / 1,055,501",
            "group_tag.col0 == job.col1 (inst_id)": "1,053,971",
            "group_tag.col1 == job.col2 (user)": "1,430",
            "official notebook header bug": "its task_cols lists task_name at index 1, but index 1 is "
                                            "framework (24 distinct values); index 0 is the job name",
        },
        "timing_semantics": {
            "job.start_time == min(task.start_time)": "1,051,838 / 1,051,838 exactly equal",
            "group_tag_table_has_timestamp": False,
            "consequence": "no submission or queue-entry time exists, so queueing delay and response "
                           "time cannot be reconstructed from this trace",
        },
        "oracle_regression": s4,
        "targeted_bias": s6,
    }
    (ROOT / "RT01_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'RT01_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
