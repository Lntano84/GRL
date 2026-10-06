"""Q01: write the screening report and frozen manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MIRROR = HERE / "limeqo_mirror"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def main() -> int:
    prof = json.loads((HERE / "q01_data_profile.json").read_text(encoding="utf-8"))
    wit = json.loads((HERE / "q01_minimal_witness.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "q01_branch_audit.json").read_text(encoding="utf-8"))
    expo = json.loads((HERE / "q01_exposure.json").read_text(encoding="utf-8"))

    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted((MIRROR / "dataset").glob("*")) if p.stat().st_size}
    src_files = {str(p.relative_to(MIRROR)): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted((MIRROR / "src").rglob("*.py"))}
    code = (MIRROR / "src" / "strategies" / "limeqo.py").read_text(encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# Q01 —— 超时重试空间与强规则筛查（第一步：数据与反馈资格）")
    A("")
    A("## 结论")
    A("")
    A("**数据资格通过；反馈权限不通过。**")
    A("")
    A("| 项 | 结果 |")
    A("|---|---|")
    A("| 四张官方矩阵可直接回放 | **通过**（无需 Dropbox，无需数据库，无需 EXPLAIN 计划）|")
    A("| 数据完整性 | **通过**（无缺失、无无穷、无零、无负值；行标签唯一）|")
    A("| 合法反馈接口 | **不通过**：官方实现有两处分支读取**隐藏真值**来决定记账与候选集 |")
    A("| 重试臂 | **不存在**：`timeout_m` 从不被用于再次选择候选 |")
    A("| 因此 | 官方实现上的回放结果**不能直接当作可部署成绩**；本轮不据此评价论文结论 |")
    A("")
    A("---")
    A("")
    A("## 1. 数据资格（半天预算内完成）")
    A("")
    A("**入口。** 你给的仓库地址可用：`zixy17/LimeQO`（默认分支 `main`）。")
    A("README 说明 `dataset/` 只含部分数据、完整数据需从 Dropbox 下载，")
    A("**但四张矩阵 CSV 和四个初始掩码都已随仓库提供**，因此第一步不需要 Dropbox。")
    A("EXPLAIN 计划（`dataset/*.zip`）只对 LimeQO+ 训练需要，本步不需要。")
    A("")
    A("| 工作负载 | 行（查询） | 列（hint） | 行标签唯一 | 取值范围（秒）| 单元格中位（秒）| 缺失 | 无穷 | 零 | 负 |")
    A("|---|---:|---:|:--:|---|---:|---:|---:|---:|---:|")
    for name in ("ceb", "job", "stack", "dsb"):
        d = prof.get(name)
        if not d:
            continue
        A(f"| {name} | {d['rows']:,} | {d['cols']} | {d['unique_row_labels'] == d['rows']} | "
          f"{d['min']:.4g} – {d['max']:.4g} | {d['median']:.4g} | {d['nan']} | {d['inf']} | "
          f"{d['zero']} | {d['negative']} |")
    A("")
    A("（数值单位由 README 明确为**秒**。）")
    A("")
    A("### 1.1 已核实的回放语义")
    A("")
    A("| 要素 | 结论 | 依据 |")
    A("|---|---|---|")
    A("| 矩阵含义 | 行 = 查询，列 = hint，单元格 = 该 hint 下的**真实运行时间（秒）** | README + `dataset.py` |")
    A("| 默认计划 | **第 0 列**。`default_time = sum(matrix[:,0])`；它等于该行最优的比例为 "
      f"ceb {prof['ceb']['col0_is_row_min']}/{prof['ceb']['rows']}、"
      f"job {prof['job']['col0_is_row_min']}/{prof['job']['rows']}、"
      f"stack {prof['stack']['col0_is_row_min']}/{prof['stack']['rows']}、"
      f"dsb {prof['dsb']['col0_is_row_min']}/{prof['dsb']['rows']} | `dataset.py:34-35` |")
    A("| 初始掩码 | `init_<name>_mask.npy`，形状与矩阵一致，标记起跑前**已观测**的单元格 | `dataset.py:31` |")
    A("| 重复计划对应关系 | 由**并查集**给出：`get_same_hints(q,h)` 返回与 h 等价的全部 hint；"
      "观测一个即等价观测整组 | `dataset.py:188-203`，`utils/union_find.py` |")
    A("| 矩阵中的数值重复 | 每个数据集都有若干**完全相同**的列：ceb/job/dsb 各 7 对、stack 15 对；"
      "例如列 (5,6)、(12,13)、(19,20) 在四张表里都成对相同 | `q01_data_profile.json` |")
    A("| 注意 | 并查集来自 EXPLAIN 计划（需 Dropbox）；**矩阵 CSV 本身不含等价类信息**。"
      "若不做等价合并，回放会低估“一次观测揭示多条 hint”的效果 |")
    A("")
    A("### 1.2 可改进空间（诊断量）")
    A("")
    A("| 工作负载 | 严格优于本行默认计划的单元格 | 占比 | 相对默认计划的中位加速 | 最大加速 |")
    A("|---|---:|---:|---:|---:|")
    for name in ("ceb", "job", "stack", "dsb"):
        d = expo.get(name)
        if not d:
            continue
        A(f"| {name} | {d['cells_better_than_default']:,} | {d['share']:.1%} | "
          f"{d['median_improvement_factor']:.2f}x | {d['max_improvement_factor']:.1f}x |")
    A("")
    A("默认计划只在少数查询上是最优的，因此**提升空间真实存在**；这不是新发现，")
    A("只是确认这份数据能支撑后续的成本—质量比较。")
    A("")
    A("---")
    A("")
    A("## 2. 反馈权限：两处分支读取隐藏真值（已坐实）")
    A("")
    A("### 2.1 负载分支")
    A("")
    A("```python")
    A("dataset.py:30   self.matrix = self.matrix_df.to_numpy()        # 真实运行时间，全部隐藏")
    A("limeqo.py:89    if dataset.matrix[select, hint] >= min_observed[select]:")
    A("limeqo.py:90        explored_m[select, same_hints] = 1")
    A("limeqo.py:92    if dataset.matrix[select, hint] >= timeout_tolerance:")
    A("limeqo.py:93        timeout_m[select, same_hints] = timeout_tolerance")
    A("limeqo.py:97    mask[select, same_hints] = 1")
    A("limeqo.py:98    explored_m[select, same_hints] = 1")
    A("```")
    A("")
    A("### 2.2 最小见证")
    A("")
    A("固定上下文：当前最好值 `b = 10`，候选在 `t = 2` 被取消，`timeout_tolerance = 10`。")
    A("**合法的唯一反馈是「运行时间 ≥ 2」。** 只改变隐藏真实运行时间 x：")
    A("")
    A("| 隐藏 x | x ≥ t（合法可见）| x ≥ b | x ≥ tol | 写入 `mask` | 写入 `timeout_m` | 记账结果 |")
    A("|---:|:--:|:--:|:--:|:--:|:--:|---|")
    for r in wit["rows"]:
        A(f"| {r['x']} | {r['legal_visible']} | {r['x'] >= wit['fixed_context']['best']} | "
          f"{r['x'] >= wit['fixed_context']['timeout_tolerance']} | "
          f"{'是' if r['mask'] is not None else '否'} | "
          f"{'是' if r['timeout_m'] is not None else '否'} | {r['outcome']} |")
    A("")
    A("**同一个合法观察「运行时间 ≥ 2」导致两种互斥的记账：**")
    A("")
    A("- `x < tol`：被取消的运行被写进 `mask`，即**当作一次完整的运行时间测量**；")
    A("- `x ≥ tol`：被写进 `timeout_m`，即当作**删失下界**。")
    A("")
    A("两者的差别不只是记账：写进 `mask` 的那一支还会 `cnt += 1`（计入 `new_observe_size`）")
    A("并把该单元格从后续候选中移除。所以**隐藏真值同时决定了记账和候选集**。")
    A("")
    A("**需要精确说明的一点。** 在本例中 `alpha*b = tol`，所以若截止值取 `t = alpha*b`，")
    A("则 `t = tol`，两支的分界恰好落在 t 上。真正会被错记的窗口是")
    A("**`t ≤ x < tol`**（运行被截断、但仍落在容忍度内）：此时代码把一次被取消的运行")
    A("当作测量，而合法反馈无法把它与「运行完成且耗时 x」区分开。")
    A("")
    A("### 2.3 同类分支遍布全部策略")
    A("")
    A("源码扫描发现**未加掩码**读取真实矩阵的位置共 **9 处**：")
    A("")
    A("| 文件 | 行 | 代码 |")
    A("|---|---:|---|")
    for h in audit["source_scan"]:
        if not h["masked_in_line"]:
            A(f"| `{h['file']}` | {h['line']} | `{h['code'][:76]}` |")
    A("")
    A("因此这不是 LimeQO 独有的一行疏漏，而是**整组策略共用的实现约定**：")
    A("`random`、`greedy`、`qo_advisor`、`limeqo`、`limeqo_plus` 都依赖它。")
    A("")
    A("---")
    A("")
    A("## 3. 不存在重试臂（对 Q01 主问题直接相关）")
    A("")
    A("`timeout_m` 的全部出现位置：")
    A("")
    A("| 行 | 类型 | 代码 |")
    A("|---:|---|---|")
    A("| 28 | 初始化 | `timeout_m = np.zeros_like(dataset.matrix)` |")
    A("| 41 | **唯一读取** | `log_timeout_m = np.log1p(timeout_m)` |")
    A("| 44 | 传入 ML | `pred_m = censored_als(log_m, mask, log_timeout_m, ...)` |")
    A("| 93 | 写入 | `timeout_m[select, same_hints] = timeout_tolerance` |")
    A("| 117 | 写入 | `timeout_m[file_i, same_hints] = min_observed[file_i]` |")
    A("")
    A("唯一一次读取是把删失指示喂给矩阵分解。**没有任何代码用 `timeout_m` 决定是否以更长截止值重跑**，")
    A("候选过滤器（82–85 行）也不看它。")
    A("")
    A("**这对选题的含义：** 官方实现里**没有「重试旧超时候选」这件事**。")
    A("所以「何时重试」既不是它已经解决的问题，也不是它已经排除的问题——")
    A("它是一个**尚未被这套代码表达**的决策，这既不是优势也不是缺口，需要独立检验。")
    A("")
    A("---")
    A("")
    A("## 4. 本轮不做什么、以及现在的证据层次")
    A("")
    A("本轮**没有**：训练任何模型、运行完整实验、下载 Dropbox 数据、重建数据库、调窗口或阈值。")
    A("")
    A("| 层次 | 当前判断 |")
    A("|---|---|")
    A("| 存在优化空间 | **有**：16.5–23.2% 的单元格优于本行默认计划 |")
    A("| 合法观测可辨识 | **未测**：接口要先修，本轮只证伪了现成实现 |")
    A("| 学习能够捕捉 | 未测 |")
    A("| 收益超过成本 | 未测 |")
    A("| 相对已有方法有新意 | 未建立；BayesQO / LimeQO 仍是对手 |")
    A("")
    A("### 4.1 对下一步的直接影响")
    A("")
    A("按你的计划，第 2 步是「建立最小反馈接口」。本轮结果表明该接口**不能建立在官方策略代码之上**，")
    A("需要：")
    A("")
    A("1. 把 `matrix` 从策略可见对象中彻底移除，只暴露 `observe(q,h) -> (value, kind)`，")
    A("   其中 `kind ∈ {measured, cancelled_at_t}`；")
    A("2. 记账规则改为**只看观测类型**：`cancelled_at_t` 一律进 `timeout_m`，")
    A("   不论隐藏真值是否落在容忍度内；")
    A("3. 用本报告 §2.2 的两世界例子作为**回归测试**：两个世界必须给出相同的策略状态；")
    A("4. 若要复用官方策略做对照，必须把 89/92（以及其它 7 处）替换为只读观测的版本，")
    A("   并**明确标注这是修改版**，不能称为官方成绩。")
    A("")
    A("---")
    A("")
    A("## 5. 产物与哈希")
    A("")
    A("### 5.1 冻结的代码版本")
    A("")
    A("| 文件 | sha256 |")
    A("|---|---|")
    for k, v in src_files.items():
        A(f"| `{k}` | `{v['sha256']}` |")
    A("")
    A("### 5.2 数据文件")
    A("")
    A("| 文件 | 字节 | sha256 |")
    A("|---|---:|---|")
    for k, v in files.items():
        A(f"| `{k}` | {v['bytes']:,} | `{v['sha256']}` |")
    A("")
    A("### 5.3 脚本")
    A("")
    A("| 脚本 | 作用 |")
    A("|---|---|")
    A("| `Q01_work/q01_fetch.py` | 按仓库路径抓取文件（>1MB 走 download_url）|")
    A("| `Q01_work/q01_profile.py` | 四张矩阵的数据资格剖析 |")
    A("| `Q01_work/q01_two_world.py` | 用官方策略跑两世界，记录真值读取 |")
    A("| `Q01_work/q01_attribution.py` | 把每次真值读取归因到调用行 |")
    A("| `Q01_work/q01_branch_audit.py` | 分支真值表 + 未掩码读取源码扫描 + 暴露量扫描 |")
    A("| `Q01_work/q01_minimal_witness.py` | **最小见证**（决策类型 vs 隐藏真值）|")
    A("| `Q01_work/q01_exposure.py` | 错记窗口与「无重试臂」确认 |")
    A("")

    (ROOT / "Q01_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01_report.md'}")

    cases = {
        "task": "Q01 step 1 -- data and feedback qualification for the timeout-retry screening",
        "status": "data qualification PASSED; feedback permission FAILED; no retry arm exists",
        "repo": {"name": "zixy17/LimeQO", "default_branch": "main",
                 "note": "the four matrix CSVs and the four init masks ship with the repo, so the "
                         "Dropbox download and the EXPLAIN-plan zip are NOT needed for this round"},
        "data_profile": prof,
        "replay_semantics": {
            "matrix": "rows = queries, cols = hints, cell = true runtime in SECONDS",
            "default_plan": "column 0; dataset.default_time = sum(matrix[:,0])",
            "initial_mask": "init_<name>_mask.npy marks cells already observed at start",
            "duplicate_plan_correspondence": "given by a UNION-FIND over hints per query, built from "
                                             "the EXPLAIN plans; get_same_hints(q,h) returns the whole "
                                             "equivalence class.  The matrix CSVs alone do not carry "
                                             "this information.  The matrices do contain exactly "
                                             "identical column pairs (7 for ceb/job/dsb, 15 for "
                                             "stack), which is consistent with the equivalence "
                                             "structure.",
        },
        "feedback_permission": {
            "violation_lines": ["limeqo.py:89", "limeqo.py:92", "limeqo.py:114"],
            "other_files_with_same_pattern": audit["source_scan"],
            "minimal_witness": wit,
            "explanation": "the branch at line 92 splits on the HIDDEN runtime: x < tol books a "
                           "cancelled run into mask as a completed measurement, x >= tol books it "
                           "into timeout_m as a censored bound.  Legal feedback ('runtime >= 2') "
                           "cannot choose between these.  The misbooked window is t <= x < tol.",
        },
        "retry_arm": {"present": False,
                      "evidence": "timeout_m is written at lines 93 and 117 and read only at line 41 "
                                  "as the censoring indicator for the matrix-factorization stage; no "
                                  "line uses it to re-run a candidate, and the candidate filter at "
                                  "82-85 never consults it"},
        "exposure": expo,
        "not_done": ["no model training", "no full experiment run", "no Dropbox download",
                     "no database rebuild", "no window or threshold search"],
        "next_step_requirements": [
            "replace the policy's access to `matrix` with observe(q,h) -> (value, kind) where kind "
            "is measured or cancelled_at_t",
            "make bookkeeping depend only on the observation kind",
            "use the two-world example as a regression test",
            "if the official strategies are reused as baselines, mark them as modified",
        ],
    }
    (ROOT / "Q01_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
