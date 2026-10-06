"""Add the full-sample paired estimator to a completed SB01 replay.

This is an offline post-processing step. It reads the audited public score
matrix and the already saved sample manifest; it does not draw samples, call a
SAT solver, or change the original replay rows.
"""

from __future__ import annotations

import csv
import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WORK = ROOT / "work" / "sb01"
MATRIX_PATH = WORK / "anni-seq.csv"
EXPECTED_MATRIX_SHA256 = "561e192a585bccc8cd326af34c931e0137fcd6ee0fb8bf1e8a1535aae108b465"

METHODS = (
    "ordinary_sample_mean",
    "best_history_corrected",
    "best_history_paired192",
    "nearest_history_corrected",
    "ridge_alpha1_corrected",
)
LABELS = {
    "ordinary_sample_mean": "普通样本均值（P∪S共192次测量）",
    "best_history_corrected": "最佳历史版本 + 原分组残差校正",
    "best_history_paired192": "最佳历史版本 + 全样本配对（192）",
    "nearest_history_corrected": "先导最相近历史列 + 残差校正",
    "ridge_alpha1_corrected": "岭回归 α=1 + 残差校正",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify(gap: float) -> str:
    if abs(gap) < 1e-9:
        return "tie"
    return "new_better" if gap < 0 else "old_better"


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def relation_counts(values: list[float], baseline: list[float]) -> tuple[int, int, int]:
    differences = np.asarray(values, dtype=float) - np.asarray(baseline, dtype=float)
    lower = int(np.sum(differences < -1e-12))
    equal = int(np.sum(np.abs(differences) <= 1e-12))
    higher = len(differences) - lower - equal
    return lower, equal, higher


def method_metrics(per_seed: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, Counter[str]]]:
    version_metrics: dict[str, dict[str, float]] = {}
    confusions: dict[str, Counter[str]] = {}
    for solver, group in per_seed.groupby("hidden_solver", sort=False):
        for method in METHODS:
            signed = group[f"{method}_signed_error"].astype(float).to_numpy()
            mistakes = np.asarray([as_bool(v) for v in group[f"{method}_misclassified"]])
            version_metrics.setdefault(method, {})[str(solver)] = {
                "mae": float(np.mean(np.abs(signed))),
                "rmse": float(np.sqrt(np.mean(signed**2))),
                "bias": float(np.mean(signed)),
                "misclassification_rate": float(np.mean(mistakes)),
            }

    for method in METHODS:
        confusion: Counter[str] = Counter()
        for row in per_seed.to_dict(orient="records"):
            truth = str(row["true_comparison"])
            predicted = str(row[f"{method}_predicted_comparison"])
            if truth == "new_better":
                confusion["true_new_called_new" if predicted == "new_better" else "true_new_missed"] += 1
            elif truth == "old_better":
                if predicted == "old_better":
                    confusion["true_old_called_old"] += 1
                elif predicted == "new_better":
                    confusion["false_new_winner"] += 1
                else:
                    confusion["old_called_tie"] += 1
            elif predicted != "tie":
                confusion["true_tie_misclassified"] += 1
        confusions[method] = confusion
    return version_metrics, confusions


def section_replace(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def main() -> None:
    if sha256(MATRIX_PATH) != EXPECTED_MATRIX_SHA256:
        raise ValueError("The audited anni-seq.csv checksum does not match")

    matrix = pd.read_csv(MATRIX_PATH, dtype={"hash": "string"}).set_index("hash")
    manifest = pd.read_csv(OUT / "SB01_sample_manifest.csv", dtype={"pilot_hash_ids_in_draw_order": "string", "correction_hash_ids_in_draw_order": "string"})
    per_seed = pd.read_csv(OUT / "SB01_per_seed_estimates.csv")
    old_method_summary = pd.read_csv(OUT / "SB01_method_summary.csv").to_dict(orient="records")
    old_version_rows = pd.read_csv(OUT / "SB01_version_results.csv").to_dict(orient="records")

    manifest_by_seed = {int(row.seed): row for row in manifest.itertuples(index=False)}
    if len(per_seed) != 28 * 20 or len(manifest_by_seed) != 20:
        raise ValueError("Unexpected replay or sample-manifest dimensions")

    per_seed["best_history_paired192_estimate"] = np.nan
    update_times_ns: list[int] = []
    for row_index, row in per_seed.iterrows():
        seed = int(row["sampling_seed"])
        saved_sample = manifest_by_seed[seed]
        pilot_hashes = str(saved_sample.pilot_hash_ids_in_draw_order).split("|")
        correction_hashes = str(saved_sample.correction_hash_ids_in_draw_order).split("|")
        sample_hashes = pilot_hashes + correction_hashes
        if len(pilot_hashes) != 64 or len(correction_hashes) != 128 or len(set(sample_hashes)) != 192:
            raise ValueError(f"Invalid saved sample for seed {seed}")
        if not set(sample_hashes).issubset(matrix.index):
            raise ValueError(f"Manifest contains hashes missing from the audited matrix for seed {seed}")

        target_scores = matrix.loc[sample_hashes, str(row["hidden_solver"])].to_numpy(dtype=float)
        old_scores = matrix.loc[sample_hashes, str(row["historical_comparator"])].to_numpy(dtype=float)
        old_mean = float(row["historical_mean_par2"])
        started = time.perf_counter_ns()
        estimate = old_mean + float(np.mean(target_scores - old_scores))
        update_times_ns.append(time.perf_counter_ns() - started)

        estimated_gap = estimate - old_mean
        truth = str(row["true_comparison"])
        prediction = classify(estimated_gap)
        error = estimate - float(row["true_mean_par2"])
        per_seed.at[row_index, "best_history_paired192_estimate"] = estimate
        per_seed.at[row_index, "best_history_paired192_signed_error"] = error
        per_seed.at[row_index, "best_history_paired192_absolute_error"] = abs(error)
        per_seed.at[row_index, "best_history_paired192_estimated_new_minus_old_gap"] = estimated_gap
        per_seed.at[row_index, "best_history_paired192_predicted_comparison"] = prediction
        per_seed.at[row_index, "best_history_paired192_misclassified"] = prediction != truth

    per_seed.to_csv(OUT / "SB01_per_seed_estimates.csv", index=False, encoding="utf-8-sig")

    metrics, confusions = method_metrics(per_seed)
    solver_order = [str(x) for x in per_seed["hidden_solver"].drop_duplicates().tolist()]
    existing_version_by_solver = {str(row["hidden_solver"]): row for row in old_version_rows}
    version_rows: list[dict[str, Any]] = []
    for solver in solver_order:
        record = dict(existing_version_by_solver[solver])
        # Remove the old ambiguous relative-to-paired fields before rebuilding
        # all relative indicators against the stronger paired192 baseline.
        for key in list(record):
            if key.endswith("_improves_vs_paired") or key.endswith("_improves_vs_paired192"):
                record.pop(key)
        group = per_seed[per_seed["hidden_solver"].astype(str) == solver]
        for method in METHODS:
            vals = metrics[method][solver]
            for name, value in vals.items():
                record[f"{method}_{name}"] = value
        baseline = metrics["best_history_paired192"][solver]
        for method in METHODS:
            if method == "best_history_paired192":
                continue
            record[f"{method}_mae_improves_vs_paired192"] = (
                metrics[method][solver]["mae"] < baseline["mae"] - 1e-12
            )
            record[f"{method}_misclass_improves_vs_paired192"] = (
                metrics[method][solver]["misclassification_rate"]
                < baseline["misclassification_rate"] - 1e-12
            )
        version_rows.append(record)
    version_rows.sort(key=lambda record: abs(float(record["true_new_minus_old_gap"])))
    write_csv(OUT / "SB01_version_results.csv", version_rows)

    timing_by_method = {str(row["estimator"]): row for row in old_method_summary}
    old_pair_timing = timing_by_method["best_history_corrected"]
    summary_rows: list[dict[str, Any]] = []
    version_mae = {
        method: [metrics[method][solver]["mae"] for solver in solver_order]
        for method in METHODS
    }
    version_miss = {
        method: [metrics[method][solver]["misclassification_rate"] for solver in solver_order]
        for method in METHODS
    }
    base_mae = version_mae["best_history_paired192"]
    base_miss = version_miss["best_history_paired192"]

    for method in METHODS:
        row = dict(timing_by_method.get(method, {}))
        if not row:
            row = {
                "estimator": method,
                "fixed_setup_total_ms": float(old_pair_timing["fixed_setup_total_ms"]),
                "pilot_fit_calls": 0,
                "pilot_fit_total_ms": 0.0,
                "pilot_fit_mean_ms_per_call": 0.0,
                "pilot_fit_p95_ms_per_call": 0.0,
                "prediction_calls": 0,
                "prediction_total_ms": 0.0,
                "prediction_mean_ms_per_call": 0.0,
                "prediction_p95_ms_per_call": 0.0,
                "estimate_update_total_ms": float(sum(update_times_ns) / 1e6),
                "estimate_update_mean_ms_per_call": float(np.mean(update_times_ns) / 1e6),
                "estimate_update_p95_ms_per_call": float(np.percentile(update_times_ns, 95) / 1e6),
                "measurement_values_per_version_seed": 192,
            }
        for key in list(row):
            if "_vs_paired" in key:
                row.pop(key)
        confusion = confusions[method]
        row.update(
            {
                "estimator": method,
                "label_zh": LABELS[method],
                "macro_mean_per_version_mae_par2": float(np.mean([metrics[method][s]["mae"] for s in solver_order])),
                "macro_mean_per_version_rmse_par2": float(np.mean([metrics[method][s]["rmse"] for s in solver_order])),
                "macro_mean_per_version_signed_bias_par2": float(np.mean([metrics[method][s]["bias"] for s in solver_order])),
                "macro_misclassification_rate": float(np.mean(version_miss[method])),
                "misclassification_versions_pct": 100.0 * float(np.mean(version_miss[method])),
                "true_new_better_called_new_count": confusion["true_new_called_new"],
                "true_new_better_missed_count": confusion["true_new_missed"],
                "true_old_better_called_old_count": confusion["true_old_called_old"],
                "false_new_winner_count": confusion["false_new_winner"],
                "true_old_better_predicted_tie_count": confusion["old_called_tie"],
            }
        )
        if method == "best_history_paired192":
            for key in (
                "hidden_versions_with_lower_mae_vs_paired192",
                "hidden_versions_equal_mae_vs_paired192",
                "hidden_versions_with_higher_mae_vs_paired192",
                "hidden_versions_with_lower_misclass_vs_paired192",
                "hidden_versions_equal_misclass_vs_paired192",
                "hidden_versions_with_higher_misclass_vs_paired192",
            ):
                row[key] = ""
        else:
            mae_relation = relation_counts(version_mae[method], base_mae)
            miss_relation = relation_counts(version_miss[method], base_miss)
            row.update(
                {
                    "hidden_versions_with_lower_mae_vs_paired192": mae_relation[0],
                    "hidden_versions_equal_mae_vs_paired192": mae_relation[1],
                    "hidden_versions_with_higher_mae_vs_paired192": mae_relation[2],
                    "hidden_versions_with_lower_misclass_vs_paired192": miss_relation[0],
                    "hidden_versions_equal_misclass_vs_paired192": miss_relation[1],
                    "hidden_versions_with_higher_misclass_vs_paired192": miss_relation[2],
                }
            )
        row["measurement_values_per_version_seed"] = 192
        summary_rows.append(row)

    # Keep the table schema explicit and consistent across the original and new rows.
    summary_columns = [
        "estimator", "label_zh", "macro_mean_per_version_mae_par2", "macro_mean_per_version_rmse_par2",
        "macro_mean_per_version_signed_bias_par2", "macro_misclassification_rate", "misclassification_versions_pct",
        "true_new_better_called_new_count", "true_new_better_missed_count", "true_old_better_called_old_count",
        "false_new_winner_count", "true_old_better_predicted_tie_count",
        "hidden_versions_with_lower_mae_vs_paired192", "hidden_versions_equal_mae_vs_paired192",
        "hidden_versions_with_higher_mae_vs_paired192", "hidden_versions_with_lower_misclass_vs_paired192",
        "hidden_versions_equal_misclass_vs_paired192", "hidden_versions_with_higher_misclass_vs_paired192",
        "fixed_setup_total_ms", "pilot_fit_calls", "pilot_fit_total_ms", "pilot_fit_mean_ms_per_call",
        "pilot_fit_p95_ms_per_call", "prediction_calls", "prediction_total_ms", "prediction_mean_ms_per_call",
        "prediction_p95_ms_per_call", "estimate_update_total_ms", "estimate_update_mean_ms_per_call",
        "measurement_values_per_version_seed",
    ]
    for row in summary_rows:
        for key in summary_columns:
            row.setdefault(key, "")
    write_csv(OUT / "SB01_method_summary.csv", [{key: row[key] for key in summary_columns} for row in summary_rows])

    summary_by_method = {row["estimator"]: row for row in summary_rows}
    version_by_solver = {str(row["hidden_solver"]): row for row in version_rows}
    table_header = "| 估计器 | 版本宏平均绝对误差（PAR-2秒） | 版本宏平均RMSE | 版本宏平均偏差 | 新旧误判率 | 相对全样本配对估计（paired192）：MAE 更低 / 相同 / 更高版本 | 相对全样本配对估计（paired192）：误判更低 / 相同 / 更高版本 |"

    def count_triplet(row: dict[str, Any], stem: str) -> str:
        a = row.get(f"hidden_versions_with_lower_{stem}_vs_paired192", "")
        b = row.get(f"hidden_versions_equal_{stem}_vs_paired192", "")
        c = row.get(f"hidden_versions_with_higher_{stem}_vs_paired192", "")
        return "—" if a == "" else f"{a} / {b} / {c}"

    main_table = "\n".join(
        [table_header, "|---|---:|---:|---:|---:|---:|---:|"]
        + [
            f"| {row['label_zh']} | {float(row['macro_mean_per_version_mae_par2']):.2f} | "
            f"{float(row['macro_mean_per_version_rmse_par2']):.2f} | "
            f"{float(row['macro_mean_per_version_signed_bias_par2']):.2f} | "
            f"{float(row['misclassification_versions_pct']):.1f}% | "
            f"{count_triplet(row, 'mae')} | {count_triplet(row, 'misclass')} |"
            for row in summary_rows
        ]
    )
    report_path = OUT / "SB01_summary.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("估计四种方法。未运行 SAT 求解器。", "估计五种方法。未运行 SAT 求解器。")
    report = section_replace(report, "| 估计器 | 版本宏平均绝对误差", "### 新旧判断误报与漏报", main_table + "\n\n版本级相对优势只统计 28 个隐藏版本；每版本内先对 20 个配对种子汇总。")

    confusion_table = "\n".join(
        [
            "| 估计器 | 真正新版本更优：判对 / 漏报 | 旧版本更优：判对 / 误报新版本更优 / 判平局 |",
            "|---|---:|---:|",
        ]
        + [
            f"| {row['label_zh']} | {row['true_new_better_called_new_count']} / {row['true_new_better_missed_count']} | "
            f"{row['true_old_better_called_old_count']} / {row['false_new_winner_count']} / "
            f"{row['true_old_better_predicted_tie_count']} |"
            for row in summary_rows
        ]
    )
    report = section_replace(
        report,
        "### 新旧判断误报与漏报",
        "## 新旧总体成绩比较",
        "### 新旧判断误报与漏报\n\n下表计数覆盖每个隐藏版本的 20 个配对抽样重放；重放数不代表独立新版本数。\n\n" + confusion_table,
    )

    paired128 = summary_by_method["best_history_corrected"]
    paired192 = summary_by_method["best_history_paired192"]
    ridge = summary_by_method["ridge_alpha1_corrected"]
    ridge_mae_gain = 100.0 * (
        float(paired192["macro_mean_per_version_mae_par2"])
        - float(ridge["macro_mean_per_version_mae_par2"])
    ) / float(paired192["macro_mean_per_version_mae_par2"])
    pair192_misses = sum(as_bool(value) for value in per_seed["best_history_paired192_misclassified"])
    ridge_misses = sum(as_bool(value) for value in per_seed["ridge_alpha1_corrected_misclassified"])
    true_new_solver = str(per_seed.loc[per_seed["true_comparison"] == "new_better", "hidden_solver"].iloc[0])
    target_rows = per_seed[per_seed["hidden_solver"].astype(str) == true_new_solver]
    pair192_new_correct = int(sum(str(x) == "new_better" for x in target_rows["best_history_paired192_predicted_comparison"]))
    ridge_new_correct = int(sum(str(x) == "new_better" for x in target_rows["ridge_alpha1_corrected_predicted_comparison"]))
    close_rows = [version_by_solver[str(row["hidden_solver"])] for row in version_rows[:2]]
    close_pair_miss = 100.0 * float(np.mean([float(row["best_history_paired192_misclassification_rate"]) for row in close_rows]))
    close_ridge_miss = 100.0 * float(np.mean([float(row["ridge_alpha1_corrected_misclassification_rate"]) for row in close_rows]))
    ridge_mae_relation = relation_counts(version_mae["ridge_alpha1_corrected"], version_mae["best_history_paired192"])
    ridge_miss_relation = relation_counts(version_miss["ridge_alpha1_corrected"], version_miss["best_history_paired192"])
    closest_table = "\n".join(
        [
            "| 隐藏新版本 | 最佳历史列 | 真实差距（PAR-2秒） | 普通均值误判率 | 原分组配对误判率 | 全样本配对误判率 | 最近历史误判率 | 岭回归误判率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        + [
            f"| {row['hidden_solver']} | {row['historical_comparator']} | {float(row['true_new_minus_old_gap']):.2f} | "
            f"{100 * float(row['ordinary_sample_mean_misclassification_rate']):.1f}% | "
            f"{100 * float(row['best_history_corrected_misclassification_rate']):.1f}% | "
            f"{100 * float(row['best_history_paired192_misclassification_rate']):.1f}% | "
            f"{100 * float(row['nearest_history_corrected_misclassification_rate']):.1f}% | "
            f"{100 * float(row['ridge_alpha1_corrected_misclassification_rate']):.1f}% |"
            for row in close_rows
        ]
    )
    overall_section = (
        "## 新旧总体成绩比较\n\n"
        "本留一列比较规则把每个新版本与其余历史列中的全局最佳者比较。无并列时，只有全局第一名被留出的一折会是真正更优的挑战者；因此本轮只有一个正例是比较器定义带来的覆盖边界。\n\n"
        f"全样本配对估计的宏平均 MAE 为 {float(paired192['macro_mean_per_version_mae_par2']):.2f}，误判 {pair192_misses}/560；原分组配对估计为 {float(paired128['macro_mean_per_version_mae_par2']):.2f}、{sum(as_bool(v) for v in per_seed['best_history_corrected_misclassified'])}/560。岭回归 MAE 为 {float(ridge['macro_mean_per_version_mae_par2']):.2f}，比全样本配对低 {ridge_mae_gain:.2f}%；但总误判为 {ridge_misses}/560。按 28 个版本计，岭回归 MAE 更低 / 相同 / 更高为 {ridge_mae_relation[0]} / {ridge_mae_relation[1]} / {ridge_mae_relation[2]}，误判更低 / 相同 / 更高为 {ridge_miss_relation[0]} / {ridge_miss_relation[1]} / {ridge_miss_relation[2]}。唯一真实更优的新版本是 `{true_new_solver}`：全样本配对正确 {pair192_new_correct}/20，岭回归正确 {ridge_new_correct}/20。\n\n"
        "最近的两折分别是 `Kissat_MAB_ESA` 对 `kissat-sc2022-bulky`，以及反向比较。它们是同一无序求解器对的两个留出方向，训练条件各异，但不是两个独立近差距案例；真实差距绝对值为 4.56 PAR-2 秒，约占比较器均分的 0.162%。这两折的平均误判率为全样本配对 "
        f"{close_pair_miss:.1f}%、岭回归 {close_ridge_miss:.1f}%。对这一极接近的求解器对，本批估计没有实现可靠区分，也未观察到岭回归改善判断。\n\n"
        "| 隐藏新版本 | 最佳历史列 | 真实差距（PAR-2秒） | 普通均值误判率 | 原分组配对误判率 | 全样本配对误判率 | 最近历史误判率 | 岭回归误判率 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(closest_table.splitlines()[2:])
        + "\n\n"
        "本轮结论：合法历史信息明显改善了普通抽样估计；但充分利用全部已测数据的简单配对方法，已解释了大部分收益。固定岭回归没有显示出稳定的新旧判断优势，目前不足以支持增加模型复杂度。这里只描述这批比较的结果，不把单一近差距求解器对外推为普遍表现。本结论限于固定样本的岭回归增强方案；成本敏感选例、截断策略和可靠停止尚未检验，因此不构成对整个方向的失败结论。"
    )
    report = section_replace(report, "## 新旧总体成绩比较", "## 矩阵审计与估计定义", overall_section)

    report = report.replace(
        "- 普通样本均值直接对 P∪S 的 192 个已测成绩求均值。其余三种方法先在 P 上冻结预测，再按用户给定的总体残差校正式估计。",
        "- 普通样本均值直接对 P∪S 的 192 个已测成绩求均值。原分组配对、最近历史列和岭回归沿用先导预测加独立校正样本的残差校正式。新增全样本配对估计使用预先固定的最佳历史列和全部已测成绩：`μ_old + mean(y_i - y_old,i)`，其中 `i∈P∪S`；它无需先导拟合。",
    )
    report = report.replace(
        "- `SB01_method_summary.csv`：宏平均误差、误判和相对配对基线的版本数，以及拟合/推理成本。",
        "- `SB01_method_summary.csv`：宏平均误差、误判和相对 paired192 基线的版本数，以及拟合/推理成本。",
    )
    report = report.replace(
        "- `SB01_matrix_audit.csv`：求解器列均分、超时数和排名。",
        "- `SB01_matrix_audit.csv`：求解器列均分、超时数和排名。\n- `SB01_add_paired192.py`：从既有矩阵与抽样清单离线补算强配对基线并更新结果表；不重新抽样或运行求解器。",
    )
    report = report.replace(
        "## 来源与复现信息\n\n",
        "## 来源与复现信息\n\n- 本次强基线补算读取已保存的 20 组 P/S hash 清单，仅对已测 192 个成绩计算均值残差；未重新抽样。\n",
    )
    cost_table = "\n".join(
        [
            "| 估计器 | 固定选择总耗时 (ms) | 先导拟合均值 (ms/次) | R预测均值 (ms/次) | 最终估计更新均值 (ms/次) |",
            "|---|---:|---:|---:|---:|",
        ]
        + [
            f"| {row['label_zh']} | {float(row['fixed_setup_total_ms'] or 0):.3f} | "
            f"{float(row['pilot_fit_mean_ms_per_call'] or 0):.4f} | "
            f"{float(row['prediction_mean_ms_per_call'] or 0):.4f} | "
            f"{float(row['estimate_update_mean_ms_per_call'] or 0):.4f} |"
            for row in summary_rows
        ]
    )
    report = report.replace(
        "以下是本机 NumPy 运算耗时，不含 CSV 读取与独立扰动审计。最佳历史列的全矩阵选择每个隐藏版本只计算一次。",
        "以下是本机 NumPy 运算耗时，不含 CSV 读取与独立扰动审计。最佳历史列全矩阵选择耗时对两种配对方法相同，可在同时使用时共用；表中按单独使用时列示。",
    )
    report = section_replace(
        report,
        "| 估计器 | 固定选择总耗时 (ms)",
        "## 来源与复现信息",
        cost_table,
    )
    report_path.write_text(report, encoding="utf-8")

    print(
        f"Updated SB01 outputs from the saved manifest: pair192 MAE="
        f"{float(paired192['macro_mean_per_version_mae_par2']):.6f}, "
        f"misclassifications={pair192_misses}/560; ridge MAE gain={ridge_mae_gain:.3f}%"
    )


if __name__ == "__main__":
    main()
