from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "sb01"
OUT = ROOT / "outputs"
DATA_FILE = WORK / "anni-seq.csv"
NOTEBOOK_FILE = WORK / "prepare_data.ipynb"

N_PILOT = 64
N_CORRECTION = 128
SEEDS = tuple(range(20))
RIDGE_ALPHA = 1.0
TIMEOUT_SENTINEL = 10_000.0
DATA_REPO_COMMIT = "907cbcdd318d1036ae72ae26f70d0d3f583373a9"
CODE_REPO_COMMIT = "185123c8b57f4d9be65a934eb3a83a0561d1495f"
EXPECTED_DATA_SHA256 = "561e192a585bccc8cd326af34c931e0137fcd6ee0fb8bf1e8a1535aae108b465"
EXPECTED_NOTEBOOK_SHA256 = "c4ee3bedde5aa43a2407a821cb540e222fdf34f4aa16bd42e6b35bd08c9bdb49"
META_COLUMNS = {"hash", "benchmark", "verified-result", "claimed-result"}

METHODS = (
    "ordinary_sample_mean",
    "best_history_corrected",
    "nearest_history_corrected",
    "ridge_alpha1_corrected",
)
METHOD_LABELS = {
    "ordinary_sample_mean": "普通样本均值（P∪S共192次测量）",
    "best_history_corrected": "最佳历史版本 + 残差校正",
    "nearest_history_corrected": "先导最相近历史列 + 残差校正",
    "ridge_alpha1_corrected": "岭回归 α=1 + 残差校正",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify_gap(gap: float) -> str:
    if abs(gap) < 1e-9:
        return "tie"
    return "new_better" if gap < 0 else "old_better"


def ridge_predict(
    history: np.ndarray,
    pilot_indices: np.ndarray,
    y_pilot: np.ndarray,
    remaining_indices: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    fit_start = time.perf_counter_ns()
    x_pilot = history[pilot_indices, :]
    x_mean = x_pilot.mean(axis=0)
    x_scale = x_pilot.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = float(y_pilot.mean())
    y_scale = float(y_pilot.std())
    if y_scale < 1e-12:
        y_scale = 1.0

    x_train = (x_pilot - x_mean) / x_scale
    y_train = (y_pilot - y_mean) / y_scale
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += RIDGE_ALPHA
    beta = np.linalg.solve(gram, x_train.T @ y_train)
    fit_ns = time.perf_counter_ns() - fit_start

    predict_start = time.perf_counter_ns()
    x_remaining = history[remaining_indices, :]
    x_test = (x_remaining - x_mean) / x_scale
    predictions = y_mean + y_scale * (x_test @ beta)
    predict_ns = time.perf_counter_ns() - predict_start
    return predictions, fit_ns, predict_ns


def make_history_predictions(
    history: np.ndarray,
    pilot_indices: np.ndarray,
    remaining_indices: np.ndarray,
    pilot_target: np.ndarray,
    best_old_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
    predictions: dict[str, np.ndarray] = {}
    fit_ns: dict[str, int] = {method: 0 for method in METHODS}
    predict_ns: dict[str, int] = {method: 0 for method in METHODS}

    tic = time.perf_counter_ns()
    predictions["best_history_corrected"] = history[
        remaining_indices, best_old_index
    ].copy()
    predict_ns["best_history_corrected"] = time.perf_counter_ns() - tic

    tic = time.perf_counter_ns()
    pilot_abs_errors = np.mean(
        np.abs(history[pilot_indices, :] - pilot_target[:, None]), axis=0
    )
    nearest_index = int(np.argmin(pilot_abs_errors))
    fit_ns["nearest_history_corrected"] = time.perf_counter_ns() - tic
    tic = time.perf_counter_ns()
    predictions["nearest_history_corrected"] = history[
        remaining_indices, nearest_index
    ].copy()
    predict_ns["nearest_history_corrected"] = time.perf_counter_ns() - tic

    ridge_values, ridge_fit_ns, ridge_predict_ns = ridge_predict(
        history,
        pilot_indices,
        pilot_target,
        remaining_indices,
    )
    predictions["ridge_alpha1_corrected"] = ridge_values
    fit_ns["ridge_alpha1_corrected"] = ridge_fit_ns
    predict_ns["ridge_alpha1_corrected"] = ridge_predict_ns

    return predictions, fit_ns, predict_ns


def calibrated_estimate(
    pilot_target: np.ndarray,
    correction_target: np.ndarray,
    prediction_remaining: np.ndarray,
    correction_positions_in_remaining: np.ndarray,
    population_size: int,
    remaining_size: int,
) -> float:
    residual_total_estimate = (remaining_size / len(correction_target)) * np.sum(
        correction_target - prediction_remaining[correction_positions_in_remaining]
    )
    return float(
        (
            np.sum(pilot_target)
            + np.sum(prediction_remaining)
            + residual_total_estimate
        )
        / population_size
    )


def input_fingerprint(
    history_digest: bytes,
    pilot_indices: np.ndarray,
    remaining_indices: np.ndarray,
    pilot_target: np.ndarray,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(history_digest)
    digest.update(np.ascontiguousarray(pilot_indices, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(remaining_indices, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(pilot_target, dtype=np.float64).tobytes())
    return digest.digest()


def percentile_ms(values_ns: list[int], percentile: float) -> float:
    if not values_ns:
        return 0.0
    return float(np.percentile(np.asarray(values_ns, dtype=np.float64), percentile) / 1e6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256_file(DATA_FILE) != EXPECTED_DATA_SHA256:
        raise ValueError("anni-seq.csv checksum differs from the audited download")
    if sha256_file(NOTEBOOK_FILE) != EXPECTED_NOTEBOOK_SHA256:
        raise ValueError("prepare_data.ipynb checksum differs from the audited download")

    data = pd.read_csv(DATA_FILE, dtype={"hash": "string", "benchmark": "string"})
    solver_columns = [column for column in data.columns if column not in META_COLUMNS]
    if len(data) != 5355 or len(solver_columns) != 28:
        raise ValueError(f"Unexpected matrix shape: {data.shape}, solvers={len(solver_columns)}")
    if data["hash"].isna().any() or data["hash"].nunique() != len(data):
        raise ValueError("Instance hashes are missing or duplicated")

    raw_scores = data[solver_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_scores).all():
        raise ValueError("Solver matrix contains a missing or non-finite entry")
    if raw_scores.min() < 0 or raw_scores.max() > TIMEOUT_SENTINEL:
        raise ValueError("Unexpected runtime value outside [0, 10000]")
    timeout_mask = raw_scores == TIMEOUT_SENTINEL
    scores = raw_scores.copy()
    # The official notebook converts 10000 to a timeout marker, then uses the
    # published PAR-2 penalty of 10000 when calculating the mean score.
    scores[timeout_mask] = TIMEOUT_SENTINEL

    instance_hashes = data["hash"].astype(str).to_numpy()
    population_size, solver_count = scores.shape
    full_means = scores.mean(axis=0)
    full_medians = np.median(scores, axis=0)
    timeout_counts = timeout_mask.sum(axis=0)
    ranks = np.argsort(np.argsort(full_means, kind="stable"), kind="stable") + 1

    matrix_audit_rows = []
    for index, solver in enumerate(solver_columns):
        matrix_audit_rows.append(
            {
                "solver": solver,
                "mean_official_par2_score": float(full_means[index]),
                "median_official_par2_score": float(full_medians[index]),
                "timeout_count": int(timeout_counts[index]),
                "timeout_rate": float(timeout_counts[index] / population_size),
                "full_matrix_rank_lower_is_better": int(ranks[index]),
            }
        )
    write_csv(OUT / "SB01_matrix_audit.csv", matrix_audit_rows)

    rng_by_seed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    manifest_rows = []
    all_indices = list(range(population_size))
    for seed in SEEDS:
        rng = random.Random(seed)
        pilot = rng.sample(all_indices, N_PILOT)
        pilot_set = set(pilot)
        remainder_list = [index for index in all_indices if index not in pilot_set]
        correction = rng.sample(remainder_list, N_CORRECTION)
        pilot_indices = np.asarray(pilot, dtype=np.int32)
        correction_indices = np.asarray(correction, dtype=np.int32)
        remaining_indices = np.asarray(remainder_list, dtype=np.int32)
        if len(set(pilot) & set(correction)) or len(pilot) != N_PILOT:
            raise AssertionError("Pilot and correction sets are not valid samples")
        rng_by_seed[seed] = (pilot_indices, correction_indices, remaining_indices)
        manifest_rows.append(
            {
                "seed": seed,
                "pilot_n": len(pilot),
                "correction_n": len(correction),
                "pilot_hash_ids_in_draw_order": "|".join(instance_hashes[pilot_indices]),
                "correction_hash_ids_in_draw_order": "|".join(instance_hashes[correction_indices]),
            }
        )
    write_csv(OUT / "SB01_sample_manifest.csv", manifest_rows)

    per_seed_rows: list[dict[str, Any]] = []
    timing: dict[str, dict[str, list[int]]] = {
        method: {
            "setup_ns": [],
            "fit_ns": [],
            "predict_ns": [],
            "estimate_ns": [],
        }
        for method in METHODS
    }
    leakage_checks = 0
    leakage_input_changes = 0
    leakage_prediction_changes = 0
    exact_true_ties = 0
    selected_old_by_target: dict[str, tuple[str, float]] = {}

    for target_index, target_solver in enumerate(solver_columns):
        history_indices = [i for i in range(solver_count) if i != target_index]
        history = scores[:, history_indices]
        setup_start = time.perf_counter_ns()
        history_means = history.mean(axis=0)
        best_history_local = int(np.argmin(history_means))
        setup_ns = time.perf_counter_ns() - setup_start
        timing["best_history_corrected"]["setup_ns"].append(int(setup_ns))
        old_solver = solver_columns[history_indices[best_history_local]]
        old_mean = float(history_means[best_history_local])
        target = scores[:, target_index]
        true_mean = float(target.mean())
        true_gap = true_mean - old_mean
        true_class = classify_gap(true_gap)
        exact_true_ties += int(true_class == "tie")
        selected_old_by_target[target_solver] = (old_solver, old_mean)
        history_digest = hashlib.sha256(np.ascontiguousarray(history).tobytes()).digest()

        for seed in SEEDS:
            pilot_indices, correction_indices, remaining_indices = rng_by_seed[seed]
            pilot_target = target[pilot_indices].copy()

            # Freeze all predictions using only historical columns and the 64
            # pilot outcomes. Do not read the held-out column on R until after.
            history_predictions, fit_ns, predict_ns = make_history_predictions(
                history,
                pilot_indices,
                remaining_indices,
                pilot_target,
                best_history_local,
            )
            for method in METHODS:
                if fit_ns[method]:
                    timing[method]["fit_ns"].append(int(fit_ns[method]))
                if predict_ns[method]:
                    timing[method]["predict_ns"].append(int(predict_ns[method]))

            # Counterfactual leakage audit: alter every unrevealed target score
            # while preserving P. Model inputs and frozen predictions must hold.
            perturbed_target = target.copy()
            perturbed_target[remaining_indices] = (
                TIMEOUT_SENTINEL - perturbed_target[remaining_indices]
            )
            perturbed_pilot = perturbed_target[pilot_indices]
            fingerprint_before = input_fingerprint(
                history_digest, pilot_indices, remaining_indices, pilot_target
            )
            fingerprint_after = input_fingerprint(
                history_digest, pilot_indices, remaining_indices, perturbed_pilot
            )
            if fingerprint_before != fingerprint_after:
                leakage_input_changes += 1
            counterfactual_predictions, _, _ = make_history_predictions(
                history,
                pilot_indices,
                remaining_indices,
                perturbed_pilot,
                best_history_local,
            )
            for method in (
                "best_history_corrected",
                "nearest_history_corrected",
                "ridge_alpha1_corrected",
            ):
                leakage_checks += 1
                if not np.array_equal(
                    history_predictions[method], counterfactual_predictions[method]
                ):
                    leakage_prediction_changes += 1

            # Only now reveal the independent correction measurements.
            correction_target = target[correction_indices]
            correction_positions = np.searchsorted(
                remaining_indices, correction_indices
            )
            sample_indices = np.concatenate((pilot_indices, correction_indices))
            method_estimates: dict[str, float] = {}

            tic = time.perf_counter_ns()
            method_estimates["ordinary_sample_mean"] = float(
                target[sample_indices].mean()
            )
            timing["ordinary_sample_mean"]["estimate_ns"].append(
                time.perf_counter_ns() - tic
            )

            for method in (
                "best_history_corrected",
                "nearest_history_corrected",
                "ridge_alpha1_corrected",
            ):
                tic = time.perf_counter_ns()
                estimate = calibrated_estimate(
                    pilot_target,
                    correction_target,
                    history_predictions[method],
                    correction_positions,
                    population_size,
                    len(remaining_indices),
                )
                timing[method]["estimate_ns"].append(time.perf_counter_ns() - tic)
                method_estimates[method] = estimate

            row: dict[str, Any] = {
                "hidden_solver": target_solver,
                "sampling_seed": seed,
                "historical_comparator": old_solver,
                "pilot_n": N_PILOT,
                "correction_n": N_CORRECTION,
                "total_measured_n": N_PILOT + N_CORRECTION,
                "true_mean_par2": true_mean,
                "historical_mean_par2": old_mean,
                "true_new_minus_old_gap": true_gap,
                "true_comparison": true_class,
            }
            for method, estimate in method_estimates.items():
                estimated_gap = estimate - old_mean
                estimated_class = classify_gap(estimated_gap)
                row[f"{method}_estimate"] = estimate
                row[f"{method}_signed_error"] = estimate - true_mean
                row[f"{method}_absolute_error"] = abs(estimate - true_mean)
                row[f"{method}_estimated_new_minus_old_gap"] = estimated_gap
                row[f"{method}_predicted_comparison"] = estimated_class
                row[f"{method}_misclassified"] = estimated_class != true_class
            per_seed_rows.append(row)

    if leakage_checks != solver_count * len(SEEDS) * 3:
        raise AssertionError("Leakage audit did not cover each predictive model per replay")
    if leakage_input_changes or leakage_prediction_changes:
        raise AssertionError(
            f"Leakage audit failed: input={leakage_input_changes}, "
            f"prediction={leakage_prediction_changes}"
        )
    if len(per_seed_rows) != solver_count * len(SEEDS):
        raise AssertionError("Unexpected number of solver-version × seed records")
    write_csv(OUT / "SB01_per_seed_estimates.csv", per_seed_rows)

    per_version_rows = []
    errors_by_method: dict[str, list[list[float]]] = {method: [] for method in METHODS}
    misclass_by_method: dict[str, list[list[bool]]] = {method: [] for method in METHODS}
    version_metric_cache: dict[str, dict[str, dict[str, float]]] = {}

    for target_solver in solver_columns:
        rows = [row for row in per_seed_rows if row["hidden_solver"] == target_solver]
        old_solver, old_mean = selected_old_by_target[target_solver]
        true_mean = float(rows[0]["true_mean_par2"])
        true_gap = float(rows[0]["true_new_minus_old_gap"])
        record: dict[str, Any] = {
            "hidden_solver": target_solver,
            "historical_comparator": old_solver,
            "true_mean_par2": true_mean,
            "historical_mean_par2": old_mean,
            "true_new_minus_old_gap": true_gap,
            "true_relative_gap_pct_of_old": 100.0 * true_gap / old_mean,
            "true_comparison": classify_gap(true_gap),
            "sampling_repeats": len(rows),
        }
        version_metric_cache[target_solver] = {}
        for method in METHODS:
            signed = np.asarray([row[f"{method}_signed_error"] for row in rows])
            absolute = np.abs(signed)
            squared = signed**2
            misclassified = np.asarray(
                [row[f"{method}_misclassified"] for row in rows], dtype=bool
            )
            metric = {
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(squared.mean())),
                "bias": float(signed.mean()),
                "misclassification_rate": float(misclassified.mean()),
            }
            version_metric_cache[target_solver][method] = metric
            errors_by_method[method].append(signed.tolist())
            misclass_by_method[method].append(misclassified.tolist())
            for measure, value in metric.items():
                record[f"{method}_{measure}"] = value
        baseline_mae = version_metric_cache[target_solver]["best_history_corrected"]["mae"]
        baseline_miss = version_metric_cache[target_solver]["best_history_corrected"][
            "misclassification_rate"
        ]
        for method in ("nearest_history_corrected", "ridge_alpha1_corrected"):
            record[f"{method}_mae_improves_vs_paired"] = (
                version_metric_cache[target_solver][method]["mae"] < baseline_mae - 1e-12
            )
            record[f"{method}_misclass_improves_vs_paired"] = (
                version_metric_cache[target_solver][method]["misclassification_rate"]
                < baseline_miss - 1e-12
            )
        per_version_rows.append(record)

    per_version_rows.sort(key=lambda row: abs(row["true_new_minus_old_gap"]))
    write_csv(OUT / "SB01_version_results.csv", per_version_rows)

    confusion_by_method: dict[str, Counter[str]] = {}
    for method in METHODS:
        confusion = Counter()
        for row in per_seed_rows:
            predicted = row[f"{method}_predicted_comparison"]
            true = row["true_comparison"]
            if true == "new_better":
                confusion[
                    "true_new_called_new" if predicted == "new_better" else "true_new_missed"
                ] += 1
            elif true == "old_better":
                if predicted == "old_better":
                    confusion["true_old_called_old"] += 1
                elif predicted == "new_better":
                    confusion["false_new_winner"] += 1
                else:
                    confusion["old_called_tie"] += 1
            elif predicted != "tie":
                confusion["true_tie_misclassified"] += 1
        confusion_by_method[method] = confusion

    summary_rows = []
    paired_version_metrics = version_metric_cache
    for method in METHODS:
        version_metrics = [version_metric_cache[solver][method] for solver in solver_columns]
        per_version_mae = [metric["mae"] for metric in version_metrics]
        per_version_rmse = [metric["rmse"] for metric in version_metrics]
        per_version_bias = [metric["bias"] for metric in version_metrics]
        per_version_misclass = [metric["misclassification_rate"] for metric in version_metrics]
        if method in ("nearest_history_corrected", "ridge_alpha1_corrected"):
            mae_cmp = [
                metric["mae"] - paired_version_metrics[solver]["best_history_corrected"]["mae"]
                for solver, metric in zip(solver_columns, version_metrics)
            ]
            miss_cmp = [
                metric["misclassification_rate"]
                - paired_version_metrics[solver]["best_history_corrected"][
                    "misclassification_rate"
                ]
                for solver, metric in zip(solver_columns, version_metrics)
            ]
            mae_better = int(sum(value < -1e-12 for value in mae_cmp))
            mae_equal = int(sum(abs(value) <= 1e-12 for value in mae_cmp))
            mae_worse = solver_count - mae_better - mae_equal
            misclass_better = int(sum(value < -1e-12 for value in miss_cmp))
            misclass_equal = int(sum(abs(value) <= 1e-12 for value in miss_cmp))
            misclass_worse = solver_count - misclass_better - misclass_equal
        else:
            mae_better = mae_equal = mae_worse = ""
            misclass_better = misclass_equal = misclass_worse = ""

        flat_fit = timing[method]["fit_ns"]
        flat_predict = timing[method]["predict_ns"]
        flat_estimate = timing[method]["estimate_ns"]
        fixed_setup = timing[method]["setup_ns"]
        summary_rows.append(
            {
                "estimator": method,
                "label_zh": METHOD_LABELS[method],
                "macro_mean_per_version_mae_par2": float(np.mean(per_version_mae)),
                "macro_mean_per_version_rmse_par2": float(np.mean(per_version_rmse)),
                "macro_mean_per_version_signed_bias_par2": float(np.mean(per_version_bias)),
                "macro_misclassification_rate": float(np.mean(per_version_misclass)),
                "misclassification_versions_pct": 100.0 * float(np.mean(per_version_misclass)),
                "true_new_better_called_new_count": confusion_by_method[method][
                    "true_new_called_new"
                ],
                "true_new_better_missed_count": confusion_by_method[method][
                    "true_new_missed"
                ],
                "true_old_better_called_old_count": confusion_by_method[method][
                    "true_old_called_old"
                ],
                "false_new_winner_count": confusion_by_method[method]["false_new_winner"],
                "true_old_better_predicted_tie_count": confusion_by_method[method][
                    "old_called_tie"
                ],
                "hidden_versions_with_lower_mae_vs_paired": mae_better,
                "hidden_versions_equal_mae_vs_paired": mae_equal,
                "hidden_versions_with_higher_mae_vs_paired": mae_worse,
                "hidden_versions_with_lower_misclass_vs_paired": misclass_better,
                "hidden_versions_equal_misclass_vs_paired": misclass_equal,
                "hidden_versions_with_higher_misclass_vs_paired": misclass_worse,
                "fixed_setup_total_ms": float(sum(fixed_setup) / 1e6),
                "pilot_fit_calls": len(flat_fit),
                "pilot_fit_total_ms": float(sum(flat_fit) / 1e6),
                "pilot_fit_mean_ms_per_call": float(np.mean(flat_fit) / 1e6)
                if flat_fit
                else 0.0,
                "pilot_fit_p95_ms_per_call": percentile_ms(flat_fit, 95),
                "prediction_calls": len(flat_predict),
                "prediction_total_ms": float(sum(flat_predict) / 1e6),
                "prediction_mean_ms_per_call": float(np.mean(flat_predict) / 1e6)
                if flat_predict
                else 0.0,
                "prediction_p95_ms_per_call": percentile_ms(flat_predict, 95),
                "estimate_update_total_ms": float(sum(flat_estimate) / 1e6),
                "estimate_update_mean_ms_per_call": float(np.mean(flat_estimate) / 1e6)
                if flat_estimate
                else 0.0,
                "measurement_values_per_version_seed": N_PILOT + N_CORRECTION,
            }
        )
    write_csv(OUT / "SB01_method_summary.csv", summary_rows)

    new_better_versions = sum(
        classify_gap(row["true_new_minus_old_gap"]) == "new_better"
        for row in per_version_rows
    )
    old_better_versions = sum(
        classify_gap(row["true_new_minus_old_gap"]) == "old_better"
        for row in per_version_rows
    )
    if exact_true_ties:
        tie_text = f"有 {exact_true_ties} 个隐藏版本与历史比较器总体均分相同。"
    else:
        tie_text = "28 个隐藏版本与各自历史比较器均无完全相同的总体均分。"

    paired = summary_rows[METHODS.index("best_history_corrected")]
    nearest = summary_rows[METHODS.index("nearest_history_corrected")]
    ridge = summary_rows[METHODS.index("ridge_alpha1_corrected")]
    closest_versions = per_version_rows[:2]
    closest_pair_miss_pct = 100 * float(
        np.mean([row["best_history_corrected_misclassification_rate"] for row in closest_versions])
    )
    closest_ridge_miss_pct = 100 * float(
        np.mean([row["ridge_alpha1_corrected_misclassification_rate"] for row in closest_versions])
    )
    closest_versions = per_version_rows[:2]
    if (
        ridge["hidden_versions_with_lower_mae_vs_paired"] >= 5
        and ridge["hidden_versions_with_lower_misclass_vs_paired"] >= 5
        and ridge["macro_mean_per_version_mae_par2"]
        < paired["macro_mean_per_version_mae_par2"]
        and ridge["macro_misclassification_rate"]
        < paired["macro_misclassification_rate"]
    ):
        conclusion = (
            f"岭回归的版本级 MAE 在 {ridge['hidden_versions_with_lower_mae_vs_paired']}/28 个版本更低，"
            f"宏平均由 {paired['macro_mean_per_version_mae_par2']:.2f} 降至 "
            f"{ridge['macro_mean_per_version_mae_par2']:.2f}（下降 "
            f"{100 * (paired['macro_mean_per_version_mae_par2'] - ridge['macro_mean_per_version_mae_par2']) / paired['macro_mean_per_version_mae_par2']:.1f}%）。"
            f"新旧误判总数只由 {560 * paired['macro_misclassification_rate']:.0f} 降至 "
            f"{560 * ridge['macro_misclassification_rate']:.0f}/560：误报新版本更优由 "
            f"{paired['false_new_winner_count']} 降至 {ridge['false_new_winner_count']}，"
            f"但唯一真实更优版本的正确识别由 {paired['true_new_better_called_new_count']} 降至 "
            f"{ridge['true_new_better_called_new_count']}/20。最接近两组比较的平均误判率也从 "
            f"{closest_pair_miss_pct:.1f}% 升至 {closest_ridge_miss_pct:.1f}%。"
            "因此只有有限的总体估计精度信号，没有稳健的近差距判断优势；这不构成增加网络的充分依据，"
            "也不提供置信保证或可兑现的评测省时。"
        )
    elif (
        nearest["hidden_versions_with_lower_mae_vs_paired"] >= 5
        and nearest["hidden_versions_with_lower_misclass_vs_paired"] >= 5
        and nearest["macro_mean_per_version_mae_par2"]
        < paired["macro_mean_per_version_mae_par2"]
        and nearest["macro_misclassification_rate"]
        < paired["macro_misclassification_rate"]
    ):
        conclusion = (
            "最近历史列在多个版本上同时降低估计误差和比较误判，而岭回归没有建立相同优势；"
            "有限信号主要支持廉价历史匹配，不支持增加网络。"
        )
    elif (
        ridge["macro_mean_per_version_mae_par2"]
        < paired["macro_mean_per_version_mae_par2"]
        and ridge["macro_misclassification_rate"]
        >= paired["macro_misclassification_rate"]
    ):
        conclusion = (
            "岭回归降低了平均估计误差，但没有降低新旧比较误判；按本轮预先声明的判断目标，"
            "这只是有限的精度信号，不能据此声称决策改善。"
        )
    else:
        conclusion = (
            "没有看到历史匹配或岭回归在多个隐藏版本上稳定超过配对估计并同时改善新旧比较的信号；"
            "本轮不支持增加网络。"
        )

    matrix_cells = int(scores.size)
    timeout_cells = int(timeout_mask.sum())
    unique_benchmark_names = int(data["benchmark"].nunique())
    data_sha = sha256_file(DATA_FILE)
    notebook_sha = sha256_file(NOTEBOOK_FILE)
    system = (
        f"{platform.platform()} | {platform.processor()} | Python {platform.python_version()} "
        f"| NumPy {np.__version__}"
    )
    report_lines = [
        "# SB01：SAT 求解器历史信息回放实验",
        "",
        "## 结果概览",
        "",
        f"使用公开的 {population_size:,}×{solver_count} 性能矩阵，隐藏每个求解器列依次作为新版本，共 {solver_count} 个留出版本；"
        f"每个版本使用 {len(SEEDS)} 个固定抽样种子。每个种子从同一组 {N_PILOT} 个先导实例与 "
        f"{N_CORRECTION} 个校正实例估计四种方法。未运行 SAT 求解器。",
        "",
        "总体误差和误判率先在每个隐藏版本的 20 个抽样种子内汇总，再对 28 个版本等权宏平均。"
        "抽样重复不是独立新版本，也没有据此计算置信保证或显著性。",
        "",
        "| 估计器 | 版本宏平均绝对误差（PAR-2秒） | 版本宏平均RMSE | 版本宏平均偏差 | 新旧误判率 | 相对配对估计：MAE 更低 / 相同 / 更高版本 | 相对配对估计：误判更低 / 相同 / 更高版本 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        mae_wins = row["hidden_versions_with_lower_mae_vs_paired"]
        if mae_wins == "":
            mae_relation = "—"
            miss_relation = "—"
        else:
            mae_relation = (
                f"{mae_wins} / {row['hidden_versions_equal_mae_vs_paired']} / "
                f"{row['hidden_versions_with_higher_mae_vs_paired']}"
            )
            miss_relation = (
                f"{row['hidden_versions_with_lower_misclass_vs_paired']} / "
                f"{row['hidden_versions_equal_misclass_vs_paired']} / "
                f"{row['hidden_versions_with_higher_misclass_vs_paired']}"
            )
        report_lines.append(
            f"| {row['label_zh']} | {row['macro_mean_per_version_mae_par2']:.2f} | "
            f"{row['macro_mean_per_version_rmse_par2']:.2f} | "
            f"{row['macro_mean_per_version_signed_bias_par2']:.2f} | "
            f"{row['misclassification_versions_pct']:.1f}% | {mae_relation} | {miss_relation} |"
        )
    report_lines.extend(
        [
            "",
            "版本级相对基线的胜/平/负数只统计 28 个隐藏版本，不把抽样种子当作独立新版本。",
            "",
            "### 新旧判断误报与漏报",
            "",
            "下表计数覆盖每个隐藏版本的 20 个配对抽样重放；重放数不代表独立新版本数。",
            "",
            "| 估计器 | 真正新版本更优：判对 / 漏报 | 旧版本更优：判对 / 误报新版本更优 / 判平局 |",
            "|---|---:|---:|",
            *[
                f"| {row['label_zh']} | {row['true_new_better_called_new_count']} / "
                f"{row['true_new_better_missed_count']} | "
                f"{row['true_old_better_called_old_count']} / {row['false_new_winner_count']} / "
                f"{row['true_old_better_predicted_tie_count']} |"
                for row in summary_rows
            ],
            "",
            "## 新旧总体成绩比较",
            "",
            f"28 个隐藏版本中，{new_better_versions} 个真实总体 PAR-2 均分优于本折最佳历史列，"
            f"{old_better_versions} 个更差。{tie_text} 真实差距及每种方法按版本统计的误差、误判率见版本表；"
            "表按真实差距绝对值排序，便于检查接近的比较。",
            "",
            conclusion,
            "",
            "最接近的两组真实比较如下。负 gap 表示新版本更快，正 gap 表示新版本更慢。",
            "",
            "| 隐藏新版本 | 最佳历史列 | 真实差距（PAR-2秒） | 普通均值误判率 | 配对误判率 | 最近历史误判率 | 岭回归误判率 |",
            "|---|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row['hidden_solver']} | {row['historical_comparator']} | "
                f"{row['true_new_minus_old_gap']:.2f} | "
                f"{100 * row['ordinary_sample_mean_misclassification_rate']:.1f}% | "
                f"{100 * row['best_history_corrected_misclassification_rate']:.1f}% | "
                f"{100 * row['nearest_history_corrected_misclassification_rate']:.1f}% | "
                f"{100 * row['ridge_alpha1_corrected_misclassification_rate']:.1f}% |"
                for row in closest_versions
            ],
            "",
            "## 矩阵审计与估计定义",
            "",
            f"- 原始 CSV 为 {DATA_FILE.stat().st_size:,} 字节（约 {DATA_FILE.stat().st_size / 1e6:.2f} MB）；"
            f"{population_size:,} 行、{solver_count} 个求解器列、{matrix_cells:,} 个成绩单元。"
            f"实例 hash 唯一；benchmark 字符串有 {population_size - unique_benchmark_names} 个重复名，"
            "因此用 hash 标识抽样行。成绩没有空值，范围为 0 至 10,000。",
            f"- 矩阵含 {timeout_cells:,} 个 10,000 sentinel（{100 * timeout_cells / matrix_cells:.2f}%）。"
            "官方预处理将其转为 timeout 标记，并在名为 PAR-2 的平均评分中按 10,000 计分；"
            "SB01 不把 sentinel 当作一次真实耗时，也没有另乘 2。",
            "- 每折的历史比较器是在其余 27 列上，使用全 5,355 行均分最低者。"
            "该历史成绩对所有方法已知，不读取留出的新版本整列。",
            "- 普通样本均值直接对 P∪S 的 192 个已测成绩求均值。其余三种方法先在 P 上冻结预测，"
            "再按用户给定的总体残差校正式估计。",
            "- 最近历史列以 P 上的平均绝对误差选择；岭回归使用 27 个历史成绩列，"
            "按 P 上列均值/标准差标准化输入与目标，固定 α=1.0，之后预测 R 并进行残差校正。"
            "不调参、不裁剪预测值。",
            "- 模型输入仅含 27 列历史成绩和 64 个新版本先导成绩。benchmark 名称、实例 hash、"
            "SAT/UNSAT/unknown 标签和新版本未揭示成绩均未作为模型特征。",
            f"- 同一 seed 的 P 与 S 在所有留出列和估计器之间共用；每轮抽样无放回，P∩S为空。"
            f"每个版本/种子/方法读取相同的 {N_PILOT + N_CORRECTION} 个成绩。",
            f"- 未揭示成绩扰动审计共 {leakage_checks:,} 次预测对照（3 个预测方法×28 列×20 种子）；"
            f"模型输入变化 {leakage_input_changes} 次，预测变化 {leakage_prediction_changes} 次。",
            "- 每版本×种子测量成本固定为 192 个矩阵成绩。拟合、预测与最终校正计算的本机耗时分列记录；"
            "未把模型计算时间换算成 SAT 求解器评测省时。",
            "- 公开数据仓库的 README 标注 CC BY 4.0；本报告保留来源、提交号与校验值，复用数据时应注明来源。",
            "",
            "## 拟合与推理成本",
            "",
            "以下是本机 NumPy 运算耗时，不含 CSV 读取与独立扰动审计。最佳历史列的全矩阵选择每个隐藏版本只计算一次。",
            "",
            "| 估计器 | 固定选择总耗时 (ms) | 先导拟合均值 (ms/次) | R预测均值 (ms/次) | 最终估计更新均值 (ms/次) |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row['label_zh']} | {row['fixed_setup_total_ms']:.3f} | "
                f"{row['pilot_fit_mean_ms_per_call']:.4f} | "
                f"{row['prediction_mean_ms_per_call']:.4f} | "
                f"{row['estimate_update_mean_ms_per_call']:.4f} |"
                for row in summary_rows
            ],
            "",
            "## 来源与复现信息",
            "",
            "- [数据仓库](https://github.com/mathefuchs/al-for-sat-solver-benchmarking-data)，"
            "[官方预处理笔记本](https://github.com/mathefuchs/al-for-sat-solver-benchmarking/blob/main/prepare_data.ipynb)，"
            "[CC BY 4.0 许可](https://github.com/mathefuchs/al-for-sat-solver-benchmarking-data/blob/main/LICENSE)。",
            f"- 数据仓库 HEAD：`{DATA_REPO_COMMIT}`。CSV SHA-256：`{data_sha}`。",
            f"- 官方预处理仓库 HEAD：`{CODE_REPO_COMMIT}`。笔记本 SHA-256：`{notebook_sha}`。",
            "- 系统：`" + system + "`。NumPy 原生实现岭回归，没有安装额外依赖。",
            "- 固定抽样种子为 0–19。完整 P/S 实例 hash 抽样清单见 `SB01_sample_manifest.csv`。",
            "- 每求解器的全矩阵均分与 timeout 数见 `SB01_matrix_audit.csv`。",
            "",
            "## 交付文件",
            "",
            "- `SB01_method_summary.csv`：宏平均误差、误判和相对配对基线的版本数，以及拟合/推理成本。",
            "- `SB01_version_results.csv`：28 个隐藏版本的真实新旧差距和版本级表现，按差距绝对值排序。",
            "- `SB01_per_seed_estimates.csv`：全部 560 个版本×种子单元的估计、误差和新旧判断。",
            "- `SB01_sample_manifest.csv`：每个种子固定的 P 与 S hash 清单。",
            "- `SB01_matrix_audit.csv`：求解器列均分、超时数和排名。",
        ]
    )
    (OUT / "SB01_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "matrix_shape": [population_size, solver_count],
                "timeout_cells": timeout_cells,
                "sampling_replays": len(per_seed_rows),
                "leakage_checks": leakage_checks,
                "leakage_input_changes": leakage_input_changes,
                "leakage_prediction_changes": leakage_prediction_changes,
                "macro_results": summary_rows,
                "output_dir": str(OUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
