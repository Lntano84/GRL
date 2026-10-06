"""Offline-only BC08 RECENCY correction; no fitting, replay, or relabeling."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs"
PREDICTIONS = OUT / "BC08_eval_request_predictions.csv.gz"
METRICS = OUT / "BC08_top1pct_metrics.csv"
AUDIT = OUT / "BC08_audit.json"
REPORT = OUT / "BC08_report.md"
PROTOCOL = json.loads((OUT / "BC08_protocol.json").read_text(encoding="utf-8"))
Q = float(PROTOCOL["top_selection_fraction"])
TIE_VERSION = PROTOCOL["tie_break"]["version"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tie(seq):
    return hashlib.sha256(f"{TIE_VERSION}:{seq}".encode("utf-8")).hexdigest()


def read_predictions():
    rows = []
    with gzip.open(PREDICTIONS, "rt", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            gap_text = raw.get("seconds_since_block_last_get_s", "")
            try:
                gap = float(gap_text)
            except (TypeError, ValueError):
                gap = math.nan
            row = {
                **raw,
                "_seq": int(raw["request_sequence_audit_only"]),
                "_elapsed": float(raw["elapsed_s"]),
                "_chunks": int(raw["candidate_chunk_count"]),
                "_hits": int(raw["future_repeat_chunk_count"]),
                "_gap": gap,
                "_finite_gap": math.isfinite(gap),
                "_half": raw["evaluation_half"],
            }
            rows.append(row)
    if not rows:
        raise AssertionError("empty frozen BC08 evaluation request table")
    return rows


def select_recency(rows):
    pool_chunks = sum(r["_chunks"] for r in rows)
    target = math.ceil(Q * pool_chunks)
    ordered = sorted(rows, key=lambda r: (
        not r["_finite_gap"],
        r["_gap"] if r["_finite_gap"] else math.inf,
        tie(r["_seq"]),
    ))
    chosen = []
    selected_chunks = 0
    for row in ordered:
        before = abs(target - selected_chunks)
        after = abs(target - (selected_chunks + row["_chunks"]))
        if after < before or (after == before and selected_chunks < target):
            chosen.append(row)
            selected_chunks += row["_chunks"]
        else:
            break
        if selected_chunks >= target:
            break
    hits = sum(r["_hits"] for r in chosen)
    total_hits = sum(r["_hits"] for r in rows)
    yield_rate = hits / selected_chunks if selected_chunks else 0.0
    return {
        "method": "RECENCY",
        "candidate_pool_requests": len(rows),
        "candidate_pool_chunks": pool_chunks,
        "candidate_pool_future_repeat_chunks": total_hits,
        "target_selected_chunks": target,
        "selected_requests": len(chosen),
        "selected_chunks": selected_chunks,
        "selected_fraction_pct": 100.0 * selected_chunks / pool_chunks if pool_chunks else 0.0,
        "selection_boundary_delta_chunks": selected_chunks - target,
        "selected_future_repeat_chunks": hits,
        "selected_repeat_yield_pct": 100.0 * yield_rate,
        "scaled_repeat_capture_at_exact_target_chunks": yield_rate * target,
        "fraction_of_pool_future_repeats_captured_pct": 100.0 * hits / total_hits if total_hits else math.nan,
        "selected_request_sequences": [r["_seq"] for r in chosen],
        "selected_nonfinite_gap_requests": sum(not r["_finite_gap"] for r in chosen),
    }


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main():
    before_prediction_sha = sha(PREDICTIONS)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") == "completed_with_offline_recency_correction":
        raise SystemExit("BC08 RECENCY correction is already applied; refusing to overwrite its original audit trail.")
    rows = read_predictions()
    old_all = [r for r in rows if as_bool(r.get("selected_recency_eval_all", False))]
    old_missing_count = sum(not r["_finite_gap"] for r in old_all)
    old_metrics = {}
    old_audit = {r["slice"]: r for r in audit["selection"]["all_and_half_results"] if r["method"] == "RECENCY"}

    slices = {
        "EVAL-ALL": rows,
        "EVAL-FIRST-HALF": [r for r in rows if r["_half"] == "first"],
        "EVAL-SECOND-HALF": [r for r in rows if r["_half"] == "second"],
    }
    corrected = {name: select_recency(pool) for name, pool in slices.items()}
    expected = {"EVAL-ALL": 6074.12, "EVAL-FIRST-HALF": 2734.00, "EVAL-SECOND-HALF": 3324.08}
    for name, result in corrected.items():
        value = result["scaled_repeat_capture_at_exact_target_chunks"]
        if abs(value - expected[name]) > 0.02:
            raise AssertionError(("corrected RECENCY result differs from independent audit", name, value, expected[name]))
        # The original audit persisted aggregate selection metrics, not selected sequence IDs
        # for each slice. Preserve those audited aggregates instead of reconstructing a different
        # pre-correction half selection from the global marker.
        old_metrics[name] = {
            field: old_audit[name][field]
            for field in (
                "selected_requests", "selected_chunks", "selected_future_repeat_chunks",
                "scaled_repeat_capture_at_exact_target_chunks", "selected_fraction_pct",
            )
        }
        if name == "EVAL-ALL":
            old_metrics[name]["selected_nonfinite_gap_requests"] = old_missing_count

    # Rebuild the selection flags from the corrected complete-request selections.
    selected_all = set(corrected["EVAL-ALL"]["selected_request_sequences"])
    selected_first = set(corrected["EVAL-FIRST-HALF"]["selected_request_sequences"])
    selected_second = set(corrected["EVAL-SECOND-HALF"]["selected_request_sequences"])
    for r in rows:
        r["selected_recency_eval_all"] = str(r["_seq"] in selected_all)
        r["selected_recency_eval_first_half"] = str(r["_seq"] in selected_first)
        r["selected_recency_eval_second_half"] = str(r["_seq"] in selected_second)
    pred_fields = [k for k in rows[0].keys() if not k.startswith("_")]
    with gzip.open(PREDICTIONS, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with METRICS.open("r", encoding="utf-8-sig", newline="") as f:
        metric_rows = list(csv.DictReader(f))
        metric_fields = list(metric_rows[0])
    original_new_gains = {
        r["slice"]: r.get("new_gain_vs_strongest_other_pct")
        for r in audit["selection"]["all_and_half_results"] if r["method"] == "NEW"
    }
    columns = [
        "candidate_pool_requests", "candidate_pool_chunks", "candidate_pool_future_repeat_chunks",
        "target_selected_chunks", "selected_requests", "selected_chunks", "selected_fraction_pct",
        "selection_boundary_delta_chunks", "selected_future_repeat_chunks", "selected_repeat_yield_pct",
        "scaled_repeat_capture_at_exact_target_chunks", "fraction_of_pool_future_repeats_captured_pct",
    ]
    for result in corrected.values():
        slice_name = next(s for s, r in corrected.items() if r is result)
        for oldrow in audit["selection"]["all_and_half_results"]:
            if oldrow["slice"] == slice_name and oldrow["method"] == "RECENCY":
                for field in columns:
                    oldrow[field] = result[field]
                oldrow["selected_request_sequences"] = result["selected_request_sequences"]
                oldrow["selected_nonfinite_gap_requests"] = result["selected_nonfinite_gap_requests"]
        for row in metric_rows:
            if row["slice"] == slice_name and row["method"] == "RECENCY":
                for field in columns:
                    row[field] = result[field]

    # Comparisons are recomputed from the corrected rows; OLD/NEW fits and predictions remain untouched.
    reversals = {}
    for slice_name in slices:
        group = {r["method"]: r for r in audit["selection"]["all_and_half_results"] if r["slice"] == slice_name}
        others = ["AUTHOR-SCORE", "RECENCY", "OLD"]
        strongest = max(others, key=lambda m: group[m]["scaled_repeat_capture_at_exact_target_chunks"])
        strongest_raw = max(others, key=lambda m: group[m]["selected_future_repeat_chunks"])
        best_value = group[strongest]["scaled_repeat_capture_at_exact_target_chunks"]
        new_value = group["NEW"]["scaled_repeat_capture_at_exact_target_chunks"]
        gain = (new_value / best_value - 1.0) * 100.0 if best_value else None
        if gain is None or abs(gain - float(original_new_gains[slice_name])) > 1e-9:
            raise AssertionError(("correcting RECENCY unexpectedly changed the NEW comparison", slice_name, gain, original_new_gains[slice_name]))
        raw_win = group["NEW"]["selected_future_repeat_chunks"] > group[strongest_raw]["selected_future_repeat_chunks"]
        norm_win = new_value > best_value
        reversals[slice_name] = raw_win != norm_win
        for target_row in group.values():
            target_row["strongest_other_by_normalized_capture"] = strongest
            target_row["new_gain_vs_strongest_other_pct"] = gain
            target_row["strongest_other_by_raw_capture"] = strongest_raw
            target_row["new_beats_strongest_other_raw_capture"] = raw_win
    audit["selection"]["raw_vs_normalized_order_reversal_by_slice"] = reversals
    audit["status"] = "completed_with_offline_recency_correction"
    audit["protocol_passed"] = False
    audit["protocol_scope"] = {
        "future_labels_and_maturity_checks_passed": True,
        "temporal_split_passed": True,
        "old_new_main_comparison_valid": True,
        "recency_original_sort_check_passed": False,
        "recency_sort_corrected_offline": True,
    }
    audit["posthoc_correction"] = {
        "issue": "missing block access gaps were converted to NaN, but the original RECENCY key checked only `is None`, so NaN values were not sorted after finite gaps",
        "old_selected_missing_gap_requests": old_missing_count,
        "old_recency_metrics_by_slice": old_metrics,
        "corrected_recency_metrics_by_slice": corrected,
        "correction_source": "BC08_eval_request_predictions.csv.gz frozen labels/features/scores; no BC05 sidecar",
        "models_refit": False,
        "cache_replay_run": False,
        "labels_changed": False,
        "recency_sort_uses_math_isfinite": True,
        "prediction_file_sha256_before": before_prediction_sha,
    }
    audit["selection"]["raw_vs_normalized_order_reversal_by_slice"] = reversals
    with METRICS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)

    audit["posthoc_correction"]["prediction_file_sha256_after"] = sha(PREDICTIONS)
    audit["posthoc_correction"]["correction_script_sha256"] = sha(Path(__file__))
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    report = REPORT.read_text(encoding="utf-8")
    first_blank = report.find("\n\n")
    report = report[:first_blank] + (
        "\n\n标签、时序成熟和 OLD/NEW 主比较通过；后验审计发现 RECENCY 排序缺陷，已从封存预测表离线修正。"
        "修正只更改 RECENCY 选择、相关指标与审计说明，没有改标签、模型或其他方法结果。"
    ) + report[first_blank:]
    report = report.replace(
        "无历史间隔排在有历史间隔后。",
        "有限访问间隔升序；非有限或缺失间隔排在所有有限值之后，不编码成零。",
    )
    for slice_name, result in corrected.items():
        label = slice_name
        needle_prefix = f"| {label} | RECENCY |"
        old_line = next((line for line in report.splitlines() if line.startswith(needle_prefix)), None)
        if old_line is None:
            raise AssertionError(("RECENCY report row not found", label))
        new_line = (
            f"| {label} | RECENCY | {result['candidate_pool_requests']:,} | {result['candidate_pool_chunks']:,} | "
            f"{result['selected_requests']:,} | {result['selected_chunks']:,} | {result['selected_fraction_pct']:.4f}% | "
            f"{result['selected_future_repeat_chunks']:,} | {result['selected_repeat_yield_pct']:.4f}% | "
            f"{result['scaled_repeat_capture_at_exact_target_chunks']:.2f} | — |"
        )
        report = report.replace(old_line, new_line)
    audit_line = (
        f"- 后验审计发现原 RECENCY 实现用 `is None` 检查访问间隔，但缺失值已转为 NaN；原整体选择中有 {old_missing_count} 个缺失间隔请求未被排到末尾。现已按有限性重排并更新整体／半段选择标记，未重新拟合或回放。修正后 RECENCY 归一化捕捉量为 {corrected['EVAL-ALL']['scaled_repeat_capture_at_exact_target_chunks']:.2f} / {corrected['EVAL-FIRST-HALF']['scaled_repeat_capture_at_exact_target_chunks']:.2f} / {corrected['EVAL-SECOND-HALF']['scaled_repeat_capture_at_exact_target_chunks']:.2f}。"
    )
    report = report.replace(
        "- 作者输入向量与打分输入逐候选一致；R_q 内特征均值聚合；训练和评价样本按请求分组，没有拆分同一请求的 chunk。LightGBM 固定种子和单线程，OLD/NEW 都仅拟合一次。",
        "- 作者输入向量与打分输入逐候选一致；R_q 内特征均值聚合；训练和评价样本按请求分组，没有拆分同一请求的 chunk。LightGBM 固定种子和单线程，OLD/NEW 都仅拟合一次。\n" + audit_line +
        "\n- 固定时域内再次 GET 是重用代理，不是降峰的严格必要条件；该标签不刻画重用时的负载、数据能否驻留、预取成本或挤出其他数据的代价。请求等权 L2 拟合目标也与按 chunk 预算选择 Top 1% 不完全一致。"
    )
    REPORT.write_text(report, encoding="utf-8")

    print(json.dumps({"status": audit["status"], "old_missing_gap_requests": old_missing_count,
                      "corrected_recency": {k: {"selected_requests": v["selected_requests"],
                                                 "selected_chunks": v["selected_chunks"],
                                                 "future_repeat_chunks": v["selected_future_repeat_chunks"],
                                                 "normalized_capture": v["scaled_repeat_capture_at_exact_target_chunks"]}
                                            for k, v in corrected.items()},
                      "protocol_scope": audit["protocol_scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
