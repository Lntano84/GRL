"""BC08: request-level next-GET reuse ranking using a frozen temporal split.

The script reads only the frozen BC06 candidate/context snapshots, their schema,
and the original Region1 trace for its observation endpoint. It never reads the
BC05 intervention sidecar or outcomes.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = OUT / "BC06_candidate_decisions.jsonl.gz"
CONTEXTS = OUT / "BC06_get_request_contexts.jsonl.gz"
SCHEMA = OUT / "BC06_feature_schema.json"
MANIFEST = OUT / "BC06_snapshot_manifest.json"
TRACE = ROOT / "work" / "bc01" / "Baleen-FAST24" / "data" / "tectonic" / "201910" / "Region1" / "full_0_0.1.trace"
AUTHOR_MODEL = ROOT / "work" / "bc01" / "Baleen-FAST24" / "tmp" / "example" / "201910_Region1_0_0.1" / "ea_5892.86_wr_35.599_admit_threshold_binary.model"

HORIZON_S = 5892.856
EVAL_START_S = 86401.23277902603
HIGH_THRESHOLD = 0.8879900141503271
TOP_FRACTION = 0.01
SEED = 20260930
TIE_VERSION = "BC08-request-tiebreak-v1"

LOAD_METRICS = [
    ("total_get_dt_pct", "total_get_dt_pct"),
    ("demand_read_dt_pct", "demand_read_dt_pct"),
    ("extra_prefetch_dt_pct", "extra_prefetch_dt_pct"),
    ("put_dt_pct", "put_dt_pct"),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_number(value):
    if value is None:
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


def trace_bounds(path: Path):
    first_ts = None
    last_ts = None
    previous = None
    line_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise AssertionError(f"trace row has fewer than four fields at line {line_count + 1}")
            ts = float(fields[3])
            if previous is not None and ts < previous:
                raise AssertionError(f"trace timestamp regressed at line {line_count + 1}")
            first_ts = ts if first_ts is None else first_ts
            previous = ts
            last_ts = ts
            line_count += 1
    if not line_count:
        raise AssertionError("empty Region1 trace")
    return {"start_physical_s": first_ts, "end_physical_s": last_ts,
            "duration_s": last_ts - first_ts, "rows": line_count}


def history_features(ctx):
    last = ctx.get("last_completed_window") or {}
    six = ctx.get("last_six_completed_windows") or []
    result = {
        "current_request_chunk_count": as_number(ctx.get("current_request_chunk_count")),
        "cache_occupancy_fraction_at_request_start": as_number(ctx.get("cache_occupancy_fraction_at_request_start")),
        "last_total_get_dt_pct": as_number(last.get("total_get_dt_pct")),
        "last_demand_read_dt_pct": as_number(last.get("demand_read_dt_pct")),
        "last_extra_prefetch_dt_pct": as_number(last.get("extra_prefetch_dt_pct")),
        "last_put_dt_pct": as_number(last.get("put_dt_pct")),
        "last_writes": as_number(last.get("writes")),
        "last_evictions": as_number(last.get("evictions")),
    }
    for source, _ in LOAD_METRICS:
        values = [as_number(w.get(source)) for w in six]
        values = [v for v in values if math.isfinite(v)]
        result[f"last6_mean_{source}"] = float(np.mean(values)) if values else math.nan
        result[f"last6_max_{source}"] = float(np.max(values)) if values else math.nan
    return result


def request_tiebreak(seq: int) -> str:
    return hashlib.sha256(f"{TIE_VERSION}:{seq}".encode("utf-8")).hexdigest()


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for output_path, manifest_key in [(CANDIDATES, "candidate_decisions.jsonl.gz"),
                                      (CONTEXTS, "get_request_contexts.jsonl.gz")]:
        info = manifest["files"][manifest_key]
        if sha(output_path) != info["sha256"] or output_path.stat().st_size != info["bytes"]:
            raise AssertionError(f"BC06 frozen file mismatch: {manifest_key}")
    return manifest


def context_features(raw):
    return history_features(raw)


def empty_group(seq, elapsed, block, ctx_features, vector):
    return {
        "request_sequence": int(seq),
        "elapsed_s": float(elapsed),
        "block_key_for_join_only": str(block),
        "candidate_chunk_ids_for_join_only": [],
        "seen_candidate_chunk_ids_for_join_only": set(),
        "candidate_count": 0,
        "author_vector_sum": np.zeros(18, dtype=np.float64),
        "author_score_sum": 0.0,
        "block_get_count": None,
        "block_get_gap_s": None,
        "chunk_get_count_sum": 0.0,
        "chunk_get_gap_sum": 0.0,
        "chunk_get_gap_known": 0,
        "chunk_get_gap_missing": 0,
        "reject_count_sum": 0.0,
        "reject_gap_sum": 0.0,
        "reject_gap_known": 0,
        "reject_gap_missing": 0,
        "ctx_features": ctx_features,
        "first_vector_width": len(vector),
    }


def append_candidate(group, row):
    history = row.get("legal_key_history_at_decision_request_start") or {}
    vector = row.get("author_model_input_vector")
    if not isinstance(vector, list) or len(vector) != 18:
        raise AssertionError(("unexpected author vector", row.get("candidate_id"), len(vector or [])))
    if row.get("candidate_model_input_vector") != row.get("author_model_input_vector"):
        raise AssertionError(("insert and scorer vectors differ", row.get("candidate_id")))
    chunk = int(row["chunk_id"])
    if chunk in group["seen_candidate_chunk_ids_for_join_only"]:
        raise AssertionError(("duplicate candidate chunk within request", row["actual_decision_request_sequence"], chunk))
    group["seen_candidate_chunk_ids_for_join_only"].add(chunk)
    group["candidate_chunk_ids_for_join_only"].append(chunk)
    group["author_vector_sum"] += np.asarray(vector, dtype=np.float64)
    group["author_score_sum"] += float(row["author_model_score"])
    group["candidate_count"] += 1

    block_count = int(history["prior_block_get_count"])
    block_gap = history.get("seconds_since_block_last_get")
    block_gap = None if block_gap is None else float(block_gap)
    if group["block_get_count"] is None:
        group["block_get_count"] = block_count
        group["block_get_gap_s"] = block_gap
    elif group["block_get_count"] != block_count or group["block_get_gap_s"] != block_gap:
        raise AssertionError(("block history changed inside request", row["actual_decision_request_sequence"]))

    group["chunk_get_count_sum"] += int(history["prior_chunk_get_count"])
    chunk_gap = history.get("seconds_since_chunk_last_get")
    if chunk_gap is None:
        group["chunk_get_gap_missing"] += 1
    else:
        group["chunk_get_gap_sum"] += float(chunk_gap)
        group["chunk_get_gap_known"] += 1

    group["reject_count_sum"] += int(history["prior_actual_rejection_count"])
    reject_gap = history.get("seconds_since_latest_actual_rejection")
    if reject_gap is None:
        group["reject_gap_missing"] += 1
    else:
        group["reject_gap_sum"] += float(reject_gap)
        group["reject_gap_known"] += 1


def finish_group(group, label_hits):
    count = group["candidate_count"]
    if count <= 0 or len(group["candidate_chunk_ids_for_join_only"]) != count:
        raise AssertionError("empty or inconsistent request candidate group")
    seq = group["request_sequence"]
    extra = {
        "prior_block_get_count": float(group["block_get_count"]),
        "seconds_since_block_last_get_s": as_number(group["block_get_gap_s"]),
        "prior_chunk_get_count_mean": group["chunk_get_count_sum"] / count,
        "seconds_since_chunk_last_get_mean_known_s": (
            group["chunk_get_gap_sum"] / group["chunk_get_gap_known"]
            if group["chunk_get_gap_known"] else math.nan),
        "chunk_get_gap_missing_fraction": group["chunk_get_gap_missing"] / count,
        "prior_actual_rejection_count_mean": group["reject_count_sum"] / count,
        "seconds_since_latest_actual_rejection_mean_known_s": (
            group["reject_gap_sum"] / group["reject_gap_known"]
            if group["reject_gap_known"] else math.nan),
        "reject_gap_missing_fraction": group["reject_gap_missing"] / count,
        **group["ctx_features"],
    }
    vec = group["author_vector_sum"] / count
    return {
        "request_sequence_audit_only": seq,
        "elapsed_s": group["elapsed_s"],
        "block_key_for_join_only": group["block_key_for_join_only"],
        "chunk_ids_for_join_only": group["candidate_chunk_ids_for_join_only"],
        "candidate_chunk_count": count,
        "author_vector": vec,
        "author_score_mean": group["author_score_sum"] / count,
        "block_recency_gap_s": extra["seconds_since_block_last_get_s"],
        "extra_features": extra,
        "future_repeat_chunk_count": int(label_hits),
        "label_fraction": float(label_hits / count),
    }


def train_regressor(train_rows, feature_names, get_matrix):
    x = get_matrix(train_rows, feature_names)
    y = np.asarray([r["label_fraction"] for r in train_rows], dtype=np.float64)
    if not len(y) or np.any(~np.isfinite(y)) or np.any(y < 0) or np.any(y > 1):
        raise AssertionError("invalid training targets")
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "num_threads": 1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    dataset = lgb.Dataset(x, label=y, feature_name=feature_names, free_raw_data=False)
    started = time.perf_counter()
    booster = lgb.train(params, dataset, num_boost_round=100)
    elapsed = time.perf_counter() - started
    return booster, elapsed, params


def request_matrix(rows, feature_names, author_names, new_names):
    matrices = []
    for row in rows:
        values = []
        for name in feature_names:
            if name in author_names:
                values.append(float(row["author_vector"][author_names.index(name)]))
            elif name == "author_score_mean":
                values.append(float(row["author_score_mean"]))
            else:
                values.append(float(row["extra_features"].get(name, math.nan)))
        matrices.append(values)
    return np.asarray(matrices, dtype=np.float64)


def ranking_value(row, method):
    if method == "AUTHOR-SCORE":
        return row["author_score_mean"]
    if method == "RECENCY":
        return row["block_recency_gap_s"]
    return row[f"prediction_{method.lower()}"]


def select_complete_requests(rows, method):
    total_chunks = sum(r["candidate_chunk_count"] for r in rows)
    target = int(math.ceil(TOP_FRACTION * total_chunks)) if total_chunks else 0
    tie = {id(r): request_tiebreak(r["request_sequence_audit_only"]) for r in rows}
    if method == "RECENCY":
        def recency_key(row):
            gap = row["block_recency_gap_s"]
            finite = gap is not None and math.isfinite(float(gap))
            return (not finite, float(gap) if finite else math.inf, tie[id(row)])
        ordered = sorted(rows, key=recency_key)
    else:
        ordered = sorted(rows, key=lambda r: (-float(ranking_value(r, method)), tie[id(r)]))
    chosen = []
    selected_chunks = 0
    for row in ordered:
        before = abs(target - selected_chunks)
        after = abs(target - (selected_chunks + row["candidate_chunk_count"]))
        if after < before or (after == before and selected_chunks < target):
            chosen.append(row)
            selected_chunks += row["candidate_chunk_count"]
        else:
            break
        if selected_chunks >= target:
            break
    selected_positive = sum(r["future_repeat_chunk_count"] for r in chosen)
    pool_positive = sum(r["future_repeat_chunk_count"] for r in rows)
    capture_yield = selected_positive / selected_chunks if selected_chunks else 0.0
    scaled_capture = capture_yield * target
    return {
        "method": method,
        "candidate_pool_requests": len(rows),
        "candidate_pool_chunks": total_chunks,
        "candidate_pool_future_repeat_chunks": pool_positive,
        "target_selected_chunks": target,
        "selected_requests": len(chosen),
        "selected_chunks": selected_chunks,
        "selected_fraction_pct": 100.0 * selected_chunks / total_chunks if total_chunks else 0.0,
        "selection_boundary_delta_chunks": selected_chunks - target,
        "selected_future_repeat_chunks": selected_positive,
        "selected_repeat_yield_pct": 100.0 * capture_yield,
        "scaled_repeat_capture_at_exact_target_chunks": scaled_capture,
        "fraction_of_pool_future_repeats_captured_pct": (
            100.0 * selected_positive / pool_positive if pool_positive else math.nan),
        "selected_request_sequences": [r["request_sequence_audit_only"] for r in chosen],
    }


def main():
    started_all = time.perf_counter()
    manifest = validate_manifest()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    author_names = schema["model_feature_names_in_exact_order"]
    if len(author_names) != 18:
        raise AssertionError("expected frozen 18-dimensional author vector")
    trace = trace_bounds(TRACE)
    trace_end_elapsed = trace["duration_s"]
    eval_mature_end = trace_end_elapsed - HORIZON_S
    if eval_mature_end <= EVAL_START_S:
        raise AssertionError("no mature evaluation horizon")

    new_names = [
        "prior_block_get_count", "seconds_since_block_last_get_s",
        "prior_chunk_get_count_mean", "seconds_since_chunk_last_get_mean_known_s",
        "chunk_get_gap_missing_fraction", "prior_actual_rejection_count_mean",
        "seconds_since_latest_actual_rejection_mean_known_s", "reject_gap_missing_fraction",
        "current_request_chunk_count", "cache_occupancy_fraction_at_request_start",
        "last_total_get_dt_pct", "last_demand_read_dt_pct", "last_extra_prefetch_dt_pct",
        "last_put_dt_pct", "last_writes", "last_evictions",
    ]
    for source, _ in LOAD_METRICS:
        new_names += [f"last6_mean_{source}", f"last6_max_{source}"]
    old_names = author_names + ["author_score_mean"]
    new_model_names = old_names + new_names

    protocol = {
        "protocol": "BC08 frozen request-level next-GET reuse ranking",
        "created_before_label_generation": True,
        "horizon_seconds": HORIZON_S,
        "evaluation_start_trace_elapsed_seconds": EVAL_START_S,
        "training_rule": "warmup demand candidates with saved author score <= frozen HIGH threshold; request mature only when request time + H <= evaluation start",
        "evaluation_rule": "evaluation demand candidates actually rejected under frozen HIGH; request mature only when request time + H <= raw trace end",
        "high_threshold": HIGH_THRESHOLD,
        "label": "per request, fraction of HIGH-rejected demand candidate chunks with any later GET of the same full block/chunk key within H seconds; candidate chunk counted at most once",
        "model_unit": "one row per request; candidate features averaged over R_q; label is the fraction among R_q",
        "top_selection_fraction": TOP_FRACTION,
        "complete_request_selection": "rank requests; accumulate a prefix of complete requests and choose the closer of the prefix immediately before/after crossing the ceil(1% of candidate chunks) quota; shared SHA-256 request tie break",
        "half_split": "mature evaluation interval midpoint: (evaluation start + raw trace end - H)/2; the ranking is independently evaluated at approximately 1% within each half",
        "tree": {"library": "LightGBM", "version": lgb.__version__, "objective": "regression_l2", "num_boost_round": 100, "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 50, "num_threads": 1, "seed": SEED},
        "old_features": old_names,
        "new_additional_features": new_names,
        "new_features_excluded": {
            "evaluation_write_budget_progress": "not defined during the warmup training interval, so it cannot provide a matched historical training signal",
            "request_sequence_block_key_chunk_id_bc05_membership": "audit/join fields only; never model inputs",
        },
        "recency_rule": "ascending seconds_since_block_last_get; missing history sorts after finite gaps and is never encoded as zero",
        "author_score_rule": "descending mean saved author model score over the request's HIGH-rejected demand candidates",
        "fit_metric": "request-level RMSE is descriptive only; primary metric is future-repeat capture in approximately top 1% of candidate chunks",
        "tie_break": {"version": TIE_VERSION, "key": "SHA-256(version:request_sequence) used only for deterministic ties, not a feature"},
        "input_hashes": {
            "candidate_snapshot_sha256": sha(CANDIDATES),
            "get_context_snapshot_sha256": sha(CONTEXTS),
            "feature_schema_sha256": sha(SCHEMA),
            "snapshot_manifest_sha256": sha(MANIFEST),
            "region1_trace_sha256": sha(TRACE),
            "author_model_sha256": sha(AUTHOR_MODEL),
        },
        "snapshot_manifest_declares_frozen_before_BC05_link": bool(manifest.get("snapshot_frozen_before_BC05_target_link")),
        "bc05_sidecar_read": False,
    }
    protocol_path = OUT / "BC08_protocol.json"
    json_dump(protocol_path, protocol)

    # Compact request contexts use author-parsed complete block keys and chunk lists.
    contexts = []
    context_by_seq = {}
    previous_seq = 0
    previous_time = -math.inf
    context_chunk_events = 0
    with gzip.open(CONTEXTS, "rt", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            seq = int(raw["request_sequence"])
            elapsed = float(raw["trace_elapsed_s"])
            block = str(raw["full_block_key"])
            chunks = tuple(int(c) for c in raw["requested_chunks"])
            if raw.get("operation") != "GET":
                raise AssertionError(("non-GET in GET context table", seq, raw.get("operation")))
            if seq <= previous_seq or elapsed + 1e-7 < previous_time:
                raise AssertionError(("request contexts not monotonic", seq, elapsed, previous_seq, previous_time))
            if len(chunks) != len(set(chunks)):
                raise AssertionError(("duplicate requested chunk in one GET", seq))
            if int(raw["current_request_chunk_count"]) != len(chunks):
                raise AssertionError(("request chunk count mismatch", seq))
            ctx = {"seq": seq, "elapsed": elapsed, "block": block,
                   "chunks": chunks, "features": context_features(raw)}
            contexts.append(ctx)
            context_by_seq[seq] = ctx
            previous_seq, previous_time = seq, elapsed
            context_chunk_events += len(chunks)
    if not contexts:
        raise AssertionError("empty GET request context table")
    if contexts[-1]["elapsed"] > trace_end_elapsed + 1e-3:
        raise AssertionError("GET context extends past the original trace endpoint")

    stats = Counter()
    groups = {}
    candidate_rows = 0
    with gzip.open(CANDIDATES, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            candidate_rows += 1
            stats["all_candidate_rows"] += 1
            if row.get("candidate_source") != "demand":
                stats["excluded_non_demand_candidate_rows"] += 1
                continue
            stats["demand_candidate_rows"] += 1
            created_seq = int(row["candidate_created_request_sequence"])
            decision_seq = int(row["actual_decision_request_sequence"])
            if created_seq != decision_seq:
                raise AssertionError(("candidate delayed beyond current request", row.get("candidate_id")))
            seq = decision_seq
            elapsed = float(row["actual_decision_trace_elapsed_s"])
            if abs(elapsed - float(row["candidate_created_trace_elapsed_s"])) > 1e-6:
                raise AssertionError(("candidate time differs from decision request time", row.get("candidate_id")))
            ctx = context_by_seq.get(seq)
            if ctx is None:
                raise AssertionError(("demand candidate lacks GET request context", seq))
            block = str(row["full_block_key"])
            chunk = int(row["chunk_id"])
            if ctx["block"] != block or chunk not in ctx["chunks"]:
                raise AssertionError(("candidate does not match parsed current GET", seq, block, chunk))
            score = float(row["author_model_score"])
            if elapsed < EVAL_START_S:
                if score > HIGH_THRESHOLD:
                    stats["warmup_not_high_rejected_rows"] += 1
                    continue
                stats["warmup_counterfactual_high_rejected_rows"] += 1
                if elapsed + HORIZON_S > EVAL_START_S:
                    stats["warmup_unmatured_rows"] += 1
                    continue
                split = "train"
            else:
                threshold = float(row["threshold"])
                if abs(threshold - HIGH_THRESHOLD) > 1e-12:
                    stats["evaluation_rows_not_at_frozen_high_threshold"] += 1
                    continue
                if bool(row["author_decision_accept"]):
                    stats["evaluation_high_accepted_rows"] += 1
                    continue
                stats["evaluation_high_rejected_rows"] += 1
                if elapsed + HORIZON_S > trace_end_elapsed:
                    stats["evaluation_unmatured_rows"] += 1
                    continue
                split = "evaluation"
            if seq not in groups:
                groups[seq] = empty_group(seq, elapsed, block, ctx["features"], row["author_model_input_vector"])
                groups[seq]["split"] = split
                groups[seq]["block_recency_gap_s"] = row["legal_key_history_at_decision_request_start"].get("seconds_since_block_last_get")
                if groups[seq]["block_recency_gap_s"] is not None:
                    groups[seq]["block_recency_gap_s"] = float(groups[seq]["block_recency_gap_s"])
                if int(ctx["features"]["current_request_chunk_count"]) != int(row["decision_current_request_chunk_count"]):
                    raise AssertionError(("request width mismatch between candidate and context", seq))
            elif groups[seq]["split"] != split:
                raise AssertionError(("request crossed temporal split", seq))
            append_candidate(groups[seq], row)

    if candidate_rows != int(manifest["counts"]["actual_candidate_decisions"]):
        raise AssertionError(("candidate row count differs from frozen manifest", candidate_rows))
    if stats["evaluation_rows_not_at_frozen_high_threshold"]:
        raise AssertionError("evaluation includes rows outside frozen HIGH threshold")
    request_samples = {}
    for seq, group in groups.items():
        request_samples[seq] = finish_group(group, label_hits=0)
        request_samples[seq]["split"] = group["split"]

    # Reverse the complete GET stream. Before inserting the current request,
    # next_get contains only later requests, so it labels true future GETs.
    next_get_time = {}
    labeled = set()
    repeat_chunk_events_seen = 0
    for ctx in reversed(contexts):
        sample = request_samples.get(ctx["seq"])
        if sample is not None:
            hits = 0
            current = ctx["elapsed"]
            if sample["block_key_for_join_only"] != ctx["block"]:
                raise AssertionError(("sample/current GET block mismatch", ctx["seq"]))
            for chunk in sample["chunk_ids_for_join_only"]:
                later = next_get_time.get((ctx["block"], chunk))
                if later is not None:
                    if later + 1e-7 < current:
                        raise AssertionError(("future GET time precedes current request", ctx["seq"], later, current))
                    if later <= current + HORIZON_S + 1e-7:
                        hits += 1
            if hits > sample["candidate_chunk_count"]:
                raise AssertionError(("request repeat label exceeds Rq", ctx["seq"]))
            sample["future_repeat_chunk_count"] = hits
            sample["label_fraction"] = hits / sample["candidate_chunk_count"]
            labeled.add(ctx["seq"])
            repeat_chunk_events_seen += hits
        for chunk in ctx["chunks"]:
            next_get_time[(ctx["block"], chunk)] = ctx["elapsed"]

    if len(labeled) != len(request_samples):
        raise AssertionError(("some request labels were not assigned", len(labeled), len(request_samples)))

    train_rows = sorted((r for r in request_samples.values() if r["split"] == "train"), key=lambda r: r["elapsed_s"])
    eval_rows = sorted((r for r in request_samples.values() if r["split"] == "evaluation"), key=lambda r: r["elapsed_s"])
    if not train_rows or not eval_rows:
        raise AssertionError(("empty train/evaluation cohort", len(train_rows), len(eval_rows)))
    half_midpoint = (EVAL_START_S + eval_mature_end) / 2.0
    eval_half1 = [r for r in eval_rows if r["elapsed_s"] < half_midpoint]
    eval_half2 = [r for r in eval_rows if r["elapsed_s"] >= half_midpoint]
    if not eval_half1 or not eval_half2:
        raise AssertionError(("empty evaluation half", len(eval_half1), len(eval_half2)))

    get_matrix = lambda rows, names: request_matrix(rows, names, author_names, new_names)
    old_model, old_fit_s, model_params = train_regressor(train_rows, old_names, get_matrix)
    new_model, new_fit_s, _ = train_regressor(train_rows, new_model_names, get_matrix)
    eval_x_old = get_matrix(eval_rows, old_names)
    eval_x_new = get_matrix(eval_rows, new_model_names)
    eval_y = np.asarray([r["label_fraction"] for r in eval_rows], dtype=np.float64)
    old_pred = old_model.predict(eval_x_old, num_threads=1)
    new_pred = new_model.predict(eval_x_new, num_threads=1)
    for row, po, pn in zip(eval_rows, old_pred, new_pred):
        row["prediction_old"] = float(po)
        row["prediction_new"] = float(pn)
    rmse_old = float(np.sqrt(np.mean((old_pred - eval_y) ** 2)))
    rmse_new = float(np.sqrt(np.mean((new_pred - eval_y) ** 2)))

    model_dir = HERE / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    old_model_path = model_dir / "BC08_OLD_lightgbm.txt"
    new_model_path = model_dir / "BC08_NEW_lightgbm.txt"
    old_model.save_model(str(old_model_path))
    new_model.save_model(str(new_model_path))

    methods = ["AUTHOR-SCORE", "RECENCY", "OLD", "NEW"]
    slices = [("EVAL-ALL", eval_rows), ("EVAL-FIRST-HALF", eval_half1), ("EVAL-SECOND-HALF", eval_half2)]
    selection_rows = []
    selected_by_slice_method = {}
    raw_capture_order_reversal = {}
    for slice_name, rows in slices:
        slice_methods = {}
        for method in methods:
            result = select_complete_requests(rows, method)
            selected_by_slice_method[(slice_name, method)] = result
            slice_methods[method] = result
            selection_rows.append({k: v for k, v in result.items() if k != "selected_request_sequences"} | {"slice": slice_name})
        others = ["AUTHOR-SCORE", "RECENCY", "OLD"]
        best_other = max(others, key=lambda m: slice_methods[m]["scaled_repeat_capture_at_exact_target_chunks"])
        best_value = slice_methods[best_other]["scaled_repeat_capture_at_exact_target_chunks"]
        new_value = slice_methods["NEW"]["scaled_repeat_capture_at_exact_target_chunks"]
        normalized_gain = None if best_value <= 0 else (new_value / best_value - 1.0)
        raw_best = max(others, key=lambda m: slice_methods[m]["selected_future_repeat_chunks"])
        raw_order = (slice_methods["NEW"]["selected_future_repeat_chunks"] > slice_methods[raw_best]["selected_future_repeat_chunks"])
        norm_order = new_value > best_value
        raw_capture_order_reversal[slice_name] = raw_order != norm_order
        for item in selection_rows[-4:]:
            if item["slice"] == slice_name:
                item["strongest_other_by_normalized_capture"] = best_other
                item["new_gain_vs_strongest_other_pct"] = None if normalized_gain is None else 100.0 * normalized_gain
                item["strongest_other_by_raw_capture"] = raw_best
                item["new_beats_strongest_other_raw_capture"] = raw_order
        slice_methods["_comparison"] = {
            "strongest_other": best_other,
            "normalized_gain": normalized_gain,
            "raw_strongest_other": raw_best,
            "raw_order_reversal": raw_capture_order_reversal[slice_name],
        }

    eval_by_seq = {r["request_sequence_audit_only"]: r for r in eval_rows}
    chosen_seq = {}
    for slice_name, _ in slices:
        for method in methods:
            for seq in selected_by_slice_method[(slice_name, method)]["selected_request_sequences"]:
                chosen_seq[(slice_name, method, seq)] = True

    metrics_fields = [
        "slice", "method", "candidate_pool_requests", "candidate_pool_chunks",
        "candidate_pool_future_repeat_chunks", "target_selected_chunks", "selected_requests",
        "selected_chunks", "selected_fraction_pct", "selection_boundary_delta_chunks",
        "selected_future_repeat_chunks", "selected_repeat_yield_pct",
        "scaled_repeat_capture_at_exact_target_chunks", "fraction_of_pool_future_repeats_captured_pct",
        "strongest_other_by_normalized_capture", "new_gain_vs_strongest_other_pct",
        "strongest_other_by_raw_capture", "new_beats_strongest_other_raw_capture",
    ]
    write_csv(OUT / "BC08_top1pct_metrics.csv", selection_rows, metrics_fields)

    pred_rows = []
    for r in eval_rows:
        seq = r["request_sequence_audit_only"]
        out = {
            "request_sequence_audit_only": seq,
            "elapsed_s": r["elapsed_s"],
            "evaluation_half": "first" if r["elapsed_s"] < half_midpoint else "second",
            "candidate_chunk_count": r["candidate_chunk_count"],
            "future_repeat_chunk_count": r["future_repeat_chunk_count"],
            "label_fraction": r["label_fraction"],
            "author_score_mean": r["author_score_mean"],
            "seconds_since_block_last_get_s": r["block_recency_gap_s"],
            "prediction_old": r["prediction_old"],
            "prediction_new": r["prediction_new"],
        }
        for method in methods:
            out[f"selected_{method.lower().replace('-', '_')}_eval_all"] = bool(chosen_seq.get(("EVAL-ALL", method, seq), False))
        pred_rows.append(out)
    pred_fields = list(pred_rows[0])
    with gzip.open(OUT / "BC08_eval_request_predictions.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pred_rows)

    train_block_ids = {r["block_key_for_join_only"] for r in train_rows}
    eval_block_ids = {r["block_key_for_join_only"] for r in eval_rows}
    block_overlap = train_block_ids & eval_block_ids
    train_candidate_chunks = sum(r["candidate_chunk_count"] for r in train_rows)
    eval_candidate_chunks = sum(r["candidate_chunk_count"] for r in eval_rows)
    total_eval_repeats = sum(r["future_repeat_chunk_count"] for r in eval_rows)
    total_train_repeats = sum(r["future_repeat_chunk_count"] for r in train_rows)
    summary = {
        "protocol_passed": True,
        "status": "completed",
        "elapsed_wall_seconds": time.perf_counter() - started_all,
        "model_fit_wall_seconds": {"OLD": old_fit_s, "NEW": new_fit_s},
        "python_version": __import__("sys").version,
        "lightgbm_version": lgb.__version__,
        "input_hashes": protocol["input_hashes"],
        "trace_bounds": trace,
        "maturity": {
            "horizon_seconds": HORIZON_S,
            "evaluation_start_seconds": EVAL_START_S,
            "mature_evaluation_end_seconds": eval_mature_end,
            "evaluation_half_midpoint_seconds": half_midpoint,
        },
        "source_counts": dict(stats),
        "frozen_candidate_rows": candidate_rows,
        "GET_request_contexts": len(contexts),
        "GET_chunk_events_for_future_lookup": context_chunk_events,
        "eligible_request_counts": {"train": len(train_rows), "evaluation": len(eval_rows),
                                     "evaluation_first_half": len(eval_half1), "evaluation_second_half": len(eval_half2)},
        "eligible_candidate_chunk_counts": {"train": train_candidate_chunks, "evaluation": eval_candidate_chunks,
                                             "evaluation_first_half": sum(r["candidate_chunk_count"] for r in eval_half1),
                                             "evaluation_second_half": sum(r["candidate_chunk_count"] for r in eval_half2)},
        "future_repeat_chunk_counts": {"train": total_train_repeats, "evaluation": total_eval_repeats,
                                       "evaluation_repeat_fraction": total_eval_repeats / eval_candidate_chunks},
        "block_id_is_model_input": False,
        "train_eval_block_overlap_audit_only": {"train_unique_blocks": len(train_block_ids),
                                                  "evaluation_unique_blocks": len(eval_block_ids),
                                                  "overlap_unique_blocks": len(block_overlap)},
        "model": {
            "params": model_params,
            "num_boost_round": 100,
            "old_feature_count": len(old_names),
            "new_feature_count": len(new_model_names),
            "old_feature_names": old_names,
            "new_additional_feature_names": new_names,
            "evaluation_request_rmse": {"OLD": rmse_old, "NEW": rmse_new, "new_minus_old": rmse_new-rmse_old},
            "old_top_feature_importance_gain": sorted(
                [{"feature": n, "gain": float(v)} for n, v in zip(old_names, old_model.feature_importance(importance_type="gain"))],
                key=lambda x: x["gain"], reverse=True)[:10],
            "new_top_feature_importance_gain": sorted(
                [{"feature": n, "gain": float(v)} for n, v in zip(new_model_names, new_model.feature_importance(importance_type="gain"))],
                key=lambda x: x["gain"], reverse=True)[:15],
            "saved_models": {"OLD": {"path": str(old_model_path), "sha256": sha(old_model_path)},
                             "NEW": {"path": str(new_model_path), "sha256": sha(new_model_path)}},
        },
        "selection": {
            "fraction": TOP_FRACTION,
            "methods": methods,
            "all_and_half_results": selection_rows,
            "raw_vs_normalized_order_reversal_by_slice": raw_capture_order_reversal,
        },
        "is_independent_confirmation": False,
        "bc05_sidecar_read": False,
        "bc05_objects_or_peak_position_used_as_input_or_label": False,
    }
    json_dump(OUT / "BC08_audit.json", summary)

    feature_rows = []
    for i, name in enumerate(author_names):
        feature_rows.append({"model": "OLD and NEW", "feature": name, "source": "frozen BC06 author_model_input_vector", "request_aggregation": "mean across R_q candidate chunks", "author_model_input": True, "notes": "Exact author feature order; identifiers excluded."})
    feature_rows.append({"model": "OLD and NEW", "feature": "author_score_mean", "source": "frozen BC06 author_model_score", "request_aggregation": "mean across R_q candidate chunks", "author_model_input": "derived from author model", "notes": "Ranking baseline and OLD/NEW input."})
    new_sources = {
        "prior_block_get_count": ("legal_key_history_at_decision_request_start.prior_block_get_count", "same for request"),
        "seconds_since_block_last_get_s": ("legal_key_history_at_decision_request_start.seconds_since_block_last_get", "same for request; missing remains NaN"),
        "prior_chunk_get_count_mean": ("legal_key_history_at_decision_request_start.prior_chunk_get_count", "mean across R_q"),
        "seconds_since_chunk_last_get_mean_known_s": ("legal_key_history_at_decision_request_start.seconds_since_chunk_last_get", "mean over known chunks; missing fraction is separate"),
        "chunk_get_gap_missing_fraction": ("legal_key_history_at_decision_request_start.seconds_since_chunk_last_get", "missing R_q chunks divided by |R_q|"),
        "prior_actual_rejection_count_mean": ("legal_key_history_at_decision_request_start.prior_actual_rejection_count", "mean across R_q"),
        "seconds_since_latest_actual_rejection_mean_known_s": ("legal_key_history_at_decision_request_start.seconds_since_latest_actual_rejection", "mean over known chunks; missing fraction is separate"),
        "reject_gap_missing_fraction": ("legal_key_history_at_decision_request_start.seconds_since_latest_actual_rejection", "missing R_q chunks divided by |R_q|"),
        "current_request_chunk_count": ("BC06 GET request context", "one current-request scalar"),
        "cache_occupancy_fraction_at_request_start": ("BC06 GET request context", "request start state"),
        "last_total_get_dt_pct": ("BC06 last_completed_window", "last ended window"),
        "last_demand_read_dt_pct": ("BC06 last_completed_window", "last ended window"),
        "last_extra_prefetch_dt_pct": ("BC06 last_completed_window", "last ended window"),
        "last_put_dt_pct": ("BC06 last_completed_window", "last ended window"),
        "last_writes": ("BC06 last_completed_window", "last ended window"),
        "last_evictions": ("BC06 last_completed_window", "last ended window"),
    }
    for source, _ in LOAD_METRICS:
        new_sources[f"last6_mean_{source}"] = ("BC06 last_six_completed_windows", "mean across available ended windows; active window excluded")
        new_sources[f"last6_max_{source}"] = ("BC06 last_six_completed_windows", "maximum across available ended windows; active window excluded")
    for name in new_names:
        source, aggregation = new_sources[name]
        if name in {"prior_block_get_count", "prior_chunk_get_count_mean"}:
            overlap = "partial overlap with author rolling-hour inputs; count can add resolution/context"
        elif name == "current_request_chunk_count":
            overlap = "related to, but not identical with, author byte-size features"
        else:
            overlap = "not directly in the author 18-vector"
        feature_rows.append({"model": "NEW only", "feature": name, "source": source,
                             "request_aggregation": aggregation, "author_model_input": False, "notes": overlap})
    write_csv(OUT / "BC08_feature_dictionary.csv", feature_rows,
              ["model", "feature", "source", "request_aggregation", "author_model_input", "notes"])

    all_result = next(r for r in selection_rows if r["slice"] == "EVAL-ALL" and r["method"] == "NEW")
    half1_result = next(r for r in selection_rows if r["slice"] == "EVAL-FIRST-HALF" and r["method"] == "NEW")
    half2_result = next(r for r in selection_rows if r["slice"] == "EVAL-SECOND-HALF" and r["method"] == "NEW")
    g_all = all_result["new_gain_vs_strongest_other_pct"]
    g_half1 = half1_result["new_gain_vs_strongest_other_pct"]
    g_half2 = half2_result["new_gain_vs_strongest_other_pct"]
    order_flip = any(raw_capture_order_reversal.values())
    if g_all is None or g_half1 is None or g_half2 is None:
        verdict = "有效正例不足以形成稳定排名判读；记为不确定。"
    elif order_flip:
        verdict = "NEW 与最强对照的原始捕捉数和按选择量归一化排序不一致；选择边界可能影响结论，记为不确定。"
    elif g_all >= 10.0 and g_half1 > 0 and g_half2 > 0:
        verdict = "NEW 相对最强对照的归一化重用捕捉量提升至少 10%，且两个时间半段均为正；支持新增合法历史含有可捕捉信息，值得进入在线回放检验。"
    else:
        verdict = "NEW 未达到预设的稳定优势筛查条件；不扩展本特征包、模型或重用目标。"

    method_lookup = {(r["slice"], r["method"]): r for r in selection_rows}
    md = [
        "# BC08：新增合法历史能否预测近未来重用",
        "",
        verdict,
        "",
        "## 固定目标与时间切分",
        "",
        f"- 标签时域 H={HORIZON_S:.3f} 秒。对每个请求 q，先取其 HIGH 拒绝需求候选集合 R_q；标签为 R_q 中在后续 H 秒内再次出现同一完整块键和 chunk 的比例。每个 chunk 只计一次，未来请求不看缓存命中状态。该标签只代表近未来再次 GET，不代表净收益、驻留概率或峰值贡献。",
        f"- 训练只含预热请求且 t+H≤{EVAL_START_S:.6f} 秒；预热真实按 BASE 阈值运行，训练池按保存原分数和固定 HIGH 阈值 {HIGH_THRESHOLD:.15f} 重建。评价只含评价段真实 HIGH 拒绝候选，并排除 t+H 超过原始轨迹终点的请求。",
        f"- 评价成熟区间为 [{EVAL_START_S:.3f}, {eval_mature_end:.3f}] 秒，第一／第二半以 {half_midpoint:.3f} 秒为界。相同请求的 chunk 汇成一行；请求序号和完整块键只用于关联标签，不进入模型。训练／评价有 {len(block_overlap):,} 个重复块键，但模型没有块 ID 输入。",
        "- 候选池只含需求候选，不含预取候选；未读取 BC05 稀疏 sidecar、对象名单、峰值位置或命中／淘汰结果。",
        "",
        "## 模型与选择规则",
        "",
        f"- LightGBM {lgb.__version__} 回归器固定为 100 轮、15 叶、学习率 0.05、最小叶 50、单线程；OLD 输入为作者 18 维输入均值和原模型分数均值，NEW 再加 {len(new_names)} 个合法历史／请求上下文特征。没有调参。",
        "- 所有输入按请求内 R_q 候选均值聚合；chunk 访问间隔和拒绝间隔对已知值取均值，并另存缺失比例。evaluation write budget progress 未在预热训练段定义，因此排除。",
        "- 对照包括作者原分数均值降序、最近块 GET 间隔升序、OLD、NEW。无历史间隔排在有历史间隔后。各方法按请求排序，采用同一 SHA-256 请求哈希打破并列；依次纳入完整请求，并在跨过 1% chunk 目标时选择更接近目标的边界前缀。每个时间半段独立使用同一约 1% 选择规则。",
        "- 主量是选择集未来重复 GET chunk 数，并同时报告选择数量、重用率和按实际选择率归一化到精确 1% 目标的捕捉量。归一化只用于边界请求数有小差异时的公平比较；原始选择量也保留。",
        "",
        "## 时间外样本与标签统计",
        "",
        f"- 训练：{len(train_rows):,} 个请求、{train_candidate_chunks:,} 个候选 chunk，其中 {total_train_repeats:,} 个在 H 内再次 GET；请求标签均值 {np.mean([r['label_fraction'] for r in train_rows]):.4%}。",
        f"- 评价：{len(eval_rows):,} 个请求、{eval_candidate_chunks:,} 个候选 chunk，其中 {total_eval_repeats:,} 个在 H 内再次 GET；chunk 重用率 {total_eval_repeats/eval_candidate_chunks:.4%}。两半分别有 {len(eval_half1):,} / {len(eval_half2):,} 个请求。",
        f"- 请求级 RMSE（描述指标）：OLD {rmse_old:.6f}，NEW {rmse_new:.6f}。是否继续只按 1% 排名捕捉结果判读。",
        "",
        "## Top 1% 排名结果",
        "",
        "| 时段 | 方法 | 候选请求 | 候选 chunk | 选择请求 | 实际选择 chunk | 选择比例 | 后续重用 chunk | 选择集重用率 | 精确 1% 归一化捕捉量 | 相对最强其他对照 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slice_name, _ in slices:
        for method in methods:
            r = method_lookup[(slice_name, method)]
            gain = r.get("new_gain_vs_strongest_other_pct")
            gain_text = "—" if method != "NEW" or gain is None else f"{gain:+.2f}%"
            md.append(f"| {slice_name} | {method} | {r['candidate_pool_requests']:,} | {r['candidate_pool_chunks']:,} | {r['selected_requests']:,} | {r['selected_chunks']:,} | {r['selected_fraction_pct']:.4f}% | {r['selected_future_repeat_chunks']:,} | {r['selected_repeat_yield_pct']:.4f}% | {r['scaled_repeat_capture_at_exact_target_chunks']:.2f} | {gain_text} |")
    md += [
        "",
        "对照的最强者按精确 1% 归一化捕捉量选出，见 `BC08_top1pct_metrics.csv`。若原始捕捉数与归一化排序相反，本轮按预设记为边界不确定。",
        "",
        "## 审计与限制",
        "",
        f"- BC06 冻结候选 {candidate_rows:,} 行；GET 请求上下文 {len(contexts):,} 条、未来标签展开 {context_chunk_events:,} 个 chunk 请求事件。回放使用作者解析器已产出的完整块键与 chunk；请求序列严格单调，chunk 级未来查询从当前 GET 之后的请求开始。原始 trace 时长 {trace_end_elapsed:.3f} 秒，SHA-256 `{protocol['input_hashes']['region1_trace_sha256']}`。",
        "- 作者输入向量与打分输入逐候选一致；R_q 内特征均值聚合；训练和评价样本按请求分组，没有拆分同一请求的 chunk。LightGBM 固定种子和单线程，OLD/NEW 都仅拟合一次。",
        "- 这是单一 Region1 轨迹上的离线开发筛查。两时间半段来自同一轨迹且可能共享块，不能视为独立确认；不能据此推断缓存策略净收益或系统峰值。",
        "",
        "## 交付",
        "",
        "- `BC08_protocol.json`：冻结时域、切分、输入与哈希规则。",
        "- `BC08_top1pct_metrics.csv`：整体与两半排名选择量及重用捕捉量。",
        "- `BC08_eval_request_predictions.csv.gz`：评价请求标签、冻结预测、作者分数、间隔及整体选择标记；请求序号仅为审计键。",
        "- `BC08_feature_dictionary.csv`：OLD／NEW 字段来源和请求聚合方式。",
        "- `BC08_audit.json`：输入哈希、样本成熟度、标签量、RMSE、特征重要性和执行状态。",
        "- LightGBM 文本模型保存在 `work/bc08/models/`；未训练 GRL 或其他模型。",
        "",
    ]
    (OUT / "BC08_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"complete": True, "verdict": verdict,
                      "train_requests": len(train_rows), "evaluation_requests": len(eval_rows),
                      "rmse_old": rmse_old, "rmse_new": rmse_new,
                      "selection": selection_rows,
                      "wall_seconds": time.perf_counter() - started_all}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
