"""BC02: one offline ordered-request scan using the pinned author parser.

No admission policy, prefetch policy or trained model is executed here.
The inherited _stats method determines pre-request checkpoint events.  The
checkpoint sink retains only timing/accounting, using the author's Stats API.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
REPO = BASE / "work" / "bc01" / "Baleen-FAST24"
sys.path.insert(0, str(REPO))

import compress_json
import numpy as np

from BCacheSim.cachesim.legacy_utils import read_processed_file_list_accesses
from BCacheSim.cachesim.ep_helpers import AccessPlus, record_service_time_get, record_service_time_put
from BCacheSim.cachesim.sim_cache import CacheSimulator
from BCacheSim.cachesim.utils import ods
from BCacheSim.episodic_analysis.episodes import service_time, st_to_util

WORK = BASE / "work" / "bc02"
OUT = BASE / "outputs"
ST_ATOL = 1e-9  # absolute seconds of sampled modeled service time
DT_ATOL = 1e-8  # absolute percentage points of DT; rounding only


def require_series(batches, key, n=None):
    if key not in batches:
        raise KeyError(f"missing mandatory raw-stat field: {key}")
    a = np.asarray(batches[key], dtype=np.float64)
    if a.ndim != 1 or len(a) == 0 or (n is not None and len(a) != n):
        raise ValueError(f"invalid length or shape: {key}")
    if not np.isfinite(a).all():
        raise ValueError(f"nonfinite mandatory field: {key}")
    return a


def delta(a):
    return np.diff(a, prepend=0.0)


def strict_parse(path, **kwargs):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = read_processed_file_list_accesses(str(path), get_features=True, only_gets=False, **kwargs)
    text = capture.getvalue()
    # Upstream's parser catches some malformed lines. Never silently skip them.
    if "Error in parsing line" in text or "ERROR!" in text:
        raise ValueError(f"upstream parser reported malformed records:\n{text}")
    if not result[0]:
        raise ValueError("empty request sequence")
    return result


class AccountingScan(CacheSimulator):
    """Use upstream _stats for boundaries; capture cost instead of cache actions."""
    KEYS = (
        "service_time_used", "service_time_writes", "service_time_nocache",
        "puts_ios", "puts_chunks", "iops_requests", "chunk_queries",
        "fetches_ios", "fetches_chunks_demandmiss", "seen_get_blocks",
    )

    def __init__(self, log_interval):
        ods.counters.clear()
        ods.freq.clear()
        ods.batches.clear()
        ods.idx = 0
        for key in self.KEYS:
            ods.counters[key] = 0.0 if key.startswith("service_time") else 0
        self.config = {"log_interval": log_interval}
        self.start_ts = None
        self.last_log_tracetime = None
        self.last_syscheck = time.time()
        self.last_print = {"i": 0, "time": time.time()}
        self.print_every_n_mins = 10
        self.seen_get_blocks = set()  # retained over the entire request sequence
        self.request_count = 0
        self.checkpoint_calls = 0

    def _syscheck(self):
        self.last_syscheck = time.time()

    def _checkpoint(self, acc_ts, print_log=True, save=True):
        # Same append/checkpoint ordering and final idx update as upstream.
        # Only the cache/ML/logging parts of its checkpoint sink are omitted.
        ods.append("time_phy", acc_ts.physical)
        ods.append("time_elapsed_phy", (acc_ts - self.start_ts).physical)
        ods.append("time_log", acc_ts.logical)
        ods.checkpoint_many(self.KEYS)
        self.last_log_tracetime = acc_ts
        ods.idx = int((acc_ts - self.start_ts).physical // self.config["log_interval"])
        self.checkpoint_calls += 1

    def scan(self, accesses):
        last = None
        for key, raw_access in accesses:
            acc = AccessPlus(key, raw_access)
            self._stats(acc.ts)  # author's boundary check BEFORE accounting
            if acc.is_put:
                record_service_time_put(acc)
            elif acc.is_get:
                ods.bump("iops_requests")
                ods.bump("chunk_queries", len(acc.chunks))
                ods.bump("service_time_nocache", service_time(1, len(acc.chunks)))
                if acc.block_id not in self.seen_get_blocks:
                    # Fetch only requested chunks at first GET. The whole block
                    # then becomes permanently/free available in the relaxation.
                    record_service_time_get(acc.chunks, [], acc)
                    self.seen_get_blocks.add(acc.block_id)
                    ods.counters["seen_get_blocks"] = len(self.seen_get_blocks)
            else:
                raise ValueError(f"unsupported operation: {acc.op}")
            self.request_count += 1
            last = acc
        if last is None:
            raise ValueError("empty scan")
        self._checkpoint(last.ts, print_log=True, save=False)  # upstream final settlement
        return {k: np.asarray(v, dtype=np.float64) for k, v in ods.batches.items()}


def manual_checks():
    WORK.mkdir(parents=True, exist_ok=True)
    one = float(service_time(1, 1))
    cases = [
        ("first_get_is_charged", ["A 0 131072 0 1 0 0 0"], one, 0.0, one, 1),
        # New requested chunks still free after block's first GET; crosses a
        # checkpoint to prove the seen set survives windows.
        ("later_get_same_block_is_free", [
            "A 0 131072 0 1 0 0 0",
            "A 393216 131072 601 2 0 0 0",
            "A 524288 131072 602 1 0 0 0",
        ], one, 0.0, 3 * one, 1),
        ("put_then_first_get_both_charged", [
            "A 0 131072 0 3 0 0 0", "A 0 131072 1 1 0 0 0",
        ], one, one, one, 1),
    ]
    passed = []
    for name, lines, first, put, nocache, blocks in cases:
        path = WORK / f"manual_{name}.trace"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        accesses, _, _ = strict_parse(path, with_pipeline=True)
        scanner = AccountingScan(600.0)
        result = scanner.scan(accesses)
        actual = [result["service_time_used_stats"][-1],
                  result["service_time_writes_stats"][-1],
                  result["service_time_nocache_stats"][-1]]
        np.testing.assert_allclose(actual, [first, put, nocache], rtol=0, atol=1e-14)
        if len(scanner.seen_get_blocks) != blocks:
            raise AssertionError(f"manual block identity failed: {name}")
        if name == "later_get_same_block_is_free":
            np.testing.assert_allclose(delta(result["service_time_used_stats"]), [one, 0], atol=1e-14, rtol=0)
        passed.append(name)

    path = WORK / "manual_hostname_repeat.trace"
    path.write_text("X 0 131072 0 1 0 0 11 3\nX 0 131072 2 1 0 0 12 1\n", encoding="utf-8")
    accesses, _, _ = strict_parse(path, with_pipeline=False)
    if [a.ts for _, a in accesses] != [0.0, 0.5, 1.0, 2.0]:
        raise AssertionError("repeat expansion/order mismatch")
    scanner = AccountingScan(600.0)
    result = scanner.scan(accesses)
    if len(accesses) != 4 or len(scanner.seen_get_blocks) != 2:
        raise AssertionError("hostname not preserved in block key")
    np.testing.assert_allclose(result["service_time_used_stats"][-1], 2 * one, atol=1e-14, rtol=0)
    passed.append("hostname_and_repeat_parser_semantics")

    path = WORK / "manual_boundary.trace"
    path.write_text("A 0 131072 0 1 0 0 0\nB 0 131072 600 1 0 0 0\nB 0 131072 601 1 0 0 0\n", encoding="utf-8")
    accesses, _, _ = strict_parse(path, with_pipeline=True)
    result = AccountingScan(600.0).scan(accesses)
    np.testing.assert_allclose(result["time_elapsed_phy"], [600, 601], atol=0, rtol=0)
    np.testing.assert_allclose(delta(result["service_time_used_stats"]), [one, one], atol=1e-14, rtol=0)
    np.testing.assert_allclose(delta(result["service_time_nocache_stats"]), [one, 2 * one], atol=1e-14, rtol=0)
    passed.append("checkpoint_before_current_request_and_final_settlement")

    for key in ("service_time_writes_stats", "service_time_nocache_stats", "time_elapsed_phy"):
        try:
            require_series({}, key)
        except KeyError:
            continue
        raise AssertionError(f"missing mandatory field incorrectly accepted: {key}")
    passed.append("mandatory_missing_fields_fail")
    return passed


def load_reference(name, pattern):
    paths = list(REPO.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"need exactly one {name} reference: {paths}")
    path = paths[0]
    raw = compress_json.load(str(path))
    main = compress_json.load(str(path).replace(".stats.lzma", ".lzma"))
    batches = raw["batches"]
    elapsed = require_series(batches, "time_elapsed_phy")
    required = ("time_phy", "time_log", "service_time_writes_stats", "service_time_nocache_stats",
                "service_time_used_stats", "puts_ios_stats", "puts_chunks_stats", "iops_requests_stats", "chunk_queries_stats")
    arrays = {k: require_series(batches, k, len(elapsed)) for k in required}
    arrays["time_elapsed_phy"] = elapsed
    return {"name": name, "path": path, "main": main, "arrays": arrays}


def main():
    started = time.perf_counter()
    tests = manual_checks()  # completed before reading the full Region1 trace
    print("Manual/protocol checks passed:", ", ".join(tests), flush=True)
    refs = [load_reference("Baleen", "runs/example/baleen/*/*/*.stats.lzma"),
            load_reference("RejectX", "runs/example/rejectx/*/*.stats.lzma")]
    baleen = refs[0]
    cfg = baleen["main"]["options"]
    interval = float(cfg["log_interval"])
    sample_ratio = float(baleen["main"]["sampleRatio"])
    stats_start = float(cfg["stats_start"])
    trace = REPO / cfg["trace"]
    if hashlib.sha1(trace.read_bytes()).hexdigest().upper() != "BC1BF6B5036FDB41D52F547C7820DC23C5D91B57":
        raise ValueError("Region1 trace checksum no longer matches BC01")
    for ref in refs:
        opt = ref["main"]["options"]
        if opt["trace"] != cfg["trace"] or opt["log_interval"] != interval or opt["stats_start"] != stats_start:
            raise ValueError("reference trace or window definition mismatch")
        if ref["main"]["sampleRatio"] != sample_ratio or ref["main"]["chunkSize"] != 131072:
            raise ValueError("reference metering scale mismatch")

    accesses, parsed_start, parsed_end = strict_parse(trace)
    scanner = AccountingScan(interval)
    scanned = scanner.scan(accesses)  # exactly one pass over the ordered real requests
    elapsed = require_series(scanned, "time_elapsed_phy")
    n = len(elapsed)
    duration = delta(elapsed)
    if np.any(duration <= 0):
        raise ValueError("nonpositive real-trace accounting interval duration")
    for ref in refs:
        for key in ("time_elapsed_phy", "time_phy", "time_log"):
            a = require_series(scanned, key, n)
            if not np.array_equal(a, ref["arrays"][key]):
                raise AssertionError(f"time/request boundary mismatch: {ref['name']} {key}")
        for key in ("puts_ios_stats", "puts_chunks_stats", "iops_requests_stats", "chunk_queries_stats"):
            a = require_series(scanned, key, n)
            if not np.array_equal(a, ref["arrays"][key]):
                raise AssertionError(f"request/chunk count mismatch: {ref['name']} {key}")

    scan_put = delta(require_series(scanned, "service_time_writes_stats", n))
    scan_nocache = delta(require_series(scanned, "service_time_nocache_stats", n))
    first_st = delta(require_series(scanned, "service_time_used_stats", n))
    first_ios = delta(require_series(scanned, "fetches_ios_stats", n))
    first_chunks = delta(require_series(scanned, "fetches_chunks_demandmiss_stats", n))
    seen_blocks = require_series(scanned, "seen_get_blocks_stats", n)
    if np.any(np.diff(seen_blocks) < 0) or seen_blocks[-1] != len(scanner.seen_get_blocks):
        raise AssertionError("seen-GET-block state was reset")

    # Use the same conversion and actual endpoint durations as BC01.
    def to_dt(st):
        return st_to_util(st, sample_ratio=sample_ratio, disks=36, duration_s=duration) * 100

    put_dt = to_dt(scan_put)
    first_dt = to_dt(first_st)
    lower_dt = put_dt + first_dt
    baleen_put = delta(baleen["arrays"]["service_time_writes_stats"])
    baleen_get = delta(baleen["arrays"]["service_time_used_stats"])
    baleen_dt = to_dt(baleen_get) + to_dt(baleen_put)
    errors = {}
    for ref in refs:
        ref_put = delta(ref["arrays"]["service_time_writes_stats"])
        ref_nocache = delta(ref["arrays"]["service_time_nocache_stats"])
        put_error = float(np.max(np.abs(scan_put - ref_put)))
        nocache_error = float(np.max(np.abs(scan_nocache - ref_nocache)))
        if put_error > ST_ATOL or nocache_error > ST_ATOL:
            raise AssertionError(f"metering failed for {ref['name']}: PUT={put_error}, nocache={nocache_error}")
        errors[ref["name"]] = {
            "put_seconds": put_error,
            "get_nocache_seconds": nocache_error,
            "put_dt_percentage_points": float(np.max(np.abs(to_dt(scan_put) - to_dt(ref_put)))),
            "get_nocache_dt_percentage_points": float(np.max(np.abs(to_dt(scan_nocache) - to_dt(ref_nocache)))),
        }
    violation = float(max(0, np.max(lower_dt - baleen_dt)))
    if violation > DT_ATOL:
        i = int(np.argmax(lower_dt - baleen_dt))
        raise AssertionError(f"lower bound exceeds Baleen at window {i}: {violation} percentage points")

    eval_index = int(stats_start // interval)
    if not 0 < eval_index < n:
        raise ValueError("invalid evaluation boundary")
    mask = np.arange(n) >= eval_index
    p_baleen = float(np.max(baleen_dt[mask]))
    p_lb = float(np.max(lower_dt[mask]))  # maximize across ALL evaluation windows first
    idx_baleen = int(np.flatnonzero(mask)[np.argmax(baleen_dt[mask])])
    idx_lb = int(np.flatnonzero(mask)[np.argmax(lower_dt[mask])])
    upper_gain = (p_baleen - p_lb) / p_baleen
    verdict = ("停止当前 Region1、当前动作范围内的峰值优化投入。" if upper_gain < 0.05 else
               "当前方向尚未被此下界排除；这里只保留空间，后续才可检验容量、写入成本与合法时间信息。")

    rows = []
    for i in range(n):
        rows.append({
            "window_index": i,
            "trace_start_s": 0.0 if i == 0 else elapsed[i-1],
            "trace_end_s": elapsed[i],
            "window_duration_s": duration[i],
            "phase": "train_and_warmup" if i < eval_index else "evaluation",
            "put_dt_pct": put_dt[i],
            "first_get_dt_pct": first_dt[i],
            "lower_bound_dt_pct": lower_dt[i],
            "baleen_get_dt_pct_including_prefetch": to_dt(baleen_get)[i],
            "baleen_actual_dt_pct": baleen_dt[i],
            "baleen_minus_lower_bound_pp": baleen_dt[i] - lower_dt[i],
            "scan_put_service_seconds_sample": scan_put[i],
            "scan_first_get_service_seconds_sample": first_st[i],
            "scan_get_nocache_service_seconds_sample": scan_nocache[i],
            "reference_put_service_seconds_sample": baleen_put[i],
            "reference_get_nocache_service_seconds_sample": delta(baleen["arrays"]["service_time_nocache_stats"])[i],
            "first_get_requests": int(first_ios[i]),
            "first_get_demand_chunks": int(first_chunks[i]),
            "seen_get_blocks_cumulative": int(seen_blocks[i]),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "BC02_all_windows.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    max_put_pp = max(e["put_dt_percentage_points"] for e in errors.values())
    max_nocache_pp = max(e["get_nocache_dt_percentage_points"] for e in errors.values())
    max_put_s = max(e["put_seconds"] for e in errors.values())
    max_nocache_s = max(e["get_nocache_seconds"] for e in errors.values())
    scan_seconds = time.perf_counter() - started
    metadata = {
        "protocol_passed": True, "manual_checks": tests, "reference_checks": errors,
        "dt_inequality_max_positive_violation_pp": violation,
        "service_time_tolerance_s": ST_ATOL, "dt_tolerance_pp": DT_ATOL,
        "n_windows": n, "evaluation_index_start": eval_index,
        "evaluation_actual_start_s": float(elapsed[eval_index-1]),
        "requests": scanner.request_count, "unique_get_blocks": len(scanner.seen_get_blocks),
        "parsed_start": parsed_start, "parsed_end": parsed_end,
        "p_baleen_pct": p_baleen, "p_lb_pct": p_lb,
        "upper_gain_fraction": upper_gain, "baleen_peak_window": idx_baleen,
        "lb_peak_window": idx_lb, "verdict": verdict, "wall_seconds": scan_seconds,
        "inputs_sha256": {str(ref['path'].relative_to(REPO)): hashlib.sha256(ref['path'].read_bytes()).hexdigest() for ref in refs},
    }
    (WORK / "audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# BC02：Region1 不可避免负载下界诊断

逐窗账目核对通过。{verdict}

| 指标 | 数值 |
|---|---:|
| Baleen 峰值 P_Baleen | {p_baleen:.6f}% DT |
| 下界峰值 P_LB | {p_lb:.6f}% DT |
| 相对改进上限 U_gain | {upper_gain*100:.6f}% |
| 下界峰值所在窗口（0-based） | {idx_lb} |
| 三项计量核对最大误差 | PUT {max_put_pp:.3e} pp；无缓存 GET {max_nocache_pp:.3e} pp；下界正向超出 Baleen {violation:.3e} pp |

评价段为索引 {eval_index}–{n-1}，共 {int(mask.sum())} 个窗口，实际起点 {elapsed[eval_index-1]:.4f} 秒，沿用 BC01 在跨过第一天边界后完整统计窗口开始的口径。下界峰值在 {rows[idx_lb]['trace_start_s']:.4f}–{rows[idx_lb]['trace_end_s']:.4f} 秒；Baleen 峰值窗口为 {idx_baleen}。先对全部评价窗口取最大值，再计算相对改进上限。

下界峰值窗口 {idx_lb} 由 PUT {put_dt[idx_lb]:.6f}% 和首次 GET {first_dt[idx_lb]:.6f}% 组成。原 Baleen 峰值窗口 {idx_baleen} 的下界为 {lower_dt[idx_baleen]:.6f}%；下界峰值从 {idx_baleen} 转移到 {idx_lb}，因此不能只用窗口 {idx_baleen} 判读空间。

动作范围冻结为：初始缓存为空，从轨迹起点连续处理；只允许 GET 时准入当前块和块内预取；不允许 PUT 准入、跨块主动预取或预先装载。PUT 成本保持作者定义。假想系统在某个块首次 GET 时支付需求读取，然后免费永久缓存整个块；其后该块所有 GET 免费，不计容量、额外读取和写入成本。已发生 GET 的块集合贯穿第一天预热及评价段，遇到 PUT 或 checkpoint 均不清空。

扫描使用作者 `read_processed_file_list_accesses()` 与 `AccessPlus` 解析、重复展开、逻辑排序和块键。Region1 的路径由作者判定为带 pipeline 格式；一般 hostname 组合键语义也经微型输入验证。窗口边界直接调用作者 `CacheSimulator._stats()`，在请求前触发计量 checkpoint；扫描结尾按原顺序结算。仅将 checkpoint 的缓存/模型/日志部分替换为计数器输出，保留作者 `Stats.append/checkpoint_many` 和窗口索引更新。时间端点、逻辑请求位置和每窗 GET/PUT 请求数及 chunk 数，与两份原始统计均逐项精确一致。

PUT 直接调用作者 `record_service_time_put()`。无缓存 GET 使用 `_log_st()` 中的 `service_time(1, len(acc.chunks))`；首次 GET 调用 `record_service_time_get(acc.chunks, [], acc)`，仅计需求读取。DT 使用作者 `st_to_util(...)*100`，采样值为 0.1（即 0.1%），36 个磁盘，并按实际 checkpoint 时间差归一化，复用 BC01 的完整 GET+PUT 口径。

三个指定手工序列均先于 Region1 扫描通过：首次 GET 必须收费；同块跨窗口后续 GET 免费，即使请求其他 chunk；先 PUT 后首次 GET 分别收费。另核对 hostname/重复展开、请求前边界与末尾结算、必需字段缺失报错。Region1 原始 trace 由作者解析器读入一次，排序后离线遍历 {scanner.request_count:,} 个请求一次，观察到 {len(scanner.seen_get_blocks):,} 个发生 GET 的不同块。没有运行新的缓存策略或模型训练。

计量核对覆盖全部 {n} 个窗口，包括预热。相对 RejectX、Baleen 两份原始 `.stats.lzma`，逐窗 PUT 服务时间最大绝对差为 {max_put_s:.3e} 秒，无缓存 GET 为 {max_nocache_s:.3e} 秒；逐窗下界超出 Baleen 的最大正向差为 {violation:.3e} pp。原始服务时间容差固定为 {ST_ATOL:.0e} 秒，DT 不等式容差为 {DT_ATOL:.0e} pp，仅用于浮点舍入。PUT、GET、时间和请求计数字段缺失、长度不匹配或非有限值均报错，不以零代替。

本轮按事先约定的 5% 相对峰值下降筛查尺度判读。这是工程投入门槛，非统计显著性门槛。下界是当前动作范围中的乐观成本计量，U_gain 是峰值改进空间的上限，不能当作可部署成绩或可学习收益；扩展动作范围后本轮裁决须重新计算。BC01 的 `write_mbps=0` 没有统一硬写入限额，因此其参照比较仅为实际写入量接近的官方配置比较。

输入冻结为 BC01 的 Region1 trace（SHA-1 `BC1BF6B5036FDB41D52F547C7820DC23C5D91B57`）、仓库 HEAD `4e3a9205b1f1bd72bffd66dd52bd5b06509089e9`、BCacheSim `ddeb2d8035483b5943fa57df1932ffc7d1134b6d`。支持脚本在 `work/bc02/scan_bc02.py`；逐项审计数值、输入统计哈希在 `work/bc02/audit.json`。离线验证与扫描共约 {scan_seconds:.2f} 秒。

`BC02_all_windows.csv` 每行一个窗口，包含 PUT、首次 GET、下界与 Baleen 实际 DT；另保留原始服务时间秒数、无缓存 GET 核对值、首次 GET 请求/chunk 数和已见 GET 块累计数。所有百分比列按百分数值存储（如 39.7575 表示 39.7575%）；`*_pp` 表示百分点。原始模拟统计、trace 和作者代码未修改。
"""
    report_path = OUT / "BC02_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("Wrote", report_path, "and", csv_path)


if __name__ == "__main__":
    main()
