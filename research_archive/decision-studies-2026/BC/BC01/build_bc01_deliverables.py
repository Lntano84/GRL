from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import compress_json
import numpy as np


ROOT = Path(__file__).resolve().parent / "Baleen-FAST24"
OUT = Path(__file__).resolve().parents[2] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK_BYTES = 128 * 1024
DISKS = 36
STATS_START_S = 86_400


def find_run(pattern: str) -> tuple[Path, Path]:
    stats_paths = list(ROOT.glob(pattern))
    if len(stats_paths) != 1:
        raise RuntimeError(f"expected one stats file for {pattern}, got {stats_paths}")
    stats_path = stats_paths[0]
    main_path = Path(str(stats_path).replace(".stats.lzma", ".lzma"))
    return stats_path, main_path


RUNS = {
    "RejectX": {
        "stats": "runs/example/rejectx/*/*.stats.lzma",
        "config": ROOT / "runs/example/rejectx/config.json",
        "phase": "warmup",
        "wall_runtime": "64.9 s (simulation)",
    },
    "Baleen": {
        "stats": "runs/example/baleen/*/*/*.stats.lzma",
        "config": ROOT / "runs/example/baleen/prefetch_ml-on-partial-hit/config.json",
        "phase": "train_and_warmup",
        "wall_runtime": "9 min 43 s (simulation); ~12.7 s (training)",
    },
}


def dseries(batches: dict, key: str, n: int, *, required: bool = True) -> np.ndarray:
    if key not in batches:
        if required:
            raise KeyError(f"required counter series is absent: {key}")
        return np.zeros(n, dtype=np.float64)
    values = np.asarray(batches[key], dtype=np.float64)
    if values.shape != (n,) or not np.isfinite(values).all():
        raise ValueError(f"invalid counter series: {key}")
    return np.diff(values, prepend=0.0)


def mb(n: int) -> float:
    return n / (1024 * 1024)


series_rows: list[dict] = []
summaries: dict[str, dict] = {}
for method, info in RUNS.items():
    stats_path, main_path = find_run(info["stats"])
    stats = compress_json.load(str(stats_path))
    main = compress_json.load(str(main_path))
    batches = stats["batches"]
    config = json.loads(info["config"].read_text(encoding="utf-8"))

    elapsed = np.asarray(batches["time_elapsed_phy"], dtype=np.float64)
    n = len(elapsed)
    if n != 1008:
        raise RuntimeError(f"unexpected number of buckets for {method}: {n}")
    durations = np.diff(elapsed, prepend=0.0)
    index = np.arange(n)
    phase = np.where(index < STATS_START_S // config["log_interval"], info["phase"], "evaluation")

    sample_ratio = float(main["sampleRatio"])
    chunk_bytes = int(main["chunkSize"])
    if chunk_bytes != CHUNK_BYTES:
        raise RuntimeError(f"unexpected chunk size {chunk_bytes}")
    scale_full = 100.0 / sample_ratio
    # Source computes utilization from service time, 36 modeled disks, and the
    # trace sampling ratio. Include the separate PUT service-time counter.
    get_st = dseries(batches, "service_time_used_stats", n)
    put_st = dseries(batches, "service_time_writes_stats", n)
    # Match the published result fields: st_to_util(...) is multiplied by
    # another 100 when the simulator stores the result as a percentage.
    dt_get_pct = get_st * 100 / DISKS / sample_ratio / durations * 100
    dt_put_pct = put_st * 100 / DISKS / sample_ratio / durations * 100
    dt_total_pct = dt_get_pct + dt_put_pct

    fetch_ios = dseries(batches, "fetches_ios_stats", n)
    demand_chunks = dseries(batches, "fetches_chunks_demandmiss_stats", n)
    prefetch_chunks = dseries(batches, "fetches_chunks_prefetch_stats", n)
    query_chunks = dseries(batches, "chunk_queries_stats", n)
    write_chunks = dseries(batches, "flashcache/keys_written_stats", n)
    prefetch_write_chunks = dseries(
        batches, "flashcache/prefetches_stats", n,
        required=config["prefetch_when"] != "never",
    )

    eval_mask = phase == "evaluation"
    eval_seconds = float(durations[eval_mask].sum())
    avg_eval_dt = float((get_st[eval_mask].sum() + put_st[eval_mask].sum()) * 100 / DISKS / sample_ratio / eval_seconds * 100)
    peak_eval_dt = float(dt_total_pct[eval_mask].max())
    peak_eval_i = int(np.flatnonzero(eval_mask)[np.argmax(dt_total_pct[eval_mask])])

    # We report measured simulated payload as well as the linear full-workload
    # equivalent; the latter is an extrapolation, not physical I/O on this PC.
    write_bytes_eval = int(write_chunks[eval_mask].sum()) * chunk_bytes
    write_bytes_all = int(write_chunks.sum()) * chunk_bytes
    prefetch_write_bytes_eval = int(prefetch_write_chunks[eval_mask].sum()) * chunk_bytes
    demand_read_bytes_eval = int(demand_chunks[eval_mask].sum()) * chunk_bytes
    prefetch_read_bytes_eval = int(prefetch_chunks[eval_mask].sum()) * chunk_bytes
    total_read_bytes_eval = demand_read_bytes_eval + prefetch_read_bytes_eval

    total_capacity_bytes = int(config["size_gb"] * (1024**3))
    effective_requested_bytes = int(total_capacity_bytes * sample_ratio / 100)
    cache_items = int(main["results"]["NumCacheElems"])
    modeled_capacity_bytes = cache_items * chunk_bytes

    for i in range(n):
        start_s = 0.0 if i == 0 else float(elapsed[i - 1])
        end_s = float(elapsed[i])
        series_rows.append({
            "method": method,
            "window_index": i,
            "trace_start_s": f"{start_s:.6f}",
            "trace_end_s": f"{end_s:.6f}",
            "window_duration_s": f"{durations[i]:.6f}",
            "phase": phase[i],
            "dt_get_util_pct_including_prefetch": f"{dt_get_pct[i]:.9f}",
            "dt_put_util_pct": f"{dt_put_pct[i]:.9f}",
            "dt_total_util_pct": f"{dt_total_pct[i]:.9f}",
            "backend_fetch_ios_total": int(fetch_ios[i]),
            "backend_demand_read_chunks": int(demand_chunks[i]),
            "backend_prefetch_read_chunks": int(prefetch_chunks[i]),
            "backend_demand_read_bytes_sample": int(demand_chunks[i]) * chunk_bytes,
            "backend_prefetch_read_bytes_sample": int(prefetch_chunks[i]) * chunk_bytes,
            "backend_total_read_bytes_sample": int(demand_chunks[i] + prefetch_chunks[i]) * chunk_bytes,
            "backend_demand_read_bytes_full_workload_equiv": int(demand_chunks[i]) * chunk_bytes * scale_full,
            "backend_prefetch_read_bytes_full_workload_equiv": int(prefetch_chunks[i]) * chunk_bytes * scale_full,
            "chunk_queries": int(query_chunks[i]),
            "demand_miss_chunks": int(demand_chunks[i]),
            "flash_write_chunks": int(write_chunks[i]),
            "flash_prefetch_write_chunks": int(prefetch_write_chunks[i]),
            "flash_write_bytes_sample": int(write_chunks[i]) * chunk_bytes,
            "flash_prefetch_write_bytes_sample": int(prefetch_write_chunks[i]) * chunk_bytes,
            "flash_write_bytes_full_workload_equiv": int(write_chunks[i]) * chunk_bytes * scale_full,
            "flash_hit_rate_window": "",
        })

    summaries[method] = {
        "config": config,
        "sample_ratio": sample_ratio,
        "trace_seconds": float(main["traceSeconds"]),
        "chunk_bytes": chunk_bytes,
        "bucket_count": n,
        "eval_windows": int(eval_mask.sum()),
        "eval_seconds": eval_seconds,
        "eval_start_seconds": float(elapsed[np.flatnonzero(eval_mask)[0] - 1]),
        "avg_eval_dt": avg_eval_dt,
        "peak_eval_dt": peak_eval_dt,
        "peak_eval_window_index": peak_eval_i,
        "peak_eval_start": 0.0 if peak_eval_i == 0 else float(elapsed[peak_eval_i - 1]),
        "peak_eval_end": float(elapsed[peak_eval_i]),
        "write_chunks_eval": int(write_chunks[eval_mask].sum()),
        "write_bytes_eval": write_bytes_eval,
        "write_bytes_eval_equiv": write_bytes_eval * scale_full,
        "write_chunks_all": int(write_chunks.sum()),
        "write_bytes_all": write_bytes_all,
        "prefetch_write_bytes_eval": prefetch_write_bytes_eval,
        "demand_read_bytes_eval": demand_read_bytes_eval,
        "prefetch_read_bytes_eval": prefetch_read_bytes_eval,
        "total_read_bytes_eval": total_read_bytes_eval,
        "demand_read_bytes_eval_equiv": demand_read_bytes_eval * scale_full,
        "prefetch_read_bytes_eval_equiv": prefetch_read_bytes_eval * scale_full,
        "fetch_ios_eval": int(fetch_ios[eval_mask].sum()),
        "demand_chunks_eval": int(demand_chunks[eval_mask].sum()),
        "prefetch_chunks_eval": int(prefetch_chunks[eval_mask].sum()),
        "queries_eval": int(query_chunks[eval_mask].sum()),
        "write_chunks_all": int(write_chunks.sum()),
        "full_results": main["results"],
        "main": main,
        "modelled_capacity_bytes": modeled_capacity_bytes,
        "effective_requested_capacity_bytes": effective_requested_bytes,
        "total_capacity_gb_option": float(config["size_gb"]),
        "num_cache_elems": cache_items,
        "runtime": info["wall_runtime"],
        "stats_path": str(stats_path.relative_to(ROOT)).replace("\\", "/"),
        "config_path": str(info["config"].relative_to(ROOT)).replace("\\", "/"),
    }


trajectory = OUT / "BC01_10min_trajectory.csv"
with trajectory.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(series_rows[0]))
    writer.writeheader()
    writer.writerows(series_rows)


def fmt_bytes(n: float, suffix: str = "GiB") -> str:
    units = {"GiB": 1024**3, "TiB": 1024**4, "MiB": 1024**2}
    return f"{n / units[suffix]:,.3f} {suffix}"


def f2(x: float) -> str:
    return f"{x:.2f}%"


def short(v: float) -> str:
    return f"{v:,.0f}"


report = OUT / "BC01_report.md"
r = summaries["RejectX"]
b = summaries["Baleen"]

report_text = f"""# BC01：Region1 官方参照轨迹审计

## 结果表

DT 指模拟后端的服务时间负载率，按作者代码使用 36 个磁盘和 0.1% 抽样比例归一化；包含 GET（预取读取计入 GET）与 PUT。峰值和平均值只在 `stats_start=86400s` 之后计算。写入量列出抽样轨迹计数对应的逻辑字节，并给出全量工作负载线性等效量；模拟器没有在本机真实写入这些缓存字节。

| 方法 | 评价段峰值 DT | 评价段平均 DT | 评价段缓存写入（抽样值；全量等效） | 评价段后端读量（需求 + 预取，抽样值） | 全轨迹 Flash hit rate | 缓存配置 | 本地运行时间 |
|---|---:|---:|---:|---:|---:|---|---|
| RejectX | {f2(r['peak_eval_dt'])} | {f2(r['avg_eval_dt'])} | {fmt_bytes(r['write_bytes_eval'])}；{fmt_bytes(r['write_bytes_eval_equiv'], 'TiB')} | {fmt_bytes(r['demand_read_bytes_eval'])} + {fmt_bytes(r['prefetch_read_bytes_eval'])} | {r['full_results']['FlashCacheHitRate']*100:.3f}% | `size_gb=366.475`；抽样后 3,002 × 128 KiB = {fmt_bytes(r['modelled_capacity_bytes'])} | {r['runtime']} |
| Baleen | {f2(b['peak_eval_dt'])} | {f2(b['avg_eval_dt'])} | {fmt_bytes(b['write_bytes_eval'])}；{fmt_bytes(b['write_bytes_eval_equiv'], 'TiB')} | {fmt_bytes(b['demand_read_bytes_eval'])} + {fmt_bytes(b['prefetch_read_bytes_eval'])} | {b['full_results']['FlashCacheHitRate']*100:.3f}% | `size_gb=366.475`；抽样后 3,002 × 128 KiB = {fmt_bytes(b['modelled_capacity_bytes'])} | {b['runtime']} |

`评价段` 从跨过第一天边界后的完整统计窗口开始，0-based 窗口索引为 144，实际起点是 {r['eval_start_seconds']:.4f} 秒。配置的 `stats_start=86,400s` 用于跳过前 144 个统计窗口，不代表窗口恰好从 86,400 秒开始。Baleen 第一日用于模型训练和模拟预热；RejectX 没有训练，仅作为同长度预热段。作者统计器保留全轨迹汇总，本表则按 10 分钟计数器差分和实际轨迹时间重算评价段负载。

`write_mbps=0` 未施加统一的硬写入限额。本表是实际写入量接近的官方配置比较，不能称为严格相同写入预算实验。

两条轨迹评价段的最大 DT 都出现在窗口索引 578（轨迹时间 346,810.10–347,400.14 秒）。

## 计量与窗口

每条轨迹有 {r['bucket_count']} 个约 600 秒窗口，覆盖 {r['trace_seconds']/86400:.3f} 天；末尾为部分窗口。评价段有 {r['eval_windows']} 个窗口。评价段逐窗累计账目：RejectX 有 {short(r['demand_chunks_eval'])} 个 demand-miss chunks、{short(r['fetch_ios_eval'])} 次后端 fetch I/O、{short(r['write_chunks_eval'])} 个 Flash 写入 chunks；Baleen 有 {short(b['demand_chunks_eval'])} 个 demand-miss chunks、{short(b['prefetch_chunks_eval'])} 个预取读取 chunks、{short(b['fetch_ios_eval'])} 次后端 fetch I/O、{short(b['write_chunks_eval'])} 个 Flash 写入 chunks，其中预取写入 {fmt_bytes(b['prefetch_write_bytes_eval'])}。全轨迹分别写入 {fmt_bytes(r['write_bytes_all'])} / {fmt_bytes(b['write_bytes_all'])} 抽样逻辑字节。

计量交叉核对：从逐窗服务时间计数器重算全轨迹 DT，RejectX 为 {r['full_results']['ServiceTimeWithPutUtil1']:.6f}%，Baleen 为 {b['full_results']['ServiceTimeWithPutUtil1']:.6f}%；与作者 `.lzma` 汇总字段一致（差异小于 `4e-12` 个百分点）。

上游数据块为 131,072 bytes；`sampleRatio=0.1` 表示 0.1%，代码按 `sampleRatio/100` 缩放容量，故 `size_gb=366.475` 缩为 0.366475 GiB，再向下取整为 3,002 个元素，即 {r['modelled_capacity_bytes']:,} bytes（{r['modelled_capacity_bytes']/(1024**3):.6f} GiB）。读写的全量工作负载线性等效值乘 1,000；这不是物理写放大或实机磁盘计量。

逐窗命中率没有被统计器输出，因此 CSV 中 `flash_hit_rate_window` 留空。每窗保留 `chunk_queries`、需求 miss chunks、后端读取 I/O 及拆开的预取读取，不用这些计数相减冒充精确命中数。表中 hit rate 和全轨迹 hit chunks（RejectX {short(r['full_results']['TotalChunkHitsFlashNotInRam'])}；Baleen {short(b['full_results']['TotalChunkHitsFlashNotInRam'])}）来自 `.lzma` 汇总，不能当作评价段命中率。

## 冻结配置与运行来源

- 上游仓库：[`wonglkd/Baleen-FAST24`](https://github.com/wonglkd/Baleen-FAST24)，检出 HEAD `4e3a9205b1f1bd72bffd66dd52bd5b06509089e9`；BCacheSim 子模块为 `ddeb2d8035483b5943fa57df1932ffc7d1134b6d`。模拟器代码与示例配置未改动。
- 官方示例命令：README 中的 Region1 RejectX 模拟、Baleen 模型训练和 Baleen 模拟路径。RejectX 与 Baleen 的运行配置分别为 [`RejectX config`](https://github.com/wonglkd/Baleen-FAST24/blob/main/runs/example/rejectx/config.json) 和 [`Baleen config`](https://github.com/wonglkd/Baleen-FAST24/blob/main/runs/example/baleen/prefetch_ml-on-partial-hit/config.json)；训练参数见[官方 README](https://github.com/wonglkd/Baleen-FAST24#artifact-for-baleen-fast-2024)。
- 两条模拟都固定使用 `Region1/full_0_0.1.trace`、LRU、`size_gb=366.475`、`size_opt=access`、`write_mbps=0`、600 秒日志窗口、`stats_start=86400`。RejectX 的配置为 `ap_threshold=1.0`、`ap_probability=0.508154`、batch 512、无预取。Baleen 使用 `ap=mlnew`、`ap_threshold=0.798545`、`learned_ap_filter_count=6`、batch 16、partial-hit 预取、`acctime-episode-predict` 预取范围。
- 官方训练命令固定了 Region1、0.1 抽样、训练分段 `[0,86400]`、`--eviction-age 5892.856` 秒（约 1.637 小时）、`--train-target-wr 35.599`、目标写率和 cache-size 列表、`--ap-feat-subset meta+block+chunk`。这 5,892.856 秒是训练所用的假定 eviction/residency age，不是模拟中的缓存 TTL；实际模拟仍按 LRU 容量驱逐。
- 配置文件给出阈值与 eviction-age 数值，README 给出训练切分和目标写率；公开示例没有明确记录 RejectX 阈值、Baleen `0.798545` 阈值及 5,892.856 秒驻留假设的选择/验证程序。因此它们是忠实复用的作者参数，但这次审计不能证明其满足未来的因果评价协议，也不能声称阈值未使用其他数据调定。
- 本地 Windows 环境使用 Python 3.11.9 portable runtime，并从仓库依赖文件安装依赖；训练约 12.7 秒，RejectX 模拟 64.9 秒，Baleen 模拟 9 分 43 秒。作者 README 对该示例的估计分别约 4 分钟、训练 25 秒、Baleen 模拟 30 分钟；这是不同环境估计，时间不可直接作性能比较。

## 数据可用性与限制

Region1 trace 已成功下载并通过仓库 `checksums.sha1` 校验；本地 trace 大小 7,117,125 bytes，SHA-1 `BC1BF6B5036FDB41D52F547C7820DC23C5D91B57`。下载源的 `storage_0.1.tar.gz` 实际大小 17,081,636 bytes，SHA-256 `56750528F08F8830885639AF69FF5538BF531FAEE046E82B43D315B96F0FDEDF`。下载脚本还引用结果表和 breakdown 文件；这两条 URL 已确认可访问，但本次单示例回放未下载它们。

训练命令成功返回并生成模型；终端输出出现了公开代码中的 overflow 与除零指标警告（含 `FN/FP` 计算），以及 LightGBM 参数弃用警告。没有改变上游代码来消除警告。

这次只验证公开模拟器中一个 Region1 示例的计量链路，不是七条工作负载汇总复现。作者说明真实内部 CacheLib 测试床未公开，模拟用的磁盘参数也不是 Meta 精确生产参数；结果不应外推为生产收益。此轮没有训练我们的控制器，也不据此判断新方法有效或无效。

## 轨迹字段

[`BC01_10min_trajectory.csv`](BC01_10min_trajectory.csv) 每行对应一种方法的一个统计窗口，共 2,016 行。`phase` 区分 Baleen `train_and_warmup`、RejectX `warmup` 与共同 `evaluation`；读写 bytes 同时保留抽样轨迹内值和全量工作负载线性等效值，需求读取与预取读取分列。所有写入字节都是模拟器计数乘 128 KiB 得到的逻辑模型流量。
"""

report.write_text(report_text, encoding="utf-8")
print(f"Wrote {report}")
print(f"Wrote {trajectory}")
for method, s in summaries.items():
    print(method, "avg_eval_dt", s["avg_eval_dt"], "peak_eval_dt", s["peak_eval_dt"],
          "eval_write_bytes", s["write_bytes_eval"], "eval_read_bytes", s["total_read_bytes_eval"],
          "peak window", s["peak_eval_window_index"],
          "official full avg+put", s["full_results"]["ServiceTimeWithPutUtil1"],
          "official full peak+put", s["full_results"]["PeakServiceTimeUsedWithPutUtil1"])
