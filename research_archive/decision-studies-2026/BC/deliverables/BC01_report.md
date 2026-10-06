# BC01：Region1 官方参照轨迹审计

## 结果表

DT 指模拟后端的服务时间负载率，按作者代码使用 36 个磁盘和 0.1% 抽样比例归一化；包含 GET（预取读取计入 GET）与 PUT。峰值和平均值只在 `stats_start=86400s` 之后计算。写入量列出抽样轨迹计数对应的逻辑字节，并给出全量工作负载线性等效量；模拟器没有在本机真实写入这些缓存字节。

| 方法 | 评价段峰值 DT | 评价段平均 DT | 评价段缓存写入（抽样值；全量等效） | 评价段后端读量（需求 + 预取，抽样值） | 全轨迹 Flash hit rate | 缓存配置 | 本地运行时间 |
|---|---:|---:|---:|---:|---:|---|---|
| RejectX | 45.10% | 18.95% | 18.405 GiB；17.974 TiB | 251.182 GiB + 0.000 GiB | 23.684% | `size_gb=366.475`；抽样后 3,002 × 128 KiB = 0.366 GiB | 64.9 s (simulation) |
| Baleen | 39.76% | 18.44% | 18.156 GiB；17.730 TiB | 249.083 GiB + 5.090 GiB | 24.944% | `size_gb=366.475`；抽样后 3,002 × 128 KiB = 0.366 GiB | 9 min 43 s (simulation); ~12.7 s (training) |

`评价段` 从跨过第一天边界后的完整统计窗口开始，0-based 窗口索引为 144，实际起点是 86401.2328 秒。配置的 `stats_start=86,400s` 用于跳过前 144 个统计窗口，不代表窗口恰好从 86,400 秒开始。Baleen 第一日用于模型训练和模拟预热；RejectX 没有训练，仅作为同长度预热段。作者统计器保留全轨迹汇总，本表则按 10 分钟计数器差分和实际轨迹时间重算评价段负载。

`write_mbps=0` 未施加统一的硬写入限额。本表是实际写入量接近的官方配置比较，不能称为严格相同写入预算实验。

两条轨迹评价段的最大 DT 都出现在窗口索引 578（轨迹时间 346,810.10–347,400.14 秒）。

## 计量与窗口

每条轨迹有 1008 个约 600 秒窗口，覆盖 6.998 天；末尾为部分窗口。评价段有 864 个窗口。评价段逐窗累计账目：RejectX 有 2,057,683 个 demand-miss chunks、72,708 次后端 fetch I/O、150,774 个 Flash 写入 chunks；Baleen 有 2,040,485 个 demand-miss chunks、41,697 个预取读取 chunks、62,662 次后端 fetch I/O、148,731 个 Flash 写入 chunks，其中预取写入 4.172 GiB。全轨迹分别写入 21.035 GiB / 20.815 GiB 抽样逻辑字节。

计量交叉核对：从逐窗服务时间计数器重算全轨迹 DT，RejectX 为 18.762498%，Baleen 为 18.176565%；与作者 `.lzma` 汇总字段一致（差异小于 `4e-12` 个百分点）。

上游数据块为 131,072 bytes；`sampleRatio=0.1` 表示 0.1%，代码按 `sampleRatio/100` 缩放容量，故 `size_gb=366.475` 缩为 0.366475 GiB，再向下取整为 3,002 个元素，即 393,478,144 bytes（0.366455 GiB）。读写的全量工作负载线性等效值乘 1,000；这不是物理写放大或实机磁盘计量。

逐窗命中率没有被统计器输出，因此 CSV 中 `flash_hit_rate_window` 留空。每窗保留 `chunk_queries`、需求 miss chunks、后端读取 I/O 及拆开的预取读取，不用这些计数相减冒充精确命中数。表中 hit rate 和全轨迹 hit chunks（RejectX 733,638；Baleen 772,652）来自 `.lzma` 汇总，不能当作评价段命中率。

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
