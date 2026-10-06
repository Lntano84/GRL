# BC04：原峰值未命中的历史来源

一次原 STATIC-BASE 回放和只读日志审计通过。分类是历史事实，不是拒绝错误或可盈利保留的认证；本轮没有新策略、参数搜索、训练或反事实回放。

## 范围与计费

- 沿用 Region1 完整轨迹、原 Baleen 模型、原阈值 0.798545、3,002 个配置槽位、LRU、原预取与原计量。作者配置仅更改输出目录。write_mbps=0 保持原定义。
- 从轨迹起点连续维护 GET 与 chunk 事件历史，包括预热；评价窗口为 144–1007，实际起点 86401.23277902603 秒。
- 每个 GET 在 run_get 入口冻结此前是否有块 GET、chunk 实际写入／拒绝／候选／驱逐／预取资格检查记录，以及待处理队列状态。当前请求的所有行为只供后续请求使用。
- 实际写入在 QueueCache.admit 成功且写入计数增一后记录；实际拒绝在 process_admit_buffer 返回后，核对拒绝计数增量才记录。check_only 资格检查单独落盘，仅更新资格检查历史，不形成真实写入、拒绝或候选。
- FIRST 优先于 ADMITTED-BEFORE，再到 REJECTED-BEFORE，最后 OTHER。FIRST 指整条轨迹第一次块 GET，不能按 chunk 首次读取或评价开始重新定义。
- 作者将待处理准入队列视为命中。本轮使用实际 need_fetch 做缺失分类；连续需求读取范围内的桥接 chunk 不额外造出缺失分类。
- 一次需求读取的 service_time(1, max(need_fetch)−min(need_fetch)+1) 全部归给请求类别。跨类别缺失归 MIXED，保留组合，完整 seek 不做 chunk 分摊。额外预取 service_time(0, 扩展读取范围块数) 分列。
- DT 单位为模拟磁盘负载百分点：窗口 578 的逐请求贡献相加为该窗口负载；完整评价段按评价实际总时长归一化为平均负载。占比的分母仅为该范围需求读取成本，不含 PUT 或额外预取。数值是模拟计量，不是本机磁盘 I/O 或生产收益。

## 窗口 578

| 请求类别 | 未命中请求数 | 需求读取 DT | 占该范围需求读取 DT | 额外预取 DT |
| --- | ---: | ---: | ---: | ---: |
| FIRST | 27 | 4.293015% | 15.9521% | 0.000000% |
| ADMITTED-BEFORE | 38 | 5.040811% | 18.7307% | 0.045267% |
| REJECTED-BEFORE | 77 | 10.802421% | 40.1398% | 0.502053% |
| OTHER | 44 | 4.538494% | 16.8642% | 0.604933% |
| MIXED | 17 | 2.237231% | 8.3131% | 0.000000% |

完全命中：204 次，需求读取和额外预取成本均为零。需求总 DT=26.911971524%，额外预取 DT=1.152252993%，合计 GET DT=28.064224517%。

## 完整评价段

| 请求类别 | 未命中请求数 | 需求读取 DT | 占该范围需求读取 DT | 额外预取 DT |
| --- | ---: | ---: | ---: | ---: |
| FIRST | 10,840 | 2.250377% | 16.7653% | 0.000000% |
| ADMITTED-BEFORE | 18,661 | 4.344128% | 32.3637% | 0.064583% |
| REJECTED-BEFORE | 20,142 | 5.044777% | 37.5835% | 0.031687% |
| OTHER | 6,990 | 0.898713% | 6.6954% | 0.089565% |
| MIXED | 6,029 | 0.884841% | 6.5921% | 0.009530% |

完全命中：46,656 次，需求读取和额外预取成本均为零。需求总 DT=13.422835659%，额外预取 DT=0.195364085%，合计 GET DT=13.618199744%。

## MIXED 与 OTHER 的补充事实

| 范围 | MIXED 类别组合 | 请求数 | 抽样需求服务秒 | 抽样预取服务秒 |
| --- | --- | ---: | ---: | ---: |
| evaluation | REJECTED-BEFORE+OTHER | 5,264 | 133.200335664 | 1.543706294 |
| evaluation | ADMITTED-BEFORE+REJECTED-BEFORE | 701 | 29.054506993 | 0.205419580 |
| evaluation | ADMITTED-BEFORE+OTHER | 43 | 1.742751748 | 0.028846154 |
| evaluation | ADMITTED-BEFORE+REJECTED-BEFORE+OTHER | 21 | 1.084157343 | 0.000000000 |
| peak_578 | REJECTED-BEFORE+OTHER | 17 | 0.475220280 | 0.000000000 |

| 范围 | OTHER 实际缺失 chunk | 曾为实际候选 | 请求前仍在队列 | 仅有先前资格检查 | 无候选且无资格检查 |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_578 | 747 | 0 | 0 | 0 | 747 |
| evaluation | 171,869 | 0 | 0 | 1,333 | 170,536 |

上述补充项按实际缺失 chunk 计数，不分摊请求 seek，不用于计算可恢复收益。ADMITTED-BEFORE 仅说明曾实际写入现在缺失；REJECTED-BEFORE 仅说明实际拒绝发生过；OTHER 仅说明未有实际写入或拒绝记录。

## 计量与历史审计

- 恰好一次完整回放，耗时 658.62 秒，低于 1,800 秒预算；GET 127,305 次、原请求总数 147,794。
- 逐窗 GET、PUT、需求与预取成本、写入、预取读取、需求读取、请求计数和作者窗口端点与 BC03 STATIC-BASE 对账，最大误差 0。
- 请求成本逐窗对账最大误差：{'demand_service_time_s': 7.978506744166225e-12, 'prefetch_service_time_s': 1.4210854715202004e-14, 'total_get_service_time_s': 8.15347789284715e-12, 'total_get_dt_pp': 3.7524981082692874e-11}。容差为服务时间 1e-09 秒、DT 1e-08 个百分点；整数计数与时间端点精确相等。
- 2,324,894 个实际缺失 chunk 均恰好归入一个类别。离线从每个请求的实际候选、实际写入、真实拒绝、驱逐记录及单独资格日志重新构造历史，逐一核对冻结事实；不复用日志器的 History 对象或分类函数。
- 手工序列通过：current_request_events_do_not_backdate, check_only_accept_and_reject_do_not_make_real_history, first_block_get_priority, admission_over_rejection_priority, new_chunk_on_seen_block_other, pending_and_prior_candidate_preserved, mixed_request_not_split, hit_zero_cost, hostname_identity_preserved。包括当前拒绝回填、资格检查污染真实准入历史、类别优先级、跨主机键、MIXED 与完全命中。
- 源码保持原样，输入、已训练模型和配置 SHA-256 前后相同；完整审计见 BC04_audit.json。

- 导出复核通过：主表 12 行（包括 HIT）、峰值窗口 407 个 GET、3,869 个实际缺失 chunk；CSV 与压缩请求 JSONL 的历史明细一致，逐类别成本合计与主表一致。

## 交付

- [主表 CSV](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_summary.csv)：两个范围的全精度主表，另有 HIT 行。
- [峰值窗口逐请求 CSV](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_peak_578_requests.csv)：窗口 578 的所有 GET，包括完全命中，保留实际缺失 chunk 的完整历史。
- [峰值窗口缺失 chunk 历史 CSV](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_peak_578_chunks.csv)：按请求序号关联到逐请求明细，不给 chunk 分摊 seek 或归因收益。
- [峰值窗口完整事件 JSONL.gz](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_peak_578_requests.jsonl.gz)：请求事实和本请求真实候选／写入／拒绝／驱逐事件。
- [全轨迹 GET JSONL.gz](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_all_get_requests.jsonl.gz)：全部 GET 明细，供历史来源回查。
- [资格检查 JSONL.gz](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_prefetch_qualification_checks.jsonl.gz)：单独的 check_only=True 调用与决定。
- [逐窗对账 CSV](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_window_reconciliation.csv)、[审计 JSON](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_audit.json)。

## 本轮具体认识

- 窗口 578 最大的单一需求成本类别是纯 REJECTED-BEFORE：77 个请求，占需求读取成本 40.1398%；完整评价段为 37.5835%。这些请求确有此前实际拒绝记录，值得先回查具体决策时点。
- 峰值窗口纯 ADMITTED-BEFORE 占 18.7307%，说明存在曾缓存而当前缺失的数据；其先前驱逐可能远早于峰值，保留至峰值的容量和时间成本尚未计入。
- 峰值窗口全部 17 个 MIXED 请求都是 REJECTED-BEFORE+OTHER；其 8.3131% 需求成本保留在 MIXED，没有计入任何单一历史类别的可恢复成本。
- 峰值窗口 747 个 OTHER 缺失 chunk（包含 MIXED 中的 OTHER）此前无实际候选、无待处理状态，也无资格检查记录。完整评价段另有 1,333 个 OTHER chunk 仅有此前资格检查记录，未当成实际拒绝。

以下仅为描述性样例：按整请求需求成本排序，分别保留纯 REJECTED-BEFORE 前三个和纯 ADMITTED-BEFORE 前两个；并列按请求序号排序，不把样例作为独立实验或收益见证。

| 峰值请求 | 块键 | 历史类别 | 缺失 chunk | 该请求需求 DT | 最近历史事件 | 距当前请求（小时） |
| --- | --- | --- | ---: | ---: | --- | ---: |
| 82777 | "14338612" | REJECTED-BEFORE | 64 | 0.317512% | 实际拒绝 | 4.825–4.825 |
| 82795 | "525850" | REJECTED-BEFORE | 64 | 0.317512% | 实际拒绝 | 1.928–1.928 |
| 82821 | "325469" | REJECTED-BEFORE | 64 | 0.317512% | 实际拒绝 | 1.415–1.415 |
| 82782 | "1849666" | ADMITTED-BEFORE | 64 | 0.317512% | 实际驱逐 | 46.017–46.066 |
| 82829 | "131795" | ADMITTED-BEFORE | 64 | 0.317512% | 实际驱逐 | 61.079–61.082 |

[样例全精度 CSV](C:/Users/windows/Documents/Codex/2026-09-29/mp01-pp-pbs-socs-2025-https/outputs/BC04_descriptive_examples.csv) 保留历史事件请求序号、时间和此前事件次数。

例如请求 82821 的 64 个实际缺失 chunk 此前均未写入，最近实际拒绝请求为 81555，距峰值请求约 1.4146 小时。该事实给出了具体可回查的准入决策时点；是否应该改变该次决定，仍需在容量、写入和预取计费内另做反事实回放。现阶段没有训练新模型的依据。

## 判读边界

本轮没有学习启动占比门槛。历史机会须进一步落实到具体时点、缓存容量与写入预算内的反事实回放，才能判断收益；分类占比不等于可恢复的峰值空间。仅有拒绝或曾缓存记录，不能认证当时决策错误。
