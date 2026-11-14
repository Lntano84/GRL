# e-Energy 冬季轮主表与英文实验段落底稿

方法和主表按 12 月 15 日内控日期排期。现有对照使用原 261，加 0.9 保护并采用固定 DC 混排。原启发式和带同样混排、保护的 208 出自 GRID330，独立 LogEIPC 对照出自 GRID334。CSV 给出 12 个同来源配对及各臂原始量。三月原启发式的历史计时保留供查阅，配对用时差留空。从同一批配对结果取值可避免跨轮将比例串起来。

| 证据 | 直接可写进稿件的结果 | 表述边界 |
| --- | --- | --- |
| GRID330，受保护混排 261 对原启发式 | 四个开发周共少查 1,615 个候选（20.95%）。三个共同完周两次费用下降、一次持平。六月和十二月同日控制耗时分别多 3.06% 和 0.90%。 | 共享混排的闭环表现。三月的历史 FRAME 基线仅用于轨迹、查询和费用。 |
| GRID330，受保护混排 261 对同配置 208 | 四周的轨迹、费用、候选数及预测数一致，261 的额外取消为零。 | 这批闭环未建立 261 修正的独立收益。 |
| GRID334，独立 LogEIPC 对受保护混排 261 | 四组配对少 133 个候选（2.18%），少 72 次全部预测。三共同完周的控制累计耗时少 1.16%。三月费用高 0.0861%，六月和十二月同轨迹费用一致。 | 单次配对的计时。三月首分岔命中基础 NN 继续分支，发生在 261 修正和局部保护之前。 |
| GRID337，最后一批停查资格 | 训练零个必须保留批次，校准两类分别只覆盖一个日期。按预先固定条件关闭。 | 保存状态诊断，只作为方法选择记录和补充材料。 |

上表中九月均为同轨迹第 451 步终止，保留预测、步数和异常。整周费用缺失。各阶段使用已暴露开发周，单次配对结果适合描述性报告。最终保留日期的证据、跨平台验证及可重复提速均需各自结果支撑。

## Methods and outcomes

We compare the guarded 261 configuration with the original search heuristic and two controls on four previously used development weeks. In GRID330, the matched 208 arm shares the global candidate ordering and the 0.9 local protection rule with 261. In GRID334, the independent LogEIPC arm uses the same global search and station-specific pool, then chooses candidates and stops using its own cost-aware acquisition rule. Each timing comparison uses runs from the same experiment and the same evaluation horizon. We report full-week cost where both runs complete. The September runs end at step 451 and contribute matched-horizon results.

In GRID330, the guarded mixed-order 261 configuration evaluated 20.95% fewer candidates than the original heuristic across the four matched horizons. Operating cost was lower on two of the three commonly completed weeks and equal on the third. Under the same ordering and protection rule, 208 and 261 followed identical trajectories and issued the same number of forecasts on all four weeks. In GRID334, LogEIPC evaluated 2.18% fewer candidates than guarded 261 and used 1.16% less cumulative controller time across the three commonly completed weeks. Operating cost was 0.0861% higher in March and unchanged on the other two completed weeks. At the first March divergence, the native network in the 261 configuration purchased 35 additional candidates and selected a different action. This decision preceded both the 261 residual correction and the local protection rule.

## 定稿检查

- 12 月 15 日锁定这套配对证据与表内缺失。正文以候选评估、闭环结果和消融为主线，时延保留逐周及 P95/P99。已发表的 LogEIPC 来源见 [ICML 2026 论文](https://proceedings.mlr.press/v306/xie26c.html)。
- 12 月下半月用本页段落和 CSV 整合正文、图表、负例及匿名复现材料。送合作者审阅前核对环境、软件版本、截止条件与引用。保留原稿件。
- 内部目标为 2027 年 1 月 8 日前摘要、1 月 15 日前全文，以存档冬季轮日期倒排。2026-11-14 在线核查时官方 CFP 返回 403，投稿系统未返回可读截止信息，正式提交前需核对系统。
