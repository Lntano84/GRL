# GRL 实验与研究裁决总档案

截至 **2026-10-06**。这里汇总已定位到的本地交付物，包括有效结果、否定结果、协议失败、更正和作废批次。各方向是独立探索，**不能合并成一种 GRL 算法的实验成绩**。

**当前结论：没有已建立的 GRL / GNN / RL 决策优势。** LOT 已形成可靠实验底座与有竞争力的非学习基线；LG01 接受固定 LOSS-EXPAND、停止当前 GNN 配置，不进入 RL。低预算算法选择保留开发线索，但强对照后的新算法贡献尚未建立。归档不启动实验，也不恢复暂停的自动跟进。

下面的数字来自对应阶段最终报告和结果文件；这次整合只核验文件、哈希、覆盖和链接，**没有重新运行科学实验或重新证明报告中的统计结论**。部分更正是在同一历史数据上离线完成的，不算新的独立确认。

## 如何阅读

1. 先读本页的最终裁决与 [更正说明](CORRECTIONS.md)，再查各阶段报告。
2. 报告和小型结果可直接在 GitHub 阅读；完整逐次结果、日志、方案、候选快照和代码版本在 [Release](https://github.com/Lntano84/GRL/releases/tag/experiments-2026-10-06)。
3. [文件清单](manifests/files.csv) 记录原始路径别名、大小和 SHA-256；[排除清单](manifests/excluded.csv) 明确哪些模型检查点、第三方原始输入或其他格式未上传。
4. 保存了 `BUGGY`、`INVALID`、`PRE_REVIEW` 和历史更正文件；**“存在文件”不代表其结论仍然有效**。最终报告和本索引优先于作废版本。

## 方向总览

| 方向 | 最终状态 | 关键结果与边界 | 报告 |
|---|---|---|---|
| 早期静态 IM / 过度暴露 | 历史证据与更正 | 成熟 RR 对手消除了早期学习优势主张；非单调窗口和状态机修正后的证据以 9 月审计为准，不使用修正前传播量、代理校准和负边际统计。 | [打开](EARLY_IM/merged/docs/RESEARCH_STATE.md) |
| CCIM / 调查式 IM | 多项有限配置收口 | 强非学习搜索胜过直接 DQN；合法最后一步排序未超过 random / degree_max；属性前瞻相对 observed_degree −0.058 激活节点，95% 区间 [−0.266,+0.150]。 | [打开](CCIM/CCIM_ATTRIBUTE_VOI_MINPROBE.md) |
| TAIM | 开发探针：未支持前瞻压缩 | SOF64 与便宜的立即投入策略 64/64 同选；两张小图、剩余 1/2 轮，不推广到整个 TAIM。 | [打开](TAIM/RESULT.md) |
| N01 / N01-R / N02 | 当前定向挖掘停止 | N01 的恒等式和分位点解释已撤回修正；N02 确认反转 0/8，未建立局部代理限制造成实用选种损失。 | [打开](N/N02_report.md) |
| M01 | 机制筛查停止 | 三轮小合成图上，前瞻几乎补齐贪心缺口；不否定实际肾交换或更大图。 | [打开](M/M01_report.md) |
| RT01–RT02R | 七天近期分组 KM 收口 | 初版 oracle 上界和数据异常解释撤回；最终 0/3 窗口超过 HISTORY，差距主要在回退样本，不是在线反馈问题整体无价值。 | [打开](RT/RT02R_report.md) |
| Q01–Q02R | 协议修正后不确定 | 权限问题限定在 cap<b 的预测驱动淘汰；Q02R 零重试，均值 +3.73% 但种子混合，只评估提前截止组合，不能评价重试价值。 | [打开](Q/Q02R_report.md) |
| LOT Stage 01–20 | 底座与强基线有效；新贡献未建立 | AB 的有效性曾独立复现，但 RINS 更强；Stage20 最终仅 1 个 +1.5165% 的正损失证据，0 个达到 2% 门槛，16/16 区间仍未解决。 | [打开](LOT/stage06_neighborhood_content/stage20_report.md) |
| MP01 | 不确定、偏弱 | PP/PBS 的逐状态特权选择相对地图与阶段固定选择仅约 0.07%–0.25% 起始 SOC；没有学习优势。 | [打开](MP01/deliverables/MP01_summary.md) |
| SB01 | 固定样本岭回归方案收口 | paired192：MAE 108.90、误判84/560；岭回归：106.31、85/560。预测误差略低，决策优势未建立。 | [打开](SB01/deliverables/SB01_summary.md) |
| BC01–BC08 | 预算内见证存在；当前学习配置收口 | BC05 特权 HIGH-CF 峰值下降5.6094%，不等于在线识别；BC07廉价规则和BC08新特征未建立部署优势。 | [打开](BC/deliverables/BC08_report.md) |
| GP00–GP01 | 工程资格通过；训练门槛未通过 | GP01 144次正式运行有效，但目标条件评分未建立超过简单校准的收益；未启动训练。 | [打开](GP01/outputs/GP01_report.md) |
| FA00–FA06 | 低预算开发信号保留；贡献未建立 | FA02 主动对随机对较强；FA03B 随机整实例将差距缩到2.44%，不确定；FA06权重处理不确定，未进入付费延长或GRL。 | [打开](FA/outputs/fa06/FA06_report.md) |
| LG01 | ACCEPT_CHEAP_STOP_GNN | GNN 对固定 LOSS-EXPAND 家庭均值 −0.09%，95%区间[−0.51%,+0.31%]，3/8正；验证候选池oracle额外收益0.63%。停止当前GNN，不进入RL。 | [打开](LG01/outputs/lg01/LG01_report.md) |

## 逐阶段索引

- [LOT Stage 01–20](LOT/STAGES.md)
- [FA00–FA06](FA/STAGES.md)
- [全部已归档报告](REPORT_INDEX.md)
- [发布资产及校验值](ASSETS.md)
- [整合来源与范围](integration_sources.json)
- [第三方来源版本](previous_package/source_provenance.json)与[许可](THIRD_PARTY_LICENSES/)

## 下载与恢复完整结果

每个方向有一个 `grl-20261006-<方向>.tar.gz`，含 `manifest.csv` 和以 SHA-256 命名的对象。相同文件内容只存一次；清单保留全部原路径，因此不会因为去重丢失作废批次或对应关系。下载方向所需的资产后，在仓库根目录运行：

```bash
python scripts/archive/verify_archive.py --assets /path/to/downloaded-assets
python scripts/archive/restore_artifacts.py /path/to/grl-20261006-lot.tar.gz --output research_archive/decision-studies-2026
```

第一条验证全部下载资产；只下载一个方向时可用第二条，它会验证该资产所有原始对象后恢复文件。恢复工具拒绝覆盖内容不同的已有文件，不运行实验、不安装依赖、不反序列化检查点。

历史脚本保留原字节，不保证无需改路径即可跨机器执行。部分代码依赖原本的目录布局、Windows文件名处理或作者外部代码；应先读阶段配置、依赖和来源，不能把这次归档称为全套实验的一键复现。模型检查点、虚拟环境、构建产物、未获明确再分发依据的外部输入和无关论文没有加入资产；训练标签、预测、测试方案和统计结果保留。早期模型权重等排除项可在排除清单中定位。

这里的 `in_git=0` 表示文件在 Release 资产内，**不是未归档**。资产所有对象均有大小与内容哈希；详细整合核验见 [INTEGRATION_AUDIT.json](INTEGRATION_AUDIT.json)。
