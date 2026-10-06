# Capacity Constrained Influence Maximization in Social Networks

- 发表：ACM 2023（会议论文）
- DOI：https://doi.org/10.1145/3580305.3599267
- 官方页面：https://doi.org/10.1145/3580305.3599267
- OpenAlex 记录显示该论文题名为 `Capacity Constrained Influence Maximization in Social Networks`。

## 术语

CCIM = Capacity-Constrained Influence Maximization，容量约束影响力最大化。

与标准 IM 的差别是：传播过程或节点可接收/激活的影响量受到容量限制。因此，普通 IC 模型中“一个节点被多个路径触达后只算一次”的 RR 覆盖等价关系可能需要重新证明，不能默认 IMM 的 RR 估计器仍然无偏。

## 下载状态

ACM PDF 在当前网络环境下需要授权或连接不稳定，本阅读包保留了 DOI 和官方入口，没有放入来源不明的盗版文件。拿到 PDF 后可直接放到本目录，命名为 `capacity_constrained_influence_maximization.pdf`。

## 与当前 GRL 的关系

如果 CCIM 是你们课题中的目标问题，应先明确容量约束的数学定义，再决定认证 oracle 是 CCIM 专用算法、Monte Carlo，还是可证明正确的 RR 变种。不能直接把标准 IMM 当作 CCIM 的无条件 baseline。
