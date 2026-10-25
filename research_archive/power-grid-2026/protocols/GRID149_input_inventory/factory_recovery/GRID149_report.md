# GRID149：历史输入、目标和剩余对照核查

结论：不能把过去“学习失败”视为同一问题上的重复证据。不同阶段有轨迹等价、局部净收益和全局动作保持三类目标；结构配对已经在旧净收益目标上试过，当前全局目标仍缺一个同输入的正规低容量对照。

| 阶段 | 目标 | 输入 | 基础动作如何使用 |
|---|---|---|---|
| GRID101 | Whole same-station immediate reward-gain vsquery fee | 37 paid-prefix numeric summaries | No explicit1605 command vector |
| GRID102/103/107/120/123 | Local next-batch net gain or sequential query utility | 37 summaries +4 locked-station/batch/budget channels=41 | No explicit continuous command vector;effects already in paid candidate feedback |
| GRID110/111 | Diagnostic structure/paid reward association for local net gain | Public combined command partitions +observed feedback +current local pressure | Used for combined action reconstruction,not a complete continuous feature vector |
| GRID113/116/119 | Local next-batch net gain | 41 summary +68 per-action partition +8 feedback fields;paired vsfactorized controls | Topology effects included during combined-command construction;continuous base commands not explicitly encoded |
| GRID133 | Local no-improvement/action-preservation gate | Original first37 numeric fields | Not explicit |
| GRID140/144 | Earlier global STOP +original completion preserves originalFRAME action and reduces queries | 18 summaries +18missing;class weights changed in144 | Not explicit |
| GRID142 | Same global action-preserving compression | Original18 +6batch reward/rho summaries +24missing | Not explicit |
| GRID146 | Same global target but only one learning exit at128 | Original18 +18missing,153fit rows | Not explicit |
| GRID148 | Current global label,descriptive-only geometry | 12 physical/graph/visible-feedback summaries compared with18/combined30 | Not encoded |

旧关系模型并非完全没用基础动作：重建站内连接意图时已合并公共候选与基础动作。准确的缺口是连续调度/限发/储能等基础命令没有作为完整数值上下文输入，不能把这说成以前模型完全没有动作结构。已支付候选反馈也已经包含共同基础动作的运行后果，缺少显式命令不等于完全没有相应信息。

305个既有基础动作向量有16种精确字节取值。按序列化字段解码的活动命令计数另存；它们只是控制命令，未求解执行后功率或潮流，未据这些分布证明它们能预测标签。

## 下一项可控实验

当前18项全局动作保持任务只试过森林及输入/权重/单入口变体，尚没有同输入、同日期、同目标的正则化逻辑回归。GRID101等旧逻辑回归预测的是另一个标签，不能当作这个对照已经做过。

因此下一步优先补一个低容量基线，直接检验森林拟合偏乐观的问题；不再给现有模型追加12项物理摘要，不重做旧关系网络。参数继承GRID101的C=1、lbfgs、最多2000次、种子20261022、无类别平衡，仅训练一次。标准化必须只用拟合行，原18项/缺失标记、原386分层权重、日期和动作标签保持。

这改变分类器及其必要预处理，不称同参数量消融。仍按原校准规则和完整搜索/固定256强对照评价，模型与阈值在开发验证评分前封存，失败后不补训。当前阶段没有执行拟合；下一阶段训练前通知。

所有历史坏结果和各自目标继续保留，基础动作是否值得新增输入仍是未验证问题，不用它为下一训练编造依据。最终8测试未生成。

## 基础命令原生复核

用原生序列化动作空间核对默认空动作及305次向量往返：229次为空动作，76次只有线路重连（状态命令+1）；调度、储能、限发及拓扑命令均与空动作一致。因此，这批搜索中没有被遗漏的连续基础控制变化可供学习，不能把它当作下一特征扩展的依据。线路重连位置本身仍可能作为公开上下文，但尚未建立预测价值。没有构造环境或发送仿真。

工程记录：首次原生复核把序列化动作空间当作可调用工厂，遇接口错误后停止；原源码/错误已保留。仅改用安装源码使用的space.subtype()构造空动作，305次校验通过，没有拟合、求解或源数据改变。原先提交消息提前写了原生审计完成，本收据与后续提交纠正这一发布时序；以native_command_audit.json为通过依据。
