# GRID05 固定回退检查

这是一项已看开发数据后的规则验证，不是独立确认。阈值直接取作者rho_safe=0.9，不扫参。没有合格Top-20动作或其选中拓扑的已执行预测rho>=0.9时才扩到同库352项；复用前20项的原始奖励，完整排序破并列。安全阈值只是作者控制器阈值，不是安全认证。

四个原开发周、环境种子0；8条交错轨迹。新NN20对照先校验与GRID04逐步相同；旧NN352只作质量参照，不能比较跨批生产延迟。恢复、重连、N-1搜索、连续优化、原生规则、成本适配器及评分范围不变。不增加候选池，不训练，不进入验证或测试。

先做零实际推进预检：模拟反馈/阈值/缓存/并列/基动作语义，以及在首个旧开发初始状态比较强制扩展与原生全扫描。预检最多800次一步模拟，180秒；强制开关仅用于该检查。实际运行不启用强制扩展。

主要检验：各周不比新NN20更早终止；1月结束步至少达到旧NN352的1719；4月完整成本至少恢复旧NN352相对NN20差距的80%；其他共同完整周成本回退不得超过2%。全部满足仅支持这条开发规则；任一更早终止或完整周>2%成本退化则不推广；其他记混合/不确定。不存在任何自动启动GRL的分支。

成本只比较共同完整周，不把提前终止算节省。一步拓扑预测与最终连续控制后的实际rho分别记录，不作同一动作残差解释。主运行包络含预检共1800秒、16200次实际推进，逐轨迹900秒，输出2GiB。错误/超限保留，不自动重跑或换周。审计另计时间，保存全部动作、反馈、扩展决策与原始模拟记录。

Preflight I/O correction: design content unchanged; hash sidecar corrected to persisted bytes after a CRLF/LF mismatch. The failed attempt stopped before environment construction (0 native simulations, 0 physical steps). Invalid hash and log retained. Failed wall time 9.73664 s charged to the original preflight/main envelope. No policy, threshold or outcome change.
