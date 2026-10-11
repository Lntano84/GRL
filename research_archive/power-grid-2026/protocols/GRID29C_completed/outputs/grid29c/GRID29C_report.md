# GRID29C 完整两臂组件计时

每条2017步动作、观测、成本及11325次预测与旧受保护FULL相同。组件调用：153DC/153安全QP/34原作者连续QP。所有计时在真实派发两臂中取得，无隔离探针插入。

| 重复 | 原控制 | 缓存控制 | 整体减少 | DC节省 | 安全QP节省 | 原作者QP节省 | 预测节省 | 其余节省 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 181.911 | 170.174 | 6.45% | 17.475 | -1.130 | -3.432 | -0.876 | -0.301 |
| 1 | 182.557 | 170.681 | 6.51% | 18.143 | -2.739 | -2.498 | -0.779 | -0.251 |

汇总控制减少6.48%是该两组点估计，不替换旧10.93%/0.39%。按组件相加严格重建总差，正数为节省，负数为变慢。

| 重复 | DC局部减少 | DC节省/原完整控制 | 非DC额外节省 | solver状态/迭代改变数 |
|---|---:|---:|---:|---:|
| 0 | 56.71% | 9.61个百分点 | -5.738s | 0 |
| 1 | 57.32% | 9.94个百分点 | -6.267s | 0 |

Single exposed March,2balanced pairs with identical instrumentation. Component accounting identifies measured savings, not thermal/OS-specific causes or independent generalization.

原作者允许迭代上限等状态的路径仍保留；恢复QP资格规则未放松。复用不等于warm_start，后者仍False。具体compile/backend/iterations记录见review.json和各run/components.json，null不能填0。
