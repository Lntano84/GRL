# META统一配置资料入口

先读 `META_统一配置说明.md`。本资料把现有META的训练配方、准确权重、执行接口及评价批次逐项对应。

- `configuration.json`：统一25维、三动作学习方法，显式绑定C0/C0025各五种子。
- `results_by_configuration.json`：138条历史运行和逐配置汇总，批次分别保留。
- `configuration_issues.json`：历史标签继承、费用索引和重复次数的歧义及处理。
- `behavior.json`：逐配置FRAME委托行为。
- `validation.json`：10权重、138运行、612搜索调用通过，20个绑定与原冻结匹配。
- `source_receipts.json`：原始证据SHA256。
- `source/`：配置构建、统一调用入口及归档脚本。

本次新训练0、新闭环0、新仿真0。原校准质量匹配False保留。主结果2.23%仍仅对应GRID84费用0、五种子和两次反向计时；GRID89费用复查是一次计时，不能合并。历史文稿未覆盖。

可执行工作区为 `D:/Codex/GRL研究`。统一入口实际位于 `work/meta_unified_20261019/runtime.py`，调用者提供公共observe接口；这里只存源码副本。依赖原冻结策略和已有本地权重，完整路径/哈希见配置。未在此重复打包大型轨迹、环境和模型。
