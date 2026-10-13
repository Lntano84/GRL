# 复核与复现入口

本包不启动回放或训练，以下命令只读已有结果、验证收据和重建论文表格：

```powershell
& work/grid02/.venv/Scripts/python.exe work/grid56/build.py --source-root C:/Users/windows/Documents/Codex/2026-09-24/influence-maximization-im-ccf-b-grl --out outputs/grid56_reproduction
```

输出目录必须不存在。需要本地大轨迹、模型权重及旧档案；Git小型结果归档不是这些资源的替代。

实际控制复现的组件入口为work/grid55/run.py的worker分支、work/grid55/worker.py，停止逻辑work/grid28/policy.py，共有缓存work/grid29/cache_dc.py、work/grid30/cache_safe.py、work/grid54/cache_prior.py；原正式运行器不允许对已完成目录重新运行。下一次有限消融必须建立新协议/输出目录，明确引用同一冻结组件，不能修改旧design.json解除保护。

最终模型：outputs/grid50/models/gnn_seed0/final.pt与gnn_seed1/final.pt。原版：outputs/grid21/models/gnn_seed0/final.pt与gnn_seed1/final.pt。模型hash见outputs/grid55b/code_freeze.json，归档中保留原新模型来源和失败记录。

八日期、动作库、作者依赖、LightSim2Grid后台和单线程要求见outputs/grid55/design.json及运行metadata.json。没有额外下载、安装、训练或隐藏成绩读取步骤。
