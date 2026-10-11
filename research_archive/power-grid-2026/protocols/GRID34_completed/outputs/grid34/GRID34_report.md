# GRID34 所有强对照共享优化后的实际控制比较

48条回放、95424物理步、299888次公开预测，完整保存数据核验。新拟合/RL/预留日期评价为0。

| 方法 | 控制秒总计 | N1查询 | N1搜索秒 | 完成回放 | 共同完成周平均成本差 |
|---|---:|---:|---:|---:|---:|
| FULL | 622.660 | 45194 | 132.859 | 6/8 | 0.00000% |
| LOCAL_GNN0 | 552.056 | 14624 | 59.291 | 6/8 | 0.04537% |
| LOCAL_GNN1 | 549.273 | 14968 | 60.680 | 6/8 | 0.00000% |
| LOCAL_PPO | 552.702 | 15554 | 62.625 | 6/8 | 0.04537% |
| LOCAL_MLP0 | 559.656 | 15882 | 65.337 | 6/8 | 0.20082% |
| LOCAL_MLP1 | 553.839 | 15652 | 62.805 | 6/8 | 0.00000% |

| GNN | 相同优化的对照 | 控制时间减少 | 查询减少 | 搜索时间减少 | 控制正向配对 |
|---|---|---:|---:|---:|---:|
| LOCAL_GNN0 | FULL | 11.339% | 67.642% | 55.373% | 8/8 |
| LOCAL_GNN0 | LOCAL_PPO | 0.117% | 5.979% | 5.322% | 3/8 |
| LOCAL_GNN0 | LOCAL_MLP0 | 1.358% | 7.921% | 9.253% | 5/8 |
| LOCAL_GNN0 | LOCAL_MLP1 | 0.322% | 6.568% | 5.594% | 4/8 |
| LOCAL_GNN1 | FULL | 11.786% | 66.881% | 54.327% | 8/8 |
| LOCAL_GNN1 | LOCAL_PPO | 0.620% | 3.768% | 3.105% | 6/8 |
| LOCAL_GNN1 | LOCAL_MLP0 | 1.855% | 5.755% | 7.127% | 7/8 |
| LOCAL_GNN1 | LOCAL_MLP1 | 0.824% | 4.370% | 3.383% | 6/8 |

4exposed development dates(two overlapoldtrain),2timingrepeats each; repetitions/modelseeds not independentfamilies.0.5%qualityinvestmentthreshold not equivalence. Shared compilation gains not GNN novelty; framework supervised scorer uses author PPO prior,not new RL training.

finished.wall_s covers remaining-phase invocation only, not previous12pilot workers. remaining_phase_timing.json records wrapper scope explicitly; controller_s per-step accounting covers all48runs.

质量：GNN1在三个共同完成周逐步与FULL一致；GNN0/PPO在三月成本高0.13612%，MLP0在九月高0.60247%且两重复一致，超过0.5%投资门槛。MLP0较差结果照常保留，不作筛除；所有方法六月共同1901步失败。

相同路径的GNN0/PPO八对：查询少5.979%，总控制仅少0.117%，其中3/8时间正向。不能把整体约11%速度收益全部归于GNN，也不能把小时间差当作统计确认。

六月共同失败不作零成本成功；不同轨迹的源搜索差含状态分布变化，另列同状态/同输入匹配比较。两个模型种子和每周两个计时重复全部展示，不选择最好种子。所有原始缓存/模型/停止配置不变。
