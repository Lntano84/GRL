# GRID57 最终配置的固定顺序消融

裁决：MIXED。STATIC_ID质量尺度通过8/8；GNN种子平均相对STATIC_ID查询减少1.897%，正向4/8。

| 日期 | STATIC完成 | 成本变化/FULL | GNN成本增益/FULL | STATIC查询 | GNN平均查询 | GNN查询减少 |
|---|---:|---:|---:|---:|---:|---:|
|2035-01-22_2|True|-9.0327%|-2.1960%|5102|7011|-37.417%|
|2035-02-26_13|True|-0.5290%|-0.6095%|2748|2546|7.369%|
|2035-05-28_0|True|+0.0000%|+0.0000%|0|0|不定义（无查询）|
|2035-06-04_6|True|-0.1765%|-0.0021%|1714|1600|6.622%|
|2035-07-30_12|True|-0.1734%|+0.1088%|1431|1024|28.442%|
|2035-09-24_4|True|-0.0973%|-0.0887%|1024|1024|0.000%|
|2035-10-15_4|True|-0.8954%|-0.4691%|3036|3236|-6.604%|
|2035-11-12_9|True|-1.0559%|-0.8124%|1759|1498|14.866%|

One fixed order,one timing repeat,eight already-exposed dates. Does not establish graph architecture necessity or new independent confirmation. Static batch wall time is descriptive only;no chosen seed/data/dropout.

不读取历史结果控制新轨迹；FULL和GNN记录仅在离线审计用于参照。固定顺序不使用先验，也不支付无用先验开销；DC与安全QP缓存仍共有。所有早失败和较差结果均保留。

查询统计只在有拓扑调用的7个日期定义；2035-05-28_0双方0查询，保留其质量/成本结果，不强造0/0比例。报告中正向分子/总体8仍沿原门槛；详见comparable_query_families和工程修正记录。
