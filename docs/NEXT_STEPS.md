# Next steps — GRID118固定基础模型校正方案已核对，未训练

1. 将四组方案交用户检查：原BASE复用；BASE_EXTRA与固定BASE+JOINT/FACTOR各三个种子，拟9次新增训练。
2. 获准训练后编写、核验并封存实际执行器；保持原目标/Huber/费用/数据/种子，使用已核对的后续抽样序列，首步与梯度检查计入7250次。
3. JOINT/FACTOR配对判据不变；同时比较原BASE和同额外训练BASE_EXTRA。不自动改目标、扩网络、上电网闭环或读取最终测试。

方案：research_archive/power-grid-2026/protocols/GRID118_fixed_base_plan/complete/固定主干校正方案_中文.md
