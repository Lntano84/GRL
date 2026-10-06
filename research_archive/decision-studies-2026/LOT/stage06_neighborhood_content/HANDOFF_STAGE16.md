# HANDOFF — Stage 16（AB 对单次 LP 引导邻域 RINS-LP-ONE：基线筛查）

目录：`08_LOT_REOPT/stage06_neighborhood_content/`

## 问题与回答

> **AB 的固定规则，在同样的 20 秒总预算下，能否优于根据 LP 松弛与已有可行解的一致性构造的邻域？**

**回答：不能。命中预先固定的第 2 条 —— RINS-LP-ONE 反向达到全部支持条件 ⇒ 通用 LP 信息规则更有竞争力；停止把固定 AB 当作核心贡献推进。**

| 判据（预先写死，对称） | AB | RINS-LP-ONE（反向） |
|---|---|---|
| 平均优势 ≥ 2% | **−7.49%** ✗ | **+7.49%** ✓ |
| ≥6/8 名义实例为正 | **0/8** ✗ | **8/8** ✓ |
| 两个 ρ 组均正 | 均负（−7.05% / −7.93%）✗ | 两组均正 ✓ |
| 逐一删例后均值仍正 | 全负（−0.0822 ~ −0.0672）✗ | 全正（+0.0672 ~ +0.0822）✓ |

主指标 `D = (J_RINS-LP-ONE − J_AB)/max(1,|J_r|)`，**正 = AB 更好**；先在实例内对 4 个故障格子等权，再对 8 个实例等权。
**更细粒度：全部 32 个 (实例, 格子) 对中 RINS-LP-ONE 32/32 都更好**（无一处 AB 反超，不属于"被少数极端实例拉动"）。
四格 mean D：`single/1p` −0.0885、`single/2p` −0.0783、`all/1p` −0.0671、`all/2p` −0.0657。

---

## ⚠️ C37：本轮先修了一个会改变结论的实现缺陷

**首次批量报 `select errors 32`，RINS-LP-ONE 的 `final_source` 全部是 `lp`。**

- **缺陷**：我在**所有方法**上都把 **AB 的短期 Y 固定集**传给候选验解。RINS-LP-ONE 找到的合法解因此被判"违反固定条件"而**丢弃**，delivered plan 退回成本高得多的 LP 候选。
- **证据**：按**它们自己真正的固定集**复检，**违反数 0/32**；其中一个解 `check_solution` 完全可行（max_violation 6.8e−13）、κ=4 恰好满足、目标 1877.26。
- **影响**：修正后 **32/32 状态的交付成本都变好**，平均 **+1542.10** 成本单位。缺陷版把 RINS 从**最强**误报成**最弱**。
- **修正**：固定集现由**本次运行自己的 `fixed`** 推导（`y_fixings_of`），不用任何常量。修正后 **0 select errors、0 fallbacks**。
- **对照**：修正前后 **AB 与 FULL 逐值完全相同（32/32，max|diff| = 0.0000）**，证明修正只动了 RINS 的验解。
- 缺陷版留档 `stage16_runs_BUGGY_C37.json`，**不作为依据**。

---

## 关键数字

- **`(J_FULL − J_AB)/D` = +0.16885（8/8 为正）**；**`(J_FULL − J_RINS)/D` = +0.24374（8/8 为正）**。两种显式邻域都优于直接求全解，且 RINS 优于 AB。
- **邻域结构**：RINS 固定 **1818–2069**（均值 1949）、释放 **451–702**（均值 570）；AB 固定 1224、释放 1296；FULL 固定 0、释放 2520。
  presolve 后列数均值 **RINS 1591.1 < AB 2105.2 < FULL 4404.0**。
  ⇒ **RINS 不是靠更大邻域取胜，而是靠 LP 一致性把固定位置选得更准**：它固定得比 AB 多、释放得更少、模型更小。
  RINS 不按组偏向（A/B/C/short_Y 分别约 597/382/572/399 被固定），是**逐变量**由 LP 信息决定的。
- **fallback 0/32**：32/32 都拿到有效最优松弛（`relax ≤ J_LP` 32/32、LP 阶段均值 0.209 s、最大 0.516 s，从未撞 5 秒上限）。
- 完整性：start 采用 96/96、选择错误 0、`final_source=none` 0、差于 `J_r` 0/96、证最优 0/96、总耗时 ≤20.25 s 96/96。

## ⚠️ 本批最意外的诊断：FULL 零改进

| 方法 | MIP 严格优于 `S_LP` |
|---|---|
| AB | **32/32** |
| RINS-LP-ONE | **32/32** |
| **FULL** | **0/32**（全部恰好停在 `S_LP`） |

日志显示 FULL 把 20 秒全花在根节点割平面分离（`Nodes 0`、`LP iterations ≈63701`、`(heuristics) 0`、`gap 56.45%`）。
**`S_LP` 不是原问题下界**（它把全部二元钉在 `S_r` 后求 LP，故其最优值 ≥ 整数最优值——这正是 Stage 09A 能优于它的原因），所以这不表示"FULL 不如 LP"，只表示**本批 32 状态 / 20 秒 / 单种子下直接求全解零产出**。
**⇒ 它放大了 AB 与 RINS 相对 FULL 的全部读数**（+16.9% / +24.4%），本轮**无法分离**"邻域好"与"完整任务在这批状态上尤其难"。**不要把 +24% 读成 RINS 的真实增益幅度。**

## 边界（勿过度解读）

- **不是新的独立确认**：这是**已有开发状态上的基线筛查**；32 状态**不是独立样本**，名义实例是基本单位（8 实例 × **仅种子 0**）。
- **胜过单次 RINS ≠ 胜过完整 RINS ≠ 胜过全部强基线。** `RINS-LP-ONE` 只有**一次** LP 松弛与**一次**邻域求解，无搜索树内反复调用，**不是原论文 RINS 的复现**。
- **不能声称 FULL 没有这类搜索**：HiGHS 官方源码已含 RINS 类启发式；本轮比的是**显式构造邻域并分配预算的策略**，所有方法保留原求解器设置。
- **种子只有 0** ⇒ 无法区分"方法差"与"种子运气"，**稳定性未检验**。
- 未测"达到相同质量所需的时间"，**不能**做收敛速度结论。2% 是**工程参考线**，不是显著性水平。

## 下一步建议

1. **按预设收束 AB 作为核心贡献的定位**，但**别把 `RINS-LP-ONE` 当新核心**——它只是"一个通用规则能追上"的证据，**尚未与领域内成熟的 fix-and-optimize 对手比较**（Sahling 等按变量/资源/时间的邻域；完整 RINS / local branching / proximity search）。
2. **最值得买的下一条信息**：把 `RINS-LP-ONE` 升级为多轮 / 与求解器内启发式并用的版本，或换成更强的成熟基线，检验"通用规则普遍支配固定规则"是否成立。若成立，贡献主张需从"更好的固定规则"转向**别的轴**（如故障后重优化的**建模与协议**本身）。
3. **最便宜且最能保护结论的一步：补种子**（1–2 个），检验 §3 的方向对种子是否稳健。
4. **单独查一次 FULL 零改进**（例如放宽到 60 秒看是否只是预算太短）——它同时影响所有相对 FULL 的读数。

## 交付物

`stage16_report.md` / `stage16_analysis.json` / `stage16_runs.json` / `stage16_results.csv` / `stage16_per_instance.csv` / `stage16_run.py` / `stage16_analyse.py` / `stage16_preflight.py` / `stage16_logs/`；`lsp_release.py` 新增 `solve_lp_relax`、`stage16_rins_columns`、`stability_row_local`、`binary_slot_map`、`structural_pin_columns`、`group_of_column`；缺陷版留档 `stage16_runs_BUGGY_C37.json`。

## 复现命令

```powershell
cd "C:\Users\windows\Desktop\im算法与基准阅读包\08_LOT_REOPT\stage06_neighborhood_content"
.\.venv-hs\Scripts\python.exe stage16_preflight.py    # 三项预检，≈30 s
.\.venv-hs\Scripts\python.exe stage16_run.py --resume  # 96 次 MIP，wall ≈ 1942 s
.\.venv-hs\Scripts\python.exe stage16_analyse.py       # 主指标 / ρ 组 / 删例 / 冻结判据
```

## 环境

- `.venv-hs\` 是指向 `stage05_mipstart\.venv-hs` 的目录联接：highspy 1.15.1 / numpy 2.5.3 / scipy 1.18.1。
- 系统 Python 3.13.14 **没有** highspy；清华 pip 镜像不含 highspy，需 `--index-url https://pypi.org/simple`。
- 运行器支持逐运行 checkpoint + `--resume`。
- **重跑会覆盖** `stage16_runs.json` / `stage16_results.csv` / `stage16_logs/`；先备份。
