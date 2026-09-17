# 阶段 5 报告：把 GRL pipeline 接到 overexposure oracle

> 日期：2026-09-17
> 脚本：`scripts/experiments/stage5_grl_pipeline.py`、`stage5_predictor_diagnostics.py`
> 数据：`docs/results/stage5_predictor_diagnostics_20260917.json`
> 结论：**pipeline 打通了,audit / verification / fallback 都在工作;
> 但"学习型 predictor 能帮上忙"这一点目前没有得到证据支持。**

---

## 1. Pipeline 状态：六项检查点

| # | 检查点 | 状态 | 证据 |
| --- | --- | --- | --- |
| V1 | predictor 是否 condition on 当前状态 S | ✅ **是** | `StateConditionedMarginalPredictor` 输入含 `seed_mask` 与 `exposure_features`(实现的 δ 场);有 `use_seed_mask=False` 的对照变体 |
| V2 | 预测的是 marginal gain 而非静态节点分 | ⚠️ **未证实** | 见 §3:首次检查在 `\|S\|=0` 处测,那里没有状态;重测仍未显示优势 |
| V3 | audit 是否真检查 prediction 可靠性 | ✅ **是** | `adaptive_selective` 用 `residual_max + β·residual_std` 构造未验证候选的经验上界,certified 2/2 步 |
| V4 | progressive verification 是否真省 oracle | ❌ **当前没有** | `adaptive_selective` 花 **490** 次级联,比 full oracle 的 **410** 还多 |
| V5 | fallback 是否真存在 | ✅ **是**(代码路径),⚠️ **未被触发** | `fallback_steps = 0`,因为 audit 每次都在 `max_m` 之前认证成功 |
| V6 | corrupted predictor 是否触发加强验证/fallback | ⏳ **未测**（阶段 7） | corruption 接口已实现(`clean/noise/shuffle/random/sign_flip`),尚未跑 |

---

## 2. 首次端到端运行（Congress-Twitter, k=2, pool 20）

| policy | spread | gap to oracle | MC cascades | verified | fallback |
| --- | --- | --- | --- | --- | --- |
| full_oracle | 355.80 | — | 410 | 39 | 0 |
| learned_only | 340.24 | 4.37% | **20** | 0 | 0 |
| selective_greedy | 360.80 | −1.41% | 150 | 12 | 0 |
| adaptive_selective | 355.80 | **0.00%** | **490** | 39 | 0 |

**读法（保守）**：
- `learned_only` 用 **20** 次级联拿到 4.37% 的 gap——**成本轴的雏形是存在的**；
- `selective_greedy` 用 150 次级联达到与 oracle 无差别的 spread；
- 但 `adaptive_selective` **比 oracle 更贵**，与"省 oracle"的主张相反。

⚠️ **这些 spread 数字不能归因于学习**——见 §3。

---

## 3. ⚠️ 核心否定结果：learning 打不过 degree

`stage5_predictor_diagnostics.py` 在 **context 内**比较排序能力（这才是种子选择真正用的东西）。
3 种特征 × 2 种 exposure 开关 × 3 个 ranking loss 权重 = 18 个变体，15 个测试 context：

| 变体 | ρ_model | ρ_degree | ρ_random | top-1 |
| --- | --- | --- | --- | --- |
| random emb, seed-mask, rw0 | **+0.201** | +0.320 | +0.051 | 0.27 |
| random emb, +exposure, rw0 | +0.169 | +0.320 | +0.051 | 0.33 |
| onehot, +exposure, rw5 | +0.195 | +0.320 | +0.051 | **0.40** |
| structural, +exposure, rw0 | +0.070 | +0.320 | +0.051 | 0.13 |
| （最好 18 个中的最大值） | **+0.201** | **+0.320** | +0.051 | 0.40 |

**结论**：
- ✅ **learning 确实在学**：ρ_model ≈ 0.20 ≫ ρ_random = 0.051；
- ❌ **但始终输给普通 degree（0.320）**，18 个变体无一例外；
- top-1 命中率有例外（onehot+exposure+rw5 达 0.40 vs degree 0.20），但 ρ 仍低于 degree。

### 我尝试过的补救及其结果

| 补救 | 动机 | 结果 |
| --- | --- | --- |
| 加 listwise ranking loss (rw=5) | 纯回归 MSE 被跨 context 的量级差异主导，可能忽略排序 | **无改善**（0.195 vs 0.201），仅 top-1 略升 |
| 加 ranking loss (rw=20) | 同上，加强 | **更差**（−0.012 ~ +0.19） |
| onehot / structural 特征 | 怀疑随机 embedding 让节点不可区分 | **无一致改善** |

**没有找到让 learning 超过 degree 的配置。** ranking loss 不进 pipeline。

---

## 4. 为什么阶段 5 的 spread 数字不能作为证据

`adaptive_selective` 达到 gap 0.00% 看起来很好，但：

1. **predictor 的 context 内排序弱于 degree**（§3），
   所以它选择的种子不太可能真的更好；
2. **图在 k=2 附近接近饱和**（阶段 4 已测：congress k≥5 全饱和），
   spread 差异被压缩，任何合理选择都接近 oracle；
3. **`selective_greedy` 的 −1.41%**（比 oracle 还高）正是饱和噪声的表现，
   不是"超过了 oracle"。

⇒ **必须换到不饱和区间重做**，否则无法区分"学习有效"与"饱和掩盖了差异"。

---

## 5. 当前最大的科学风险（**升级**）

**到目前为止,没有任何证据表明学习型 predictor 优于一个免费的 degree 基线。**

这与阶段 4 的一个发现吻合：**在 headroom 最大的图上（NetHEPT），状态依赖性只有 0.075**，
即那里本质上是静态排序问题,degree 已经很接近最优。

论文的 cost 轴主张需要"学习 + 验证"在**质量可比**的前提下**减少 oracle 调用**。
目前：
- 质量可比 = 只在饱和区间成立（无区分度）；
- oracle 调用 = `adaptive_selective` 反而更多。

⇒ **这两点都还没被支持。** 这是当前最需要解决的科学问题。

---

## 6. 三个可能的出路（需 PI 判断）

| 方案 | 做法 | 我的评估 |
| --- | --- | --- |
| **A. 换评估区间** | 在**不饱和**的图上重做（`ca_grqc` k=3、`facebook` k=2，阶段 4b 显示 headroom 38~90%） | 最直接。若学习在那里仍输给 degree，则是硬否定 |
| **B. 改预测目标** | 不预测绝对边际增益，改预测**对比/排序**或**相对优势**，并让 degree 作为特征之一 | 承认"degree 是强基线"，把它纳入输入而不是对手 |
| **C. 改主张重心** | 放弃"预测更准"，改为"**用 audit 控制风险**"：learned 只做 proposal，质量由 verification+fallback 保证，卖点是**风险可控**而非更准 | 与已确认定位兼容，但需要"风险"有可测量的定义 |

我倾向 **A 先做**（成本低、可否证），A 通过后再考虑 B/C。

---

## 7. 交付物

| 文件 | 内容 |
| --- | --- |
| `src/grl/training/overexposure_dataset.py` | 签名标签的 overexposure 数据集（**不截断负值**） |
| `src/grl/models/overexposure_marginal.py` | state-aware predictor（含 `score_candidates` 批处理） |
| `src/grl/oracle/overexposure_mc.py` | 新增 `state()` 方法（平均曝光场） |
| `scripts/experiments/stage5_grl_pipeline.py` | 端到端 pipeline + 六项检查点 |
| `scripts/experiments/stage5_predictor_diagnostics.py` | **context 内排序诊断（本报告 §3 的来源）** |

**已知实现细节（诚实记录）**：
- 数据集用**随机 embedding** 作为节点特征；这是一期基线，结构化/学习型 embedding 是后续工作；
- `label_std` 在数据集中记为 `nan`，因为 oracle 的 `score()` 未逐候选记录 stderr；
- `adaptive_selective` 的 MC 成本包含**曝光观测**（每步一次 cascade），需要时应分开计费。
