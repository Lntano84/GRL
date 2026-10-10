"""Write the report only after the read-back audit succeeds."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03"

def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))

a=load("GRID03_audit.json")
assert a["passed"]
s=load("scenario_split_frozen.json")
d=load("download_parallel_state.json")
idx=load("archive_index.json")
sm=load("formal_smoke_corrected_ledger/summary.json")
init=load("official_initialization_manifest.json")
score=load("scoring_smoke/summary.json")
sch=load("schema_check.json")
cleanup=load("range_part_cleanup.json")
groups="\n".join(f"| {k} | {', '.join(map(str,v))} | {len(s['all_splits'][k])} | {len(s['extracted_subsets'][k])} |" for k,v in s["replica_groups"].items())
report=f"""# GRID03 正式数据与运营计量资格

## 裁决

**正式数据、分组划分、静态接口和原始运营成本计量通过；比赛原版归一化总分尚未认证。** 本轮没有训练、没有调整策略或 Top-k，没有对验证/测试场景运行代理。

这解决的是后续实验的工程资格，不是学习优势、泛化或论文新意。GRID02 两个随包场景上的结果不因此升级。

## 数据与资源

- 官方 `l2rpn_idf_2023` 原包：{a['archive_bytes']:,} 字节；SHA-256 `{a['archive_sha256']}`。
- 已完整读过 tar/bzip 索引：{idx['members']:,} 个成员、{a['archive_scenarios']} 个场景，52 个周日期 × 16 个模拟序列后缀。整个包内文件字节数 {idx['sum_member_file_bytes']:,}。
- 本轮归档响应体实读 {a['download_application_body_bytes']:,} 字节，低于 6 GiB 上限。首个串行前缀 {d['serial_prefix_bytes']:,} 字节保留；后续四条不重叠 Range 续传，没有第二次完整下载。下载耗时约 {(d['completed_epoch']-d['first_started_epoch'])/60:.2f} 分钟，低于 2 小时上限。
- 只解压 48 个冻结场景及静态配置，共 {a['extraction_payload_bytes']:,} 字节，低于 8 GiB 解压上限；{a['extracted_files_rehashed']} 个文件落盘后逐项重算哈希。
- Range 临时片段曾占用近一个原包的额外空间；逐段与保留原包核对后已清除 {cleanup['bytes_reclaimed']:,} 字节临时片段，原包保留以复用；这不是第二次网络下载。
- 本地摘要用于完整性对账；没有取得作者发布的独立 SHA-256 校验值，不把本地摘要称为外部真实性认证。

数据来自 [Grid2Op 官方数据索引](https://github.com/Grid2Op/grid2op-datasets/blob/master/datasets.json)。场景是法国 2035 能源结构设定下的合成电网时间序列，不是真实运营日志。[官方环境说明](https://beta-grid2op.readthedocs.io/en/bd_dev/available_envs.html#l2rpn-idf-2023)

## 冻结划分

| 分组 | 整组后缀 | 原包分组场景数 | 本轮解压数 |
|---|---|---:|---:|
{groups}

同一后缀全部归一组。后缀 0 因含此前用过的两个随包前缀而固定留在开发组；剩余后缀按固定 SHA-256 排序。每月选开发 2、验证 1、测试 1 个场景，按名字哈希排序，不按难度、负载、代理成绩挑选。完整 832 项划分和 48 项抽取清单在 `scenario_split_frozen.json`，其哈希在解压前已保存。

三类生成种子的重复计数：`{s['duplicate_generation_seed_counts']}`。分组是一项保守评价约定，不证明统计独立。公开 LJN 模型未提供训练场景身份，因此不能声称验证/测试数据对该发布模型未见。

48/48 场景通过所需文件、五分钟间隔、99 负载/62 发电列、字段唯一与行宽/行数检查；预测行数为实际行数的 12 倍。本轮未对 48 个场景做全时域 AC 可行性证明，也未把结构检查称为全部数值质量认证。

## 接口与计量

- 原始压缩包不含 `alerts_info.json`，第一次静态检查在执行任何正式物理步之前中止，动作/观测维度不兼容。官方下载器在解压后本来还会调用配置更新；本轮最初的手动解压省略了这一步，不能把原始归档直接当作已完成初始化的环境。
- 按官方数据仓库固定提交 `{init['commit']}` 的 `updates.json`，在独立副本中更新 `alerts_info.json`、`difficulty_levels.json`、`config.py`。更新补充告警元数据和时间窗、清理重复分区编号；原始解压文件全部保留，未调用会改写用户全局环境的 `update_env`。副本额外复制约 {init['copied_payload_bytes']/1024**2:.2f} MiB。来源、差异及哈希在 `official_initialization_manifest.json`。[固定版本的官方更新索引](https://github.com/Grid2Op/grid2op-datasets/blob/{init['commit']}/updates.json)
- 完成官方初始化后，静态索引、热限额、参数、分区规则、动作字段和告警线路与随包环境一致。这项一致性检查不是通过自选规则覆盖获得的。
- 开发场景 `{sm['scenario']}` 的原生最大时域为 **{sm['native_horizon']} 步**；只跑前 12 步接口检查，不称完整周评估。
- {sm['passive_pairs']} 对带/不带附加评价奖励的控制步，观测、搜索奖励、终止和原生标志逐项相同；实际应用 {a['scripted_probe_steps_applied']} 次小幅弃能/储能探针。
- 原始运营成本由官方 `L2RPNSandBoxScore` 输出；另从相邻交付观测独立计算损耗、再调度、弃能变化和储能吞吐量。逐步最大绝对残差 {a['passive_cost_residual_max']:.8g}，按保持不变的预设浮点容差通过。物理弃能总量另列；当前版本官方费用的弃能项是其相邻步变化量，解除弃能时该项可为负，不能把它当作每步弃能能量费用。
- 352/352 个 NN 动作向量在正式环境中可重建、无歧义且内容顺序保持。现有 NN20 做了 {sm['nn_steps']} 步冒烟；原生非法、歧义、异常为零。{a['reused_assets_unchanged']} 个复用源码/模型/动作文件哈希未变。
- 总计 **{a['total_physical_steps']} 个物理环境步**（隔离随包评分参考轨迹 {score['physical_steps']} 步、失败的初版弃能账本检查 {a['failed_stateless_ledger_physical_steps']} 步、修正后的正式检查 {sm['physical_steps']} 步），低于 100 步上限；没有启动完整正式参考统计或策略比较。

## 评分边界

搜索仍用作者 `MaxRhoReward`；运营成本、可再生利用和告警表现作为附加评价量，不能用最大 rho 冒充运营分。官方辅助类默认总分权重为运营 0.60、可再生 0.15、告警 0.25。[官方 API](https://grid2op.readthedocs.io/en/latest/user/utils.html)

隔离副本上实际运行官方辅助类：3 条各 10 步轨迹，DoNothing 的预期分项 `(0,100,100)`、总分 40 均通过。这是限长接口冒烟，人工截短结束不代表完整周的能源/告警评价。

**当前 Grid2Op 1.12.5 父类在归一化参考损耗时两次读取 `load_p`，使该参考损耗计算为零。** 这是可直接核验的源代码行为，不据此推断策略排名反转；库未修改。详见 `NORMALIZER_AUDIT.md`。因此本轮不能把辅助类结果称为已经复现原比赛评分。原始成本分项资格与这项归一化问题分开报告。

提前停电时，较低的累计成本不是优势。后续正式评价必须先保留生存/终止，再报告完整运营成本和终端能源/告警分项。

## 执行说明与下一步

手算检查首次因 Windows 默认 GBK 读取 UTF-8 源码失败；改为显式 UTF-8 后通过。报告生成器的一处 f-string 括号错误在编译预检中修正。正式物理轨迹没有因这两处错误重跑。串行下载主动停止后保留前缀，传输并发修订写入原协议，没有修改数据选择规则或预算。

**初版独立账本的多步弃能解释错误。** 单步手算未暴露这个问题：弃能从 0 增至 5 时，总量与变化量相同；持续为 5 或解除为 0 时两者不同。初始化后的成对轨迹在第 4 步中止，共计 8 次物理调用。源码核对后改为显式接收上一观测的弃能总量，补了持续、解除、增加三种手算。失败输出保留在 `formal_smoke_initialized`，初版手算结果保留为 `metric_unit_checks_initial.json`；修正后在同一冻结场景重跑，输出为 `formal_smoke_corrected_ledger`。未扩大容差、换场景或改变代理，失败调用全部计入上限。

并行下载中，Windows 检查点 JSON 的原子替换因读者短暂占用而报 `PermissionError`，3 个分段部分中断。逐文件长度与响应体账目完全一致；增加仅针对文件替换的有限重试后，只补齐 30,408,704 字节缺失内容，未重传完整包。原错误日志保留在 `download_parallel.log`，恢复日志为 `download_resume.log`。

下一步应先冻结正式开发周上的成熟基线质量—时间比较，以及精确的评分版本/口径；验证/测试暂不解封。应先确认现成 NN 筛选在更长时域下的成本与安全表现、再分类候选覆盖/排序/连续优化差异，并补同信息权限的廉价对手。当前不安排新训练、RL、候选库扫参或按测试成绩挑场景。

主要审计：`GRID03_audit.json`。边界：本轮审计验证文件、分组、接口和账目，不是独立的 AC 潮流物理实现，也不是研究假设的独立确认。
"""
(OUT/"GRID03_report.md").write_text(report,encoding="utf-8")
(OUT/"RUN_STATE.md").write_text("# GRID03 run state\n\nCompleted 2026-10-08. No pending simulation, training or automation.\n\n- Official archive and 48-scenario subset acquired and audited.\n- Grouped split frozen; no validation/test agent outcomes.\n- Raw archive preserved; pinned official post-download updates applied to separate copy.\n- Static compatibility and stateful passive raw-cost metering passed; failed raw/static and stateless-ledger attempts retained and counted.\n- Official competition-exact normalization remains uncertified; see NORMALIZER_AUDIT.md.\n- No new policy tuning or training. Next stage not started.\n",encoding="utf-8")
print(json.dumps({"report":str(OUT/"GRID03_report.md"),"audit":str(OUT/"GRID03_audit.json")}),flush=True)
