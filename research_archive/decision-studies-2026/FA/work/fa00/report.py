import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/"outputs"/"fa00"

def main():
    read=lambda name:json.loads((OUT/name).read_text(encoding="utf-8"))
    p=read("FA00_protocol.json");v=read("FA00_verdict.json");a=read("FA00_audit.json")
    data=read("FA00_data_audit.json");batch=read("FA00_batch.json");schedule=read("FA00_execution_schedule.json")
    extra=read("FA00_invariance_extra.json")
    names={"CLOSE_CURRENT_HYPOTHESIS":"收口当前付费反馈瓶颈假设",
           "CONTINUE_SCREEN":"值得进入下一次独立筛查","UNDETERMINED":"不确定"}
    lines=["# FA00：训练与验证运行的共同预算", "",
           f"**裁决：{names[v['verdict']]}。**", "",
           "本轮仅覆盖 ASP-POTASSCO、一个固定测试划分、三个算法随机种子，以及冻结的六条截止控制规则。种子不是三个独立数据集；未作显著性或等价检验。", "",
           "## 主结果", "",
           "主指标是最后一个完成训练的选择器在 129 个测试实例上的平均 PAR-10，越低越好。未按测试成绩挑选中间轮次。", "",
           "| 方法 | seed 7 | seed 42 | seed 99 | 均值（PAR-10 秒） | 验证成本占比 |", "|---|---:|---:|---:|---:|---:|"]
    for z in v["summary"]:
        lines.append(f"| {z['arm']} | {z['seed7_par10_s']:.3f} | {z['seed42_par10_s']:.3f} | {z['seed99_par10_s']:.3f} | {z['mean_par10_s']:.3f} | {100*z['pooled_validation_fraction']:.2f}% |")
    lines += ["", "FREE-DYN 使用免费完整验证反馈，是明确标注的特权参照，并非最优上界。PAID-DYN 与 SMALL32-DYN 只查询新旧选择不同的实例，并缓存完整结果。", "",
        "## 冻结判据", "", "| 固定对照 | FREE 相对优势 | FREE 更好的配对种子数 | 5% + 2/3 门槛 |", "|---|---:|---:|---|"]
    for z in v["fixed_controls"]:
        lines.append(f"| {z['arm']} | {100*z['free_relative_advantage']:.3f}% | {z['free_positive_seeds']}/3 | {'通过' if z['pass_gate'] else '未通过'} |")
    lines += ["",f"PAID-DYN 的验证成本占训练与验证总成本 **{100*v['paid_validation_fraction']:.3f}%**，10% 成本门槛{'通过' if v['paid_validation_fraction']>=.1 else '未通过'}。", "",
              "| 合法对照 | 相对 FREE 的均值差（正数更差） | 不超过 2% 或更好 |", "|---|---:|---|"]
    for z in v["closure_controls"]:
        lines.append(f"| {z['arm']} | {100*z['relative_to_free']:.3f}% | {'是' if z['within_2pct_or_better'] else '否'} |")
    lines += ["",f"继续判据：{v['continuation_gate']}；收口判据：{v['closure_gate']}。二者同时成立时，冻结协议优先收口，因为廉价方法已经取得所需成绩。阈值是投资筛查尺度。", "",
              "## 成本与终止", "", "| 方法 | 平均训练成本（模拟 CPU 秒） | 平均验证成本 | 平均剩余预算 | 平均本机墙钟秒 |", "|---|---:|---:|---:|---:|"]
    for z in v["summary"]:
        lines.append(f"| {z['arm']} | {z['mean_train_cost_s']:.1f} | {z['mean_validation_cost_s']:.1f} | {z['mean_remaining_budget_s']:.1f} | {z['mean_local_wall_s']:.1f} |")
    lines += ["", f"每条轨迹的求解器数据获取预算是 {p['budget_s']:.0f} 模拟 CPU 秒。运行从头重启，超时不被当成完成时间；PAR-10 超时惩罚为 6,000 秒，实际一次执行最多收费 600 秒。较早的超时不能从后来重跑的费用中扣除。", "",
        f"特征共同前置成本的已知部分为 {data['feature_known_cost_lower_bound_s']:.2f} 秒，有 {data['feature_missing_cost_cells']} 项缺失，因此无法声称完整前置成本或端到端总加速。模型拟合、预测、采样和包含落盘的本机墙钟分别保存；预收集矩阵执行时间与本机拟合时间不代表同一硬件。", "",
        "## 实施范围", "",
        "复用原始 ActiveRFModel 和 PassiveRFRegressor 类，100 棵树、sqrt 特征、原规模深度与硬投票保持。modAL 的 only_new=True 重拟合由直接 sklearn fit 适配，并做了同种子概率逐项一致检查。成本回归器使用 No_Weight；单算法名称不能用于计算成对权重。采样仍为 Uncertainty + PredCost 的 Pareto 分层。", "",
        "这是受控改编：保留原始运行状态，使用本场景 600 秒上限；成本预测遵循重启执行语义；输入仅来自已付款观测；保留所有实例，对缺失特征进行训练池拟合的中位数填充和标准化；固定反事实验证、预算及缓存接口。它不是 FrugalAS 最强配置的忠实复现，也不用于否定该论文。", "",
        "动态特征包含 presolved 标记对应的预求解步骤。本轮使用给定特征向量，不额外利用其成功状态构造一个新求解策略。回放固定在 repetition 1 的确定性运行表，不模拟运行时间随机变化。", "",
        "## 审计", "",
        f"原始 ARFF 的 CSV 数据段由独立解析器重读，核对 {a['raw_matrix_entries_checked']:,} 个运行单元；审计程序不导入策略或环境实现。", "",
        f"18/18 正式轨迹通过，共检查 {a['events']:,} 个事件、{a['observation_events']:,} 个观测事件，以及 {a['test_predictions_checked']:,} 个最终测试选择。最大逐事件账目误差为 {a['max_ledger_error_s']:.3g} 秒。预算违例、缓存收费错误、验证权限违例、标签方向错误、最终模型选择错误均为 0。", "",
        f"正式执行前 10 项预检通过；另有完整两世界补充核验（{extra['events_compared']} 条实际反馈）通过，包括付费成本、剩余预算及停止条件一致。补充核验未修改冻结的策略，也未读取测试成绩。", "",
        "冻结的策略、协议、数据及预检文件哈希保持不变；测试结果只在全部 18 条轨迹完成后由离线评价器读取。", "",
        "## 执行说明", "",
        f"首次串行轨迹在未读测试成绩时中断并归档为工程试运行，最后完整模型在第 {schedule['invalid_serial_trial']['last_fitted_round']} 轮，已用 {schedule['invalid_serial_trial']['last_checkpoint_wall_s']:.1f} 秒。随后改为六条独立轨迹并行，单个随机森林仍为 n_jobs=1。", "",
        f"归档串行轨迹与正式并行轨迹的 {len(a['serial_parallel_prefix_checks'])} 个完整轮次预测逐项相同。正式矩阵并行墙钟约 {batch['parallel_wall_s']:.1f} 秒；工程试运行另计，且从两小时上限中扣除。计数为 18 条有效正式轨迹，加 1 条中断的工程试运行，以及单列预检调用。", "",
        "## 研究含义", ""]
    if v["verdict"]=="CLOSE_CURRENT_HYPOTHESIS":
        lines += ["当前配置至少存在一个付费或无需验证的合法规则，其均值达到免费反馈参照的 2% 筛查尺度。因此没有依据围绕这项瓶颈开发复杂联合控制器或 GRL。它不证明反馈无价值，也不证明所有训练—验证预算分配问题都已解决。"]
        fixed100=next(z for z in v["closure_controls"] if z["arm"]=="FIXED-100")
        lines += ["",f"本轮具体触发收口的是 FIXED-100：其均值相对 FREE-DYN 低 {-100*fixed100['relative_to_free']:.2f}%，在种子 7、42 上更好，但在种子 99 上更差。不能把均值收口写成稳定支配或等价证明。PAID-DYN 和 SMALL32-DYN 自身均未达到与 FREE 的 2% 尺度。", "",
                  "费用门槛通过，只说明完整验证有不可忽略的开销；免费动态反馈没有胜过所有固定对照，所以本轮未建立“有价值但昂贵的反馈必须联合分配”这一关键前提。FREE 优于另外两条固定对照，仍说明截止规则选择会影响结果。", "",
                  "FIXED-100 的平均本机墙钟约 890 秒，高于 FREE-DYN 的约 479 秒。因此它在主指标与求解器获取预算上占优，不意味着计算开销更低或端到端更快；这些本机时间也受到并行执行的资源竞争影响。"]
    elif v["verdict"]=="CONTINUE_SCREEN":
        lines += ["当前配置同时显示免费反馈的质量价值与获取它的显著成本，且廉价对照未在筛查尺度内补足差距。值得设计第二场景的独立确认；尚未证明学习控制器能捕捉该差距，更未建立新的算法贡献。"]
    else:
        lines += ["冻结判据尚未提供足够明确的继续或收口信号。不自动换模型、增预算或选择有利种子。当前结果仍不足以支持启动 GRL。"]
    lines += ["", "## 复算与来源", "",
        "在任务根目录使用 `work/fa00/.venv/Scripts/python.exe work/fa00/audit_analyse.py` 可重做独立审计与统计，随后运行 `work/fa00/report.py` 可重建本报告；不需要重新拟合任何模型。", "",
        "- [固定作者源码与数据](https://github.com/stacs-cp/JAIR2026-FrugalAS/tree/7a5727651a92fd2fa4960dcbbd7f6dab94130028)",
        "- [ASP-POTASSCO 数据说明](https://github.com/stacs-cp/JAIR2026-FrugalAS/blob/7a5727651a92fd2fa4960dcbbd7f6dab94130028/DATASETS/ASP-POTASSCO/description.txt)",
        "- [Frugal Algorithm Selection for Combinatorial Search，JAIR 2026](https://doi.org/10.1613/jair.1.21266)",
        "- [On the Effect of Training Data Selection in Automated Algorithm Selection，CP 2026](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2026.38)",
        "", "数据哈希见 source_manifest.json，协议与冻结哈希见 FA00_protocol.json 和 FA00_freeze_manifest.json。"]
    (OUT/"FA00_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(OUT/"FA00_report.md")

if __name__=="__main__":main()
