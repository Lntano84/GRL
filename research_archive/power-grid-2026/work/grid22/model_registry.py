"""Index preserved power-grid model versions, without mutating sealed artifacts."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid_research_audit"; OUT.mkdir(parents=True, exist_ok=True)
records = []
for stage, version, change in [("grid20", "V0", "Supervised candidate decoder on existing two-message graph encoder; current observations and known combined plans; no PPO prior in train logits."),
                              ("grid21", "V1", "Same graph/legal-input encoder, bounded residual above frozen original-PPO plus LOCAL prior. Zero correction at initialization, bound2, KL coefficient0.1. Matched-prior MLP and bias-only controls.")]:
    d = json.loads((ROOT / f"outputs/{stage}/design.json").read_text())
    comparison = json.loads((ROOT / f"outputs/{stage}/comparison.json").read_text())
    fits = []
    for checkpoint in sorted((ROOT / f"outputs/{stage}/models").glob("*/final.pt")):
        folder = checkpoint.parent
        summary = json.loads((folder / "summary.json").read_text())
        fits.append({"name": folder.name, "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "initial_checkpoint": (folder / "initial.pt").relative_to(ROOT).as_posix(), "curve": (folder / "curve.json").relative_to(ROOT).as_posix(), "fit": summary})
    records.append({"version": version, "stage": stage, "status": "SEALED", "change": change, "fits": fits,
                    "train_weeks": d["training_weeks"], "development_weeks": d["development_weeks"], "comparison": comparison,
                    "new_RL_training": False, "report": f"outputs/{stage}/{stage.upper()}_report.md", "manifest": f"outputs/{stage}/delivery_manifest.json"})
record = {"versions": records, "closed_loop": {"stage": "grid22", "status": "RUNNING" if not (ROOT / "outputs/grid22/finished.json").exists() else "FINISHED_PENDING_CLOSEOUT",
          "model_selection": "No further fitting; both GNN V1 seeds, MLP/bias seed0 fixed before rollouts, frozen originalPPO32/128 and FULL controls."},
          "historical_RL": "GRID08/10 binary restoration-policy experiments and weights preserved; that role did not establish an advantage over ALWAYS_RESTORE. Current ranking role is supervised, uses an existing pretrained PPO prior and is not new PPO/DQN training.",
          "split_boundary": "Current results are development only. Original suffix-reserved test was partially reused at GRID19. See SPLIT_EXPOSURE_CORRECTION.md.8 new date families separately reserved; no future result read.",
          "cpu_note": "Both versions use one CPU thread; individual neural fits about40s, no new GPU package justified by measured fitting cost."}
(OUT / "power_model_registry.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
text = ["# 电网模型保留与改动记录", "", "所有原始、失败与修订版本保留，不覆盖旧权重。机器清单为power_model_registry.json。", "",
        "| 版本 | 改动 | 已建立的证据 | 边界 |", "|---|---|---|---|",
        "| 原PPO与工程控制器 | 作者发布权重保留；连续控制、恢复规则保留 | 原PPO是强排名参照 | 发布模型原训练身份未知 |",
        "| GRID08/10恢复决策RL | 原图Actor-Critic、奖励尺度／critic修订均保留 | 恢复规则本身有收益，学习决策没有超过固定恢复 | 此角色的弱结果不外推到拓扑排序 |",
        "| V0／GRID20 | 现有两层图编码＋候选描述头，同信息MLP | K128有竞争力；K32未保住原PPO优势 | 监督开发检查，不是新RL |",
        "| V1／GRID21 | 原PPO先验＋有界残差；头零初始化，偏置／MLP同先验 | GNN32近最佳保留97.73%，PPO32为90.58%，偏置90.58%，MLP94.16%；两个GNN种子同向 | 改善集中少数开发状态，尚须闭环、日期家族独立确认 |",
        "| GRID22 | 冻结V1、相同端点处理／合法性／去重／回退，28条整周比较 | 运行中，不能提前宣告运营收益 | 新训练0，RL0，全部失败保留 |",
        "", "V1训练目标包含KL正则，与V0纯交叉熵曲线不可直接横比。模型名字或文件哈希变化不作为训练成功证据，已有审计逐Tensor核对参数实际变化并重新预测。",
        "", "GNN＋RL、电网候选排序、仿真兜底和先验融合都有已有文献；此记录不宣称组合首次提出。现在需要的是强对照下的实际质量—时间证据，再辨别贡献。",
        "", "当前模型不使用候选未来结果作输入。训练标签来自当时合法公开仿真，但原始教师采集成本单列保留。后续确认须排除预留8个日期家族的全部变体。"]
(OUT / "POWER_MODEL_CHANGELOG.md").write_text("\n".join(text)+"\n", encoding="utf-8")
print(json.dumps({"versions": len(records), "preserved_fitted_checkpoints": sum(len(r["fits"]) for r in records)}))
