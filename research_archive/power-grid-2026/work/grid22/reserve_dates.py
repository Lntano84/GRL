"""Reserve new date families using names only, before confirmation scores exist."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid_final_holdout"; OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "reserved_date_families.json").exists()
audit = json.loads((ROOT / "outputs/grid_research_audit/split_exposure.json").read_text())
original_path = ROOT / "outputs/grid03/scenario_split_frozen.json"
original = json.loads(original_path.read_text())
seed = "GRID_DATE_FAMILY_RESERVATION_20261010_V1"
pick = []
for q in range(1, 5):
    dates = audit["unseen_by_quarter"][str(q)]
    assert len(dates) >= 2
    chosen_dates = sorted(dates, key=lambda s: hashlib.sha256(f"{seed}:date:{s}".encode()).hexdigest())[:2]
    for date in chosen_dates:
        suffixes = original["replica_groups"]["test"]
        chosen = min(suffixes, key=lambda s: hashlib.sha256(f"{seed}:variant:{date}:{s}".encode()).hexdigest())
        pick.append({"quarter": q, "date": date, "primary_episode": f"{date}_{chosen}", "reserved_all_variants": [f"{date}_{s}" for s in range(16)]})
assert all(p["date"] not in audit["exposed_dates"] for p in pick)
assert len({p["date"] for p in pick}) == 8
record = {"metadata_only": True, "seed": seed, "source_index_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
          "date_families": pick, "candidate_unseen_pools": audit["unseen_by_quarter"],
          "status": "RESERVED_NOT_EXTRACTED_OR_EVALUATED", "new_forecasts": 0, "new_physical_steps": 0,
          "rules": "Never use any variant of these8 dates for later training, development, feature/hyperparameter/model/checkpoint selection. Keep all selected failures, no outcome-based replacement. Confirmation methods/budgets must be frozen separately before extraction/evaluation. External PPO training exposure unknown, no unseen-to-all-competitors claim."}
path = OUT / "reserved_date_families.json"
path.write_text(json.dumps(record, indent=2), encoding="utf-8")
(OUT / "reservation.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+"\n", encoding="utf-8")
(OUT / "README.md").write_text("# 新日期家族预留\n\n只使用官方场景名与本项目已用日期记录选择8个日期家族（每季度2个），每个家族全部16个序列保留，首选评价序列也以固定哈希选定。尚未解压新周、运行代理或读取其成绩。\n\n后续训练与开发必须排除全部这些家族；真正确认的模型、对照与预算另行冻结，不能在这里寻找模型赢家。外部PPO原训练身份未知。详见reserved_date_families.json。\n", encoding="utf-8")
print(json.dumps({"reserved_dates": [p["date"] for p in pick], "episodes": [p["primary_episode"] for p in pick], "new_forecasts": 0}))
