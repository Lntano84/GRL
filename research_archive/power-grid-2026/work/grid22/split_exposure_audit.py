"""Metadata-only historical partition audit; no trace arrays or scores opened."""
import hashlib
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid_research_audit"; OUT.mkdir(parents=True, exist_ok=True)
original = json.loads((ROOT / "outputs/grid03/scenario_split_frozen.json").read_text())
d19 = json.loads((ROOT / "outputs/grid19/design.json").read_text())
membership = {week: split for split, weeks in original["all_splits"].items() for week in weeks}
exposures = {week: membership[week] for week in d19["new_training_weeks"]+d19["development_weeks"]}
reserved_training = [w for w in d19["new_training_weeks"] if membership[w] == "test"]
reserved_development = [w for w in d19["development_weeks"] if membership[w] == "test"]
seen = set()
for stage in (ROOT / "outputs").glob("grid[0-9][0-9]"):
    for p in stage.rglob("*"):
        if p.is_dir():
            match = re.match(r"^(2035-\d{2}-\d{2}_\d+)(?:__.*)?$", p.name)
            if match: seen.add(match.group(1))
# Bundled early prefixes were previously used, before formal grouped selection.
seen.update(["2035-01-15_0", "2035-02-12_0"])
seen_dates = sorted({w.split("_")[0] for w in seen})
all_names = sorted(membership)
unseen_dates = sorted({w.split("_")[0] for w in all_names}-set(seen_dates))
by_q = {str(q): [d for d in unseen_dates if (int(d[5:7])-1)//3+1 == q] for q in range(1,5)}
result = {"metadata_only": True, "original_test_suffixes": original["replica_groups"]["test"],
          "grid19_reserved_test_used_for_training": reserved_training,
          "grid19_reserved_test_used_for_development": reserved_development,
          "grid19_original_membership": exposures, "observed_week_directory_names": sorted(seen),
          "exposed_dates": seen_dates, "metadata_unseen_dates": unseen_dates, "unseen_by_quarter": by_q,
          "scope": "Date directories from output paths and two known bundled prefixes. External PPO training identity unknown. Reading/extracting data qualification files alone is not scored exposure. No new heldout trace or score read."}
(OUT / "split_exposure.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
text = ["# 评价分组的历史使用更正", "", "2026-10-10，只核对分组与输出路径元数据，没有新增模型、仿真或成绩访问。", "",
        "GRID03原来按序列后缀划分开发、验证和测试；GRID19后来在48个已解压周中重新选训练／开发周，没有保留最初的后缀分组。以下原测试周已被训练使用：", ""]
text += ["- "+w for w in reserved_training]
text += ["", "以下原测试周已作为开发数据使用：", ""]+["- "+w for w in reserved_development]
text += ["", "因此不能再称最初GRID03测试组整体未解封。GRID20、21、22本来已将当前四周定位为开发检查，其数值不变，但这条历史边界必须额外披露。旧文件封存不改写；本记录为累积更正。",
         "", "后续独立确认需要新的、未用于本项目训练或开发的日期家族，冻结全部同日期序列的身份；不能把已读周重新命名为测试。外部PPO的训练身份仍未知，不能保证对发布模型未见。",
         "", f"输出目录与已用随包前缀覆盖{len(seen_dates)}个日期，完整官方索引另有{len(unseen_dates)}个未用于这些输出的日期。仅为元数据资格，尚未执行或查看其代理成绩。机器表保留完整来源和逐季度池。"]
(OUT / "SPLIT_EXPOSURE_CORRECTION.md").write_text("\n".join(text)+"\n", encoding="utf-8")
print(json.dumps({"original_test_now_training": reserved_training, "original_test_now_development": reserved_development, "seen_dates": len(seen_dates), "unseen_by_quarter": {k: len(v) for k,v in by_q.items()}}))
