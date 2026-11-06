"""Archive GRID307 byte for byte and record how much of the approved scope ran."""

import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

R = Path("D:/Codex/GRL研究")
O = R / "outputs/grid307_unstarted_calibration_cont"
I = R / "work/GRL-results-integration"
A = I / "research_archive/power-grid-2026/protocols/GRID307_unstarted_calibration_cont"
B = R / "archives/GRID307_unstarted_calibration_cont_20261106/raw_records"
C = R / "论文协作资料/FRAME_校准CONT回放_GRID307"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not B.exists() and not (A / "results").exists()
    count = subprocess.check_output([
        "powershell", "-NoProfile", "-Command",
        "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' "
        "-and $_.CommandLine -match 'GRID307_unstarted_calibration_cont[/\\\\]runner\\.py'}).Count",
    ]).decode().strip()
    assert count == "0"
    assert not subprocess.check_output(["git", "-C", str(I), "status", "--porcelain"]).strip()
    manifest_path = O / "manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {"records": [], "new_accepted": 0}
    records = manifest["records"]
    indices = [r["index"] for r in records]
    assert indices == list(range(1, 1+len(records)))
    finished = O / "finished.json"
    error = O / "collection_error.json"
    assert finished.exists() != error.exists()
    assert not finished.exists() or len(records) == 16
    exit_receipt = read(finished if finished.exists() else error)
    assert exit_receipt["new_accepted"] == len(records)
    claims_dir = O / "claims"
    claimed = {int(p.stem.split("_")[1]) for p in claims_dir.glob("replay_*.json")}
    unaccepted = sorted(claimed - set(indices))
    unstarted = sorted(set(range(1,17)) - claimed)
    partial = []
    for item in read(A / "execution_order.json"):
        if item["index"] in unaccepted:
            folder = O / "runs" / f"r{item['repeat']}__{item['week']}__{item['method']}"
            rows_file = folder / "partial_steps.jsonl"
            partial.append({
                "index": item["index"],
                "saved_step_rows": sum(1 for line in rows_file.open(encoding="utf8") if line.strip()) if rows_file.exists() else 0,
                "summary_present": (folder / "summary.json").exists(),
            })
    approval = read(A / "approval.json")
    for name, expected in approval["source_sha256"].items():
        assert sha(R / name) == expected
    assert all(r["event"]["ordinary261_continue"] is False and r["event"]["forced_continue"] is True
               and r["event"]["delivered"] for r in records)
    for item in partial:
        ref = read(A / "execution_order.json")[item["index"]]
        directory = O / "runs" / f"r{ref['repeat']}__{ref['week']}__{ref['method']}"
        partial_receipt = directory / "partial_intervention.json"
        item["forecast_count_saved"] = (read(partial_receipt)["forecasts"]
                                        if partial_receipt.exists() else None)

    start = time.perf_counter()
    B.mkdir(parents=True)
    entries = []
    objects = {}
    files = sorted(p for p in O.rglob("*") if p.is_file())
    with zipfile.ZipFile(B / "raw_objects.zip", "x", allowZip64=True) as archive:
        for path in files:
            assert time.perf_counter() - start < 600
            assert path.resolve().is_relative_to(O.resolve())
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            name = "objects/" + digest[:2] + "/" + digest
            entries.append({"path": path.relative_to(R).as_posix(), "sha256": digest,
                            "bytes": len(data), "object": name})
            if digest not in objects:
                archive.writestr(name, data,
                                 compress_type=zipfile.ZIP_STORED if path.suffix in {".npz", ".gz"} else zipfile.ZIP_DEFLATED,
                                 compresslevel=1)
                objects[digest] = name
    with zipfile.ZipFile(B / "raw_objects.zip") as archive:
        for digest, name in objects.items():
            assert time.perf_counter() - start < 600
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    raw = {
        "passed": True, "local_archive": str(B), "logical_files": len(entries),
        "logical_bytes": sum(e["bytes"] for e in entries), "unique_objects": len(objects),
        "zip_sha256": sha(B / "raw_objects.zip"), "independent_object_readback": True,
        "entries": entries,
    }
    save(B / "raw_records_manifest.json", raw)
    save(B / "archive_complete.json", {"passed": True, "manifest_sha256": sha(B / "raw_records_manifest.json")})
    save(O / "raw_records_manifest.json", raw)

    full = sum(r["summary"]["complete"] for r in records)
    assert all(r["summary"]["steps"] == 2017 for r in records if r["summary"]["complete"])
    status = "完成" if finished.exists() else "停止"
    result = {
        "status": "COMPLETED" if finished.exists() else "STOPPED",
        "accepted_new": len(records), "planned_new": 16, "full2017": full,
        "early_terminated": len(records) - full, "partial_or_unaccepted": partial,
        "unstarted_indices": unstarted,
        "physical_steps_in_accepted_runs": manifest.get("physical_steps_new", 0),
        "forecasts_in_accepted_runs": manifest.get("forecasts_new", 0),
        "physical_steps_new": manifest.get("physical_steps_new", 0) + sum(x["saved_step_rows"] for x in partial),
        "forecasts_new": manifest.get("forecasts_new", 0) + sum(x["forecast_count_saved"] or 0 for x in partial),
        "wall_s": exit_receipt["wall_s"],
        "new_fits": 0, "cost_targets_derived": 0, "original_grid297_301_baselines_preserved": True,
        "raw_manifest_sha256": sha(O / "raw_records_manifest.json"),
    }
    save(O / "closeout.json", result)
    msg = (
        f"GRID307 续采{status}，接纳 {len(records)}/16 条，其中 {full} 条完整走完 2017 步，"
        f"{len(records)-full} 条提前结束。另有 {len(unaccepted)} 条启动后未接纳，"
        f"{len(unstarted)} 条未启动。接纳条目物理步 {result['physical_steps_in_accepted_runs']:,}，"
        f"forecast {result['forecasts_in_accepted_runs']:,}，总计时 {result['wall_s']/60:.1f} 分钟。"
    )
    if error.exists():
        msg += " 执行失败收据和已保存部分记录单列。"
    report = (
        "# GRID307 校准续采收尾\n\n" + msg
        + "\n\n仅执行 GRID306 从未启动的 1–16 号。首条旧部分轨迹已独立保留。"
        "复用原 261 STOP 基线，共同前缀通过后仅多查一批原候选。"
        "15 条旧标记按原整数键复原候选别名，另 1 条完整标记保持原值。"
        "采集阶段新增费用标签与训练均为 0。"
        + f"\n\n原始记录无损归档于 {B.as_posix()}，"
        f"共 {raw['logical_files']:,} 个文件和 {raw['logical_bytes']:,} 字节，逐对象 SHA-256 读回通过。\n"
    )
    (O / "GRID307_report.md").write_text(report, encoding="utf8")
    S = A / "results"
    S.mkdir()
    C.mkdir(parents=True)
    names = ["GRID307_report.md", "closeout.json", "raw_records_manifest.json",
             finished.name if finished.exists() else error.name, "pipeline_state.json"]
    if manifest_path.exists():
        names.append("manifest.json")
    for name in names:
        shutil.copy2(O / name, S / name)
        shutil.copy2(O / name, C / name)
    shutil.copy2(Path(__file__), S / Path(__file__).name)
    save(S / "archive_manifest.json", {p.name: sha(p) for p in S.iterdir() if p.is_file()})
    save(C / "delivery_hashes.json", {name: sha(C / name) for name in names})
    for name in ("RESEARCH_STATE.md", "DECISIONS.md", "EXPERIMENT_LOG.md"):
        path = I / "docs" / name
        old = path.read_text(encoding="utf8")
        if name == "RESEARCH_STATE.md":
            new = "# 当前 GRID307 续采收尾\n\n" + msg + "\n\n" + old
        else:
            new = old + "\n\n## 2026-11-06 — GRID307 续采收尾\n\n" + msg + "\n"
        path.write_text(new, encoding="utf8")
    (I / "docs/NEXT_STEPS.md").write_text(
        "# 当前 GRID307 续采已收尾\n\n" + msg
        + "\n\n按固定清单读取已接纳保存配对，双方完整、安全且同一 2017 步才出费用标签。"
          "保留其余失败、部分和未启动状态。按日期家族报告覆盖。\n", encoding="utf8")
    index = I / "research_archive/power-grid-2026/README.md"
    old = index.read_text(encoding="utf8")
    index.write_text(
        "# 最新 GRID307 续采收尾\n\n" + msg +
        "\n\n[收尾报告](protocols/GRID307_unstarted_calibration_cont/results/GRID307_report.md)\n\n" + old,
        encoding="utf8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
