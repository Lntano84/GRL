"""Archive GRID309 source artifacts once, after all live processes have exited."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
O = R/"outputs/grid309_action_opportunity_preparation"
E = R/"work/grid309_action_opportunity_preparation/train_env"
A = I/"research_archive/power-grid-2026/protocols/GRID309_action_opportunity_preparation"
B = R/"archives/GRID309_four_family_pilot_20261106/raw_records"
C = R/"论文协作资料/FRAME_四日期试采准备_GRID309/collection_closeout"
S = A/"collection_results"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def archive():
    assert not B.exists()
    start = time.perf_counter()
    B.mkdir(parents=True)
    objects, entries = {}, []
    files = sorted(p for root in (O,E) for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(B/"raw_objects.zip","x",allowZip64=True) as archive:
        for path in files:
            assert time.perf_counter()-start < 600
            assert path.resolve().is_relative_to(R.resolve())
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            name = "objects/"+digest[:2]+"/"+digest
            entries.append({"path":path.relative_to(R).as_posix(),"sha256":digest,"bytes":len(data),"object":name})
            if digest not in objects:
                archive.writestr(name,data,compress_type=zipfile.ZIP_STORED if path.suffix in {".npz",".gz",".bz2"}
                                 else zipfile.ZIP_DEFLATED,compresslevel=1)
                objects[digest] = name
    with zipfile.ZipFile(B/"raw_objects.zip") as archive:
        for digest,name in objects.items():
            assert time.perf_counter()-start < 600
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    raw = {"passed":True,"local_archive":B.as_posix(),"logical_files":len(entries),
           "logical_bytes":sum(row["bytes"] for row in entries),"unique_objects":len(objects),
           "zip_sha256":sha(B/"raw_objects.zip"),"independent_object_readback":True,"entries":entries}
    save(B/"raw_records_manifest.json",raw)
    save(B/"archive_complete.json",{"passed":True,"manifest_sha256":sha(B/"raw_records_manifest.json")})
    shutil.copy2(B/"raw_records_manifest.json",O/"raw_records_manifest.json")
    return raw


def close():
    assert not S.exists() and not C.exists()
    count = subprocess.check_output(["powershell","-NoProfile","-Command",
        "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' "
        "-and $_.CommandLine -match 'GRID309_action_opportunity_preparation[/\\\\]runtime\\.py'}).Count"]).decode().strip()
    assert count=="0"
    assert not read(O/"active_worker.json")["active"]
    finished,error = O/"finished.json",O/"collection_error.json"
    assert finished.exists() != error.exists()
    ready = read(finished if finished.exists() else error)
    gen = read(O/"generation_manifest.json")["records"] if (O/"generation_manifest.json").is_file() else []
    m = read(O/"manifest.json") if (O/"manifest.json").is_file() else {"records":[],"physical_steps":0,"forecasts":0}
    records = m["records"]
    assert [r["index"] for r in records]==list(range(len(records)))
    selection = read(O/"selected_markers.json") if (O/"selected_markers.json").is_file() else None
    base = [x for x in records if x["index"]<4]
    cont = [x for x in records if x["index"]>=4]
    if finished.exists():
        assert len(gen)==4 and len(base)==4 and len(cont)==ready["cont_runs"]==len(selection["selection"])
        assert ready["physical_steps"]==m["physical_steps"] and ready["forecasts"]==m["forecasts"]
    else:
        assert ready["completed_generation"]==len(gen) and ready["accepted"]==len(records)
    partial = []
    runtime = read(O/"runtime_design.json")
    for item in runtime["execution_order"][len(records):]:
        folder = O/"runs"/f"r{item['repeat']}__{item['week']}__{item['method']}"
        if not folder.exists():
            continue
        rows = folder/"partial_steps.jsonl"
        receipt = folder/"partial_intervention.json"
        partial.append({"index":item["index"],"date":item["date"],
                        "saved_step_rows":sum(1 for line in rows.open(encoding="utf8") if line.strip()) if rows.exists() else 0,
                        "forecasts":read(receipt)["forecasts"] if receipt.exists() else None})
    generation_started = sorted(int(p.stem.split("_")[1]) for p in (O/"claims").glob("generation_*.started"))
    raw = archive()
    result = {"status":"FINISHED" if finished.exists() else "STOPPED","generation_completed":len(gen),
              "generation_started":generation_started,"baseline_replays_accepted":len(base),
              "CONT_replays_accepted":len(cont),"full_baseline_replays":sum(r["summary"]["complete"] for r in base),
              "full_CONT_replays":sum(r["summary"]["complete"] for r in cont),
              "accepted_physical_steps":m["physical_steps"],"accepted_replay_forecasts":sum(r["forecasts"] for r in records),
              "accepted_generation_schema_forecasts":sum(x["schema_forecast_calls"] for x in gen),
              "partial_replays":partial,"selection":None if selection is None else
                  {k:selection[k] for k in ("coverage_passed","primary_families","reference_families")},
              "new_labels":0,"new_fits":0,"raw_manifest_sha256":sha(O/"raw_records_manifest.json")}
    if finished.exists():
        result["wall_s"] = ready["wall_s"]
    save(O/"collection_closeout.json",result)
    summary = (f"GRID309 {'完成' if finished.exists() else '停止'}。生成 {len(gen)}/4 个场景，接纳 "
               f"{len(base)}/4 条基线、{len(cont)} 条 CONT。接纳回放 {result['accepted_physical_steps']:,} 步，"
               f"{result['accepted_replay_forecasts']:,} 次 forecast。"
               f"基线完整走完 2,017 步 {result['full_baseline_replays']} 条。"
               "新增费用标签及训练均为 0。")
    if selection and not selection["coverage_passed"]:
        summary += ("覆盖门未通过。仅 2 个日期有主状态，0 个有同批大小参考状态。"
                    "固定试采范围已关闭，未启动 CONT。")
    paragraphs = [summary]
    if selection:
        paragraphs.append(f"找到主状态的日期有 {selection['primary_families']} 个，有同批大小参考的日期有 "
                          f"{selection['reference_families']} 个。覆盖门{'通过' if selection['coverage_passed'] else '未通过'}。"
                          f"实际固定 {len(selection['selection'])} 条 CONT。")
    if selection and not selection["coverage_passed"]:
        assert len(base)==4 and len(cont)==0
        reasons = {
            "2039-03-14":"四个状态均为 32 个候选，拓扑分数相同",
            "2039-06-13":"没有符合条件的停止点",
            "2039-09-12":"第 451 步因后端潮流不收敛提前结束",
            "2039-12-12":"主状态有 64 个候选，其余为 22、13、31，无法匹配",
        }
        assert all(record["date"] in reasons for record in base)
        by_date = {row["date"]:row for row in selection["dates"]}
        table = ["| 日期 | 基线步数 | 合格状态 | 缺项原因 |","| --- | ---: | ---: | --- |"]
        for record in base:
            day = record["date"]
            table.append(f"| {day} | {record['summary']['steps']:,} | "
                         f"{by_date[day]['eligible_states']} | {reasons[day]} |")
        paragraphs.append("\n".join(table))
        paragraphs.append("预设门槛为至少 3 个日期有主状态、至少 2 个日期有同批大小且分数严格更低的参考状态。"
                          "本轮实际为 2 个和 0 个。结果反映了这一批固定场景的选点覆盖，"
                          "尚未测到补查动作改变率或费用差。")
    if finished.exists():
        paragraphs.append(f"从管理器启动到最后收据的墙钟为 {ready['wall_s']/60:.1f} 分钟，含两次发布等待。")
    if error.exists():
        paragraphs.append("首次错误已停止，未重启或补跑。停止原因和部分记录见收据及本地归档。")
    paragraphs.append(f"已将输出和新环境原始文件归档到 {B.as_posix()}，共 {raw['logical_files']:,} 个逻辑文件、"
                      f"{raw['logical_bytes']:,} 字节，逐对象 SHA-256 读回通过。")
    report = O/"GRID309_collection_report.md"
    report.write_text("# GRID309 四日期试采收尾\n\n"+"\n\n".join(paragraphs)+"\n",encoding="utf8")
    handoff = O/"COLLECTION_HANDOFF.md"
    handoff.write_text("# GRID309 试采已关闭\n\n"+summary+
         "\n\n禁止重启、补跑或延窗。本轮没有已接纳 CONT，因此没有新的配对费用标签可核验。"
        "保持原 261。若继续这条研究路线，须有新的数据来源或独立的具体研究决定支持。\n",encoding="utf8")
    C.mkdir(parents=True)
    S.mkdir()
    names = ["GRID309_collection_report.md","COLLECTION_HANDOFF.md","collection_closeout.json",
             "raw_records_manifest.json","pipeline_state.json",finished.name if finished.exists() else error.name]
    for name in ("generation_publication.json","baseline_publication.json","manifest.json","selected_markers.json"):
        if (O/name).is_file():
            names.append(name)
    for name in names:
        shutil.copy2(O/name,S/name)
        shutil.copy2(O/name,C/name)
    shutil.copy2(Path(__file__),S/"close_grid309.py")
    save(S/"delivery_hashes.json",{p.name:sha(p) for p in S.iterdir()})
    save(C/"delivery_hashes.json",{name:sha(C/name) for name in names})
    for name in ("RESEARCH_STATE.md","DECISIONS.md","EXPERIMENT_LOG.md"):
        path = I/"docs"/name
        old = path.read_text(encoding="utf8")
        text = ("# 当前 GRID309 四日期试采收尾\n\n"+summary+"\n\n"+old if name=="RESEARCH_STATE.md"
                else old+"\n\n## 2026-11-06 — GRID309 四日期试采收尾\n\n"+summary+"\n")
        path.write_text(text,encoding="utf8")
    (I/"docs/NEXT_STEPS.md").write_bytes(handoff.read_bytes())
    index=I/"research_archive/power-grid-2026/README.md"
    index.write_text("# 最新 GRID309 四日期试采收尾\n\n"+summary+
        "\n\n[采集收尾报告](protocols/GRID309_action_opportunity_preparation/collection_results/GRID309_collection_report.md)\n\n"
        +index.read_text(encoding="utf8"),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    close()
