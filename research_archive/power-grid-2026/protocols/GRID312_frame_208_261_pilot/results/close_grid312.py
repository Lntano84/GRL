"""Audit fixed GRID312 saved step/call ledgers, compare and archive once."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import time
import zipfile

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
O = R/"outputs/grid312_frame_208_261_pilot"
A = I/"research_archive/power-grid-2026/protocols/GRID312_frame_208_261_pilot"
B = R/"archives/GRID312_frame_208_261/raw_records"
C = R/"论文协作资料/FRAME_三臂对照_GRID312"
ARMS = ("FRAME_GUARD","PARENT208","BOUNDED261")
COMPARISONS = (("PARENT208","FRAME_GUARD"),("BOUNDED261","FRAME_GUARD"),("BOUNDED261","PARENT208"))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path,value):
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def percentile(values,q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*q
    lo,hi = math.floor(position),math.ceil(position)
    return values[lo]+(position-lo)*(values[hi]-values[lo])


def key(candidate):
    return struct.unpack("f",struct.pack("f",candidate["reward"]))[0],-candidate["pool_id"]


def checked(record,item,reference,quality_threshold):
    s = record["summary"]
    path = O/"runs"/f"r{s['repeat']}__{s['week']}__{s['method']}"
    assert s==read(path/"summary.json")==read(O/f"worker_result_{s['index']:02d}.json")
    assert all(s[k]==v for k,v in item.items())
    assert s["new_model_fits"]==0
    metadata = read(path/"metadata.json")
    with gzip.open(path/"steps.jsonl.gz","rt",encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows)==s["steps"] and 1<=len(rows)<=2017
    assert [r["step"] for r in rows]==list(range(1,len(rows)+1))
    assert all(a["after_hash"]==b["before_hash"] and not a["done"] for a,b in zip(rows,rows[1:]))
    assert rows[-1]["complete"]==s["complete"]
    assert not s["complete"] or len(rows)==2017
    assert abs(math.fsum(r["raw_cost"] for r in rows)-s["cost"])<1e-6
    assert abs(math.fsum(r["decision_s"] for r in rows)-s["controller_s"])<1e-6
    assert sum(r["simulations"] for r in rows)==s["public_forecasts"]<=record["counted_forecasts"]
    assert all(r["simulations"]==r["all_forecast_calls"] for r in rows)
    assert all(abs(r["ledger"]["recomputed_raw_cost"]-r["raw_cost"])<=r["ledger_tolerance"]
               for r in rows if r["ledger"] is not None)
    assert [sum(bool(r[k]) for r in rows) for k in ("illegal","ambiguous","exceptions")]==[
        s["illegal"],s["ambiguous"],s["exception_steps"]]
    calls = [read(p) for p in sorted(path.glob("call[0-9][0-9][0-9].json"))]
    assert len(calls)==s["n1_calls"] and sum(c["public_queries"] for c in calls)==s["n1_forecasts"]
    fallback = cancels = stops = 0
    for call in calls:
        assert call["learning_method"]==s["arm"] and 1<=call["step"]<=len(rows)
        outcomes = call["outcomes"]
        assert len(outcomes)==len({r["pool_id"] for r in outcomes})==call["public_queries"]
        eligible = []
        for r in outcomes:
            valid = bool(0<r["rho"]<call["rho_limit"] and not any(r[k] for k in ("done","illegal","ambiguous","exceptions")))
            assert valid==r["strict_admissible"]
            if valid:
                eligible.append(r)
        best = max(eligible,key=key,default=None)
        assert call["chosen_pool_id"]==(None if best is None else best["pool_id"])
        fallback += bool(call["engineering_fallback"])
        if s["arm"]=="FRAME_GUARD":
            assert not call["learned_decisions"] and not call["engineering_fallback"]
        for d in call["learned_decisions"]:
            if not d["optional_extension"]:
                if "native_q" in d:
                    assert bool(d["action"])==(d["native_q"][1]>d["native_q"][0])
                continue
            stops += 1
            assert d is call["learned_decisions"][-1] and d["native_q"][1]<=d["native_q"][0]
            expected = d["parent_CONT"]
            if s["arm"]=="BOUNDED261":
                expected = expected and d["residual_logit"]>=0
                cancels += bool(d["parent_CONT"] and not expected)
            assert bool(d["action"])==expected
    active = {x["step"] for x in calls}
    total_time = [r["decision_s"] for r in rows]
    search_time = [r["decision_s"] for r in rows if r["step"] in active]
    fields = ("step","before_hash","action_hash","after_hash","raw_cost","done","complete")
    signature = hashlib.sha256(json.dumps([[r[k] for k in fields] for r in rows],
                                         separators=(",",":")).encode()).hexdigest()
    full_ratio = s["cost"]/reference["cost"] if s["complete"] and reference["complete"] else None
    quality = ("REFERENCE_LIMITED" if not reference["complete"] else "INCOMPLETE" if not s["complete"]
               else "PASS" if full_ratio<=1+quality_threshold else "FAIL_COST")
    return {**{k:s[k] for k in ("index","week","arm","repeat","pool","steps","complete","cost","controller_s",
                                 "exception_steps","illegal","ambiguous","wall_s")},
             "candidate_queries":s["n1_forecasts"],"total_forecasts":s["public_forecasts"],
             "counted_forecasts":record["counted_forecasts"],"signature":signature,
             "initial_hash":rows[0]["before_hash"],"library_hashes":metadata["library_hashes"],
             "dt_hours":metadata["dt_hours"],"cost_per_MW":metadata["cost_per_MW"],"quality":quality,
             "full_cost_ratio":full_ratio,"fallback_calls":fallback,"native_stops":stops,"residual_cancellations":cancels,
             "all_p95":percentile(total_time,.95),"all_p99":percentile(total_time,.99),
             "search_p95":percentile(search_time,.95),"search_p99":percentile(search_time,.99)}


def compare(rows):
    index = {(r["week"],r["repeat"],r["arm"]):r for r in rows}
    pairs = []
    for x,y in COMPARISONS:
        for week,repeat in sorted({(r["week"],r["repeat"]) for r in rows}):
            if not all((week,repeat,arm) in index for arm in (x,y)):
                continue
            a,b = index[week,repeat,x],index[week,repeat,y]
            for k in ("initial_hash","library_hashes","dt_hours","cost_per_MW"):
                assert a[k]==b[k],(week,repeat,x,y,k)
            same = a["steps"]==b["steps"] and a["complete"]==b["complete"]
            full = a["complete"] and b["complete"]
            def change(k):
                return 100*(a[k]/b[k]-1) if same and a[k] is not None and b[k] else None
            pair = {"arm":x,"reference":y,"week":week,"repeat":repeat,"same_horizon":same,
                    "complete_both":full,"same_trajectory":a["signature"]==b["signature"],
                    "candidate_delta":a["candidate_queries"]-b["candidate_queries"] if same else None,
                    "cost_delta_pct":100*(a["cost"]/b["cost"]-1) if full else None}
            for k in ("controller_s","candidate_queries","total_forecasts","all_p95","all_p99","search_p95","search_p99"):
                pair[k+"_change_pct"]=change(k)
            pairs.append(pair)
    return pairs


def average(rows,key):
    values = [r[key] for r in rows if r[key] is not None]
    return math.fsum(values)/len(values) if values else None


def archive():
    assert not B.exists()
    B.mkdir(parents=True)
    seen,entries = {},[]
    start = time.perf_counter()
    with zipfile.ZipFile(B/"raw_objects.zip","x",allowZip64=True) as archive:
        for path in sorted(p for p in O.rglob("*") if p.is_file()):
            assert time.perf_counter()-start<600 and path.resolve().is_relative_to(O.resolve())
            data = path.read_bytes()
            h = hashlib.sha256(data).hexdigest()
            name = "objects/"+h[:2]+"/"+h
            entries.append({"path":path.relative_to(R).as_posix(),"sha256":h,"bytes":len(data),"object":name})
            if h not in seen:
                archive.writestr(name,data,compress_type=zipfile.ZIP_STORED if path.suffix in {".npz",".gz"}
                                else zipfile.ZIP_DEFLATED,compresslevel=1)
                seen[h]=name
    with zipfile.ZipFile(B/"raw_objects.zip") as archive:
        for h,name in seen.items():
            assert time.perf_counter()-start<600 and hashlib.sha256(archive.read(name)).hexdigest()==h
    raw = {"passed":True,"local_archive":B.as_posix(),"logical_files":len(entries),
           "logical_bytes":sum(e["bytes"] for e in entries),"unique_objects":len(seen),
           "zip_sha256":sha(B/"raw_objects.zip"),"independent_object_readback":True,"entries":entries}
    save(B/"raw_records_manifest.json",raw)
    save(B/"archive_complete.json",{"passed":True,"manifest_sha256":sha(B/"raw_records_manifest.json")})
    shutil.copy2(B/"raw_records_manifest.json",O/"raw_records_manifest.json")
    return raw


def close():
    S = A/"results"
    assert not S.exists() and not C.exists()
    assert not (O/"active_worker.json").exists() or not read(O/"active_worker.json")["active"]
    proc = subprocess.check_output(["powershell","-NoProfile","-Command",
        "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and "
        "$_.CommandLine -match 'GRID312_frame_208_261_pilot[/\\\\]runtime\\.py'}).Count"]).decode().strip()
    assert proc=="0"
    finished,error = O/"finished.json",O/"collection_error.json"
    assert finished.exists()!=error.exists()
    end = read(finished if finished.exists() else error)
    d = read(A/"design.json")
    refs = {k:read(R/v) for k,v in d["full_references"].items()}
    manifest = read(O/"manifest.json") if (O/"manifest.json").exists() else {"records":[],"steps":0,"forecasts":0}
    assert len(manifest["records"])==end["accepted"]
    if finished.exists():
        assert end["accepted"]==24 and end["steps"]==manifest["steps"] and end["forecasts"]==manifest["forecasts"]
    rows = [checked(record,item,refs[item["week"]],d["quality_threshold"])
            for record,item in zip(manifest["records"],d["execution_order"])]
    pairs = compare(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["week"],r["arm"])].append(r)
    repeats = [{"week":k[0],"arm":k[1],"same_trajectory":v[0]["signature"]==v[1]["signature"]}
               for k,v in sorted(by.items()) if len(v)==2]
    fields = ("controller_s_change_pct","candidate_queries_change_pct","total_forecasts_change_pct",
              "all_p95_change_pct","all_p99_change_pct","search_p95_change_pct","search_p99_change_pct","cost_delta_pct")
    averages = {}
    for x,y in COMPARISONS:
        part = [p for p in pairs if p["arm"]==x and p["reference"]==y]
        averages[x+"_vs_"+y] = {name:{"pairs":len(g),**{k:average(g,k) for k in fields}} for name,g in
            (("all",part),("full",[p for p in part if p["complete_both"]]),
             ("repeat0",[p for p in part if p["repeat"]==0]),("repeat1",[p for p in part if p["repeat"]==1]))}
    for r in rows:
        del r["library_hashes"]
    save(O/"paired_results.json",{"records":rows,"pairs":pairs,"repeat_checks":repeats,"averages":averages,
                                  "new_physics_in_readout":0,"new_predictions_in_readout":0,"new_fits":0})
    raw = archive()
    save(O/"closeout.json",{"status":"FINISHED" if finished.exists() else "STOPPED","accepted":len(rows),
         "physical_steps":manifest["steps"],"counted_forecasts":manifest["forecasts"],"wall_s":end["wall_s"],
         "raw_records_manifest_sha256":sha(O/"raw_records_manifest.json")})
    def pct(x):
        return "—" if x is None else f"{x:+.3f}%"
    lines = ["# GRID312 原启发式与学习版同批对照","",
      f"24 条计划中接纳 {len(rows)} 条，耗时 {end['wall_s']/60:.1f} 分钟。"
      f"共 {manifest['steps']:,} 物理步及 {manifest['forecasts']:,} 次计入上限的 forecast。"
      "固定 f0_s0、原四个开发周，两次逆序。未训练或重选模型。","",
      "| 对照 | 等时域候选查询变化 | 等时域控制时间变化 | 搜索步 P95 | 搜索步 P99 | 完整周费用变化 |",
      "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for x,y in COMPARISONS:
        v = averages[x+"_vs_"+y]["all"]
        vals = [v[k] for k in ("candidate_queries_change_pct","controller_s_change_pct","search_p95_change_pct","search_p99_change_pct","cost_delta_pct")]
        lines.append("| "+x+" 对 "+y+" | "+" | ".join(map(pct,vals))+" |")
    lines += ["","先逐日期与重复配对再等权平均。查询及时间只在同等时域比较，费用只在双方完整时比较。"
              "早停保留为完成差异，未完成的 FULL 只作参照。各轮计时方向及每条原值见 paired_results.json。","",
              "| 日期 | 方法 | 步数（两重复） | FULL 质量 | 候选查询均值 | 控制时间均值（秒） |",
              "| --- | --- | --- | --- | ---: | ---: |"]
    for (week,arm),rr in sorted(by.items()):
        lines.append(f"| {week} | {arm} | {' / '.join(str(x['steps']) for x in rr)} | "
                     f"{' / '.join(x['quality'] for x in rr)} | {average(rr,'candidate_queries'):.1f} | {average(rr,'controller_s'):.3f} |")
    lines += ["",f"同法跨重复同轨迹 {sum(x['same_trajectory'] for x in repeats)}/{len(repeats)}。"
              "结果仅限单一冻结模型池和已暴露开发日期。","",
              f"原始资料已无损归档并逐对象读回，共 {raw['logical_files']:,} 个文件。"]
    if error.exists():
        lines += ["","出现故障或到限后停止，收据和部分文件保留。"]
    (O/"GRID312_report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    S.mkdir(); C.mkdir(parents=True)
    files = ("GRID312_report.md","paired_results.json","closeout.json","raw_records_manifest.json",
             "pipeline_state.json",finished.name if finished.exists() else error.name)
    for name in files:
        shutil.copy2(O/name,S/name)
        shutil.copy2(O/name,C/name)
    shutil.copy2(Path(__file__),S/"close_grid312.py")
    save(S/"delivery_hashes.json",{p.name:sha(p) for p in S.iterdir()})
    save(C/"delivery_hashes.json",{name:sha(C/name) for name in files})
    print(json.dumps({"accepted":len(rows),"averages":averages,"same_repeats":sum(r["same_trajectory"] for r in repeats)},ensure_ascii=False))


if __name__=="__main__":
    close()
