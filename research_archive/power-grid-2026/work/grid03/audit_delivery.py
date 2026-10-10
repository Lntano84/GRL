"""Read-back audit of exact delivery files; no simulations or model fitting."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03"


def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""):
            h.update(block)
    return h.hexdigest()


download=load("download_parallel_state.json")
archive=ROOT/"work/grid03/data/l2rpn_idf_2023.tar.bz2"
assert download["complete"] and archive.stat().st_size==5358993257
assert sha(archive)==download["sha256"]
assert download["body_bytes_received"]<=6*1024**3
index=load("archive_index.json")
split=load("scenario_split_frozen.json")
extract=load("extraction_manifest.json")
schema=load("schema_check.json")
unit=load("metric_unit_checks.json")
score=load("scoring_smoke/summary.json")
SMOKE="formal_smoke_corrected_ledger/"
smoke=load(SMOKE+"summary.json")
initialization=load("official_initialization_manifest.json")
assert index["archive_sha256"]==split["archive_sha256"]==extract["archive_sha256"]==download["sha256"]
assert sha(OUT/"scenario_split_frozen.json")==extract["split_sha256"]==(OUT/"scenario_split_frozen.sha256").read_text().strip()
assert sha(OUT/"archive_index.json")==split["index_sha256"]
assert index["scenario_count"]==832
whole=split["all_splits"]
assert len(set(sum(whole.values(),[])))==sum(map(len,whole.values()))==832
assert set(sum(whole.values(),[]))==set(index["scenario_files"])
chosen=split["extracted_subsets"]
assert {k:len(v) for k,v in chosen.items()}=={"development":24,"validation":12,"test":12}
for k,v in chosen.items():
    assert set(v)<=set(whole[k])
    assert Counter(n[5:7] for n in v)=={f"{m:02d}":2 if k=="development" else 1 for m in range(1,13)}
checked=0
for row in extract["files"]:
    p=ROOT/"work/grid03/formal_env"/row["path"]
    assert p.resolve().is_relative_to((ROOT/"work/grid03/formal_env").resolve())
    assert p.stat().st_size==row["size"] and sha(p)==row["sha256"]
    checked+=1
assert initialization["passed"] and initialization["raw_preserved"] and not initialization["global_grid2op_update_called"]
assert sha(OUT/"official_updates_pinned.json")==initialization["updates_index_sha256"]
assert initialization["copied_files"]==checked and initialization["copied_payload_bytes"]==extract["payload_bytes"]
update_by_path={r["path"]:r for r in initialization["updated_files"]}
assert set(update_by_path)=={"alerts_info.json","difficulty_levels.json","config.py"}
initialized_checked=0
for row in extract["files"]:
    raw=ROOT/"work/grid03/formal_env"/row["path"]
    p=ROOT/"work/grid03/formal_env_initialized"/row["path"]
    if row["path"] in update_by_path:
        u=update_by_path[row["path"]]
        assert sha(raw)==u["raw_sha256"] and sha(p)==u["initialized_sha256"]
    else:
        assert sha(p)==row["sha256"]
    initialized_checked+=1
for name,u in update_by_path.items():
    p=ROOT/"work/grid03/formal_env_initialized"/name
    assert p.stat().st_size==u["bytes"] and sha(p)==u["initialized_sha256"]
assert not (ROOT/"work/grid03/formal_env/alerts_info.json").exists()
raw_check=load("formal_smoke/execution_count.json")
assert raw_check["physical_steps"]==0
assert set(load("formal_smoke/compatibility.json")["differences"])=={"action_vect_size","obs_vect_size","alertable_line_ids"}
assert sum(x["size"] for x in extract["files"])==extract["payload_bytes"]==split["planned_extraction_bytes"]<=8*1024**3
assert schema["passed"] and schema["count"]==48 and unit["passed"] and score["passed"] and smoke["passed"]
assert smoke["scenario"] in chosen["development"]
assert not smoke["test_or_validation_agent_outcomes_used"]
assert not load(SMOKE+"compatibility.json")["differences"]
passive=load(SMOKE+"passive_steps.json")
nn=load(SMOKE+"nn_steps.json")
mapping=load(SMOKE+"mapping_checks.json")
assert len(mapping)==352 and all(not row["ambiguous"] for row in mapping)
assert all(r["same_as_unscored"] for r in passive)
assert all(abs(r["residual"])<=r["rounding_tolerance"] for r in passive)
for r in passive+nn:
    x=r["ledger"]
    # Recompute saved arithmetic without importing the metric adapter/reward class.
    independently_recomputed=x["marginal_cost"]*(x["losses_mwh"]+x["redispatch_mwh"]+x["curtailment_delta_mwh"]+x["storage_throughput_mwh"])
    assert abs(independently_recomputed-x["recomputed_raw_cost"])<1e-9
    assert abs(x["curtailment_delta_mwh"]-(x["current_curtailed_mw"]-x["previous_curtailed_mw"])/12)<1e-9
    assert abs(x["curtailed_mwh"]-x["current_curtailed_mw"]/12)<1e-9
initial_unit=load("metric_unit_checks_initial.json")
assert all(unit["source_sha256"][k]==v for k,v in initial_unit["source_sha256"].items())
for path,expected in unit["source_sha256"].items():
    assert sha(ROOT/"work/grid00/.venv/Lib/site-packages/grid2op"/path)==expected
for row in initialization["installed_initialization_sources"].values():
    assert sha(Path(row["path"]))==row["sha256"]
failed_prefix=load("formal_smoke_initialized/passive_steps.json")
assert len(failed_prefix)==3
assert all(x["action_hash"]==y["action_hash"] and x["observation_hash"]==y["observation_hash"] for x,y in zip(failed_prefix,passive))
assert all(not r["native_flags"]["illegal"] and not r["native_flags"]["ambiguous"] and not r["native_flags"]["exceptions"] for r in passive+nn)
failed_ledger_count=load("formal_smoke_initialized/execution_count.json")["physical_steps"]
assert failed_ledger_count==8
actual=score["physical_steps"]+raw_check["physical_steps"]+failed_ledger_count+load(SMOKE+"execution_count.json")["physical_steps"]
assert actual<=100
reused=load("reused_assets_frozen.json")
assert all(sha(ROOT/path)==digest for path,digest in reused.items())
summary={"passed":True,"archive_sha256":download["sha256"],"archive_bytes":archive.stat().st_size,
    "download_application_body_bytes":download["body_bytes_received"],"archive_scenarios":832,"extracted_scenarios":48,
    "extraction_payload_bytes":extract["payload_bytes"],"extracted_files_rehashed":checked,"initialized_copy_files_rehashed":initialized_checked+1,
    "official_initialization_commit":initialization["commit"],"official_updated_files":list(update_by_path),
    "official_initialization_metadata_response_bytes":initialization["metadata_response_body_bytes"],
    "raw_static_failure_physical_steps":raw_check["physical_steps"],"total_physical_steps":actual,
    "failed_stateless_ledger_physical_steps":failed_ledger_count,
    "mapping_rows":352,"native_horizon":smoke["native_horizon"],"passive_cost_residual_max":smoke["max_cost_abs_residual"],"reused_assets_unchanged":len(reused),
    "scripted_probe_steps_applied":sum(r["probe_applied"] for r in passive),
    "saved_ledgers_arithmetic_rechecked":len(passive)+len(nn),"unmodified_library_source_hashes_rechecked":len(unit["source_sha256"]),
    "formal_passive_completed_steps":len(passive),"formal_nn_completed_steps":len(nn),
    "schema_qualification_only":True,"competition_exact_normalization_certified":False,
    "audit_scope":"File integrity, disjoint groups, quota and saved smoke checks. Not independent AC physics revalidation or full-horizon policy evaluation."}
(OUT/"GRID03_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
