"""Freeze a grouped, calendar-stratified subset using only archive naming/seed metadata."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03"
index=json.loads((OUT/"archive_index.json").read_text(encoding="utf-8"))
names=sorted(index["scenario_files"])
suffixes=sorted({int(n.rsplit("_",1)[1]) for n in names})
dates=sorted({n.split("_")[0] for n in names})
assert len(names)==832 and suffixes==list(range(16)) and len(dates)==52
assert set(names)=={f"{date}_{rep}" for date in dates for rep in suffixes}
assert set(index["generation_seeds"])==set(names)
seed="GRID03-20261008-grouped-v1"

def rank(text):
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()

ordered=sorted([x for x in suffixes if x!=0],key=lambda x:rank(f"replica:{x}"))
groups={"development":[0]+ordered[:9],"validation":ordered[9:12],"test":ordered[12:15]}
all_splits={split:[n for n in names if int(n.rsplit("_",1)[1]) in reps] for split,reps in groups.items()}
selected={}
for split,pool in all_splits.items():
    take=2 if split=="development" else 1
    selected[split]=sorted(n for month in range(1,13) for n in sorted(
        [x for x in pool if int(x[5:7])==month],key=lambda x:rank(f"scenario:{x}"))[:take])
    assert len(selected[split])==(24 if split=="development" else 12)
assert not(set(groups["development"])&set(groups["validation"]) or set(groups["development"])&set(groups["test"]) or set(groups["validation"])&set(groups["test"]))
known=["2035-01-15_0","2035-08-20_0"]
assert all(n in all_splits["development"] for n in known)
chosen=set(sum(selected.values(),[]))
static=[x for x in index["entries"] if x["file"] and len(Path(x["name"]).parts)==2]
payload=sum(x["size"] for x in static)+sum(f["size"] for n in chosen for f in index["scenario_files"][n])
assert payload < 8*1024**3
seed_counters={key:Counter(vals[key] for vals in index["generation_seeds"].values()) for key in ["load_seed","renew_seed","gen_p_forecast_seed"]}
duplicate_seeds={k:sum(n-1 for n in count.values() if n>1) for k,count in seed_counters.items()}
split={"archive_sha256":index["archive_sha256"],"index_sha256":hashlib.sha256((OUT/"archive_index.json").read_bytes()).hexdigest(),
    "rule_seed":seed,"group_rule":"Whole suffix groups; suffix 0 forced into development because two bundled prefixes already used. Remaining suffixes sorted by SHA-256.",
    "replica_groups":groups,"all_splits":all_splits,"extracted_subsets":selected,"subset_rule":"SHA-256 rank per calendar month, 2 development / 1 validation / 1 test per month.",
    "planned_extraction_bytes":payload,"static_members":[x["name"] for x in static],"known_bundled_prefixes":known,
    "duplicate_generation_seed_counts":duplicate_seeds,
    "sealed_test_usage":"No agent outcomes will be evaluated on validation/test in GRID03. Schema and integrity checks are allowed.",
    "limitations":["Suffix grouping is a conservative partition convention, not proof of statistical independence.","These are synthetic public training scenarios, not real operational logs.","Pretrained LJN training identities are unknown: not certified unseen by the released model."]}
target=OUT/"scenario_split_frozen.json"
if target.exists():
    raise RuntimeError("Split already frozen")
target.write_text(json.dumps(split,indent=2),encoding="utf-8")
(OUT/"scenario_split_frozen.sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest()+"\n",encoding="ascii")
print(json.dumps({"groups":groups,"subset_counts":{k:len(v) for k,v in selected.items()},"planned_extraction_bytes":payload,"duplicate_generation_seeds":duplicate_seeds}),flush=True)
