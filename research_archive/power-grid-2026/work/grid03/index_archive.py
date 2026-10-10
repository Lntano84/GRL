"""Index names/sizes and generation seed IDs, without reading time-series values."""
import json
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid03"
ARCHIVE = ROOT / "work/grid03/data/l2rpn_idf_2023.tar.bz2"


def main():
    state = json.loads((OUT / "download_parallel_state.json").read_text())
    assert state["complete"] and ARCHIVE.stat().st_size == 5358993257
    start = time.monotonic()
    entries, scenarios, seeds, small_static = [], defaultdict(list), {}, {}
    with tarfile.open(ARCHIVE, "r|bz2") as tf:
        for member in tf:
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "l2rpn_idf_2023":
                raise RuntimeError("Unsafe or unexpected archive member")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError("Links/special entries are not accepted")
            if member.size > 128 * 1024**2:
                raise RuntimeError("Individual member size ceiling")
            entries.append({"name": member.name, "size": member.size, "file": member.isfile()})
            if len(p.parts) >= 3 and p.parts[1] == "chronics":
                name = p.parts[2]
                if not re.fullmatch(r"2035-\d{2}-\d{2}_\d+", name):
                    raise RuntimeError(f"Unrecognized scenario naming: {name}")
                if member.isfile():
                    scenarios[name].append({"file": "/".join(p.parts[3:]), "size": member.size})
                if len(p.parts) == 4 and p.parts[3] == "_seeds_info.json":
                    assert member.size < 16384
                    seeds[name] = json.load(tf.extractfile(member))
            elif member.isfile() and len(p.parts) == 2 and p.name in ["scenario_params.json", "chronix2grid_adddata_kwargs.json"]:
                small_static[p.name] = json.load(tf.extractfile(member))
            if len(entries) % 1000 == 0:
                print(json.dumps({"members": len(entries), "scenarios_seen": len(scenarios), "wall_s": round(time.monotonic()-start,1)}),flush=True)
    index = {"archive_sha256": state["sha256"], "entries": entries,
             "scenario_files": dict(scenarios), "generation_seeds": seeds, "small_static": small_static,
             "members": len(entries), "scenario_count": len(scenarios),
             "sum_member_file_bytes": sum(x["size"] for x in entries if x["file"]),
             "wall_s": time.monotonic()-start,
             "read_permissions": "Names, sizes, generation seed IDs and generation date list only; no time-series values or agent outcomes."}
    (OUT / "archive_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in index.items() if k not in ["entries","scenario_files","generation_seeds","small_static"]}),flush=True)


if __name__ == "__main__":
    main()
