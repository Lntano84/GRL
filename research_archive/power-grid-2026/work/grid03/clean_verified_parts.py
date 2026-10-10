"""Delete only generated Range parts after verifying each against the retained full archive."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03"
BASE=(ROOT/"work/grid03/data/range_parts").resolve()
ARCHIVE=ROOT/"work/grid03/data/l2rpn_idf_2023.tar.bz2"
state=json.loads((OUT/"download_parallel_state.json").read_text(encoding="utf-8"))
assert state["complete"] and ARCHIVE.stat().st_size==5358993257
expected=[(a,min(a+16*1024**2-1,5358993257-1)) for a in range(state["serial_prefix_bytes"],5358993257,16*1024**2)]
assert set(state["parts"])=={f"{a:012d}-{b:012d}.part" for a,b in expected}
verified=[]
with ARCHIVE.open("rb") as archive:
    for a,b in expected:
        p=(BASE/f"{a:012d}-{b:012d}.part").resolve()
        assert p.is_relative_to(BASE) and p.parent==BASE and p.is_file() and not p.is_symlink()
        assert p.stat().st_size==state["parts"][p.name]==b-a+1
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        archive.seek(a)
        target=hashlib.sha256(archive.read(b-a+1)).hexdigest()
        assert actual==target
        verified.append({"path":str(p),"start":a,"end":b,"size":b-a+1,"sha256":actual})
record={"verified":True,"archive_retained":str(ARCHIVE),"files":verified,"bytes_reclaimed":sum(x["size"] for x in verified),"deleted":False}
path=OUT/"range_part_cleanup.json"
path.write_text(json.dumps(record,indent=2),encoding="utf-8")
for row in verified:
    p=Path(row["path"])
    assert p.resolve().is_relative_to(BASE) and p.parent==BASE
    p.unlink()
record["deleted"]=True
path.write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps({"verified_files":len(verified),"bytes_reclaimed":record["bytes_reclaimed"],"archive_retained":True}),flush=True)
