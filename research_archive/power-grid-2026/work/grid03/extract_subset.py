"""Extract only the frozen members. Never use archive-provided paths unchecked."""
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03"
DEST=ROOT/"work/grid03/formal_env"
ARCHIVE=ROOT/"work/grid03/data/l2rpn_idf_2023.tar.bz2"
splitpath=OUT/"scenario_split_frozen.json"
split=json.loads(splitpath.read_text(encoding="utf-8"))
assert hashlib.sha256(splitpath.read_bytes()).hexdigest()==(OUT/"scenario_split_frozen.sha256").read_text().strip()
chosen=set(sum(split["extracted_subsets"].values(),[]))
static=set(split["static_members"])
if DEST.exists():
    raise RuntimeError("Existing extraction; do not overwrite")
DEST.mkdir(parents=True)
base=DEST.resolve()
total=0
manifest=[]
start=time.monotonic()
with tarfile.open(ARCHIVE,"r|bz2") as tf:
    for m in tf:
        p=PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts or p.parts[0]!="l2rpn_idf_2023" or not(m.isfile() or m.isdir()):
            raise RuntimeError("Unsafe member")
        wanted=m.name in static or (len(p.parts)>=4 and p.parts[1]=="chronics" and p.parts[2] in chosen)
        if not wanted or not m.isfile():
            continue
        dst=(DEST/Path(*p.parts[1:])).resolve()
        if not dst.is_relative_to(base):
            raise RuntimeError("Path escapes intended extraction root")
        if dst.exists():
            raise RuntimeError("Duplicate member")
        total+=m.size
        if total>8*1024**3 or m.size>128*1024**2:
            raise RuntimeError("Extraction payload ceiling")
        dst.parent.mkdir(parents=True,exist_ok=True)
        sha=hashlib.sha256()
        written=0
        with tf.extractfile(m) as src,dst.open("xb") as f:
            for buf in iter(lambda:src.read(1024**2),b""):
                written+=len(buf)
                f.write(buf)
                sha.update(buf)
            f.flush()
            os.fsync(f.fileno())
        assert written==m.size
        manifest.append({"path":str(dst.relative_to(DEST)).replace("\\","/"),"size":m.size,"sha256":sha.hexdigest()})
        if len(manifest)%100==0:
            print(json.dumps({"extracted_files":len(manifest),"payload_bytes":total,"wall_s":round(time.monotonic()-start,1)}),flush=True)
assert total==split["planned_extraction_bytes"]
assert {x.name for x in (DEST/"chronics").iterdir()}==chosen
result={"complete":True,"root":str(DEST),"file_count":len(manifest),"payload_bytes":total,"wall_s":time.monotonic()-start,
        "archive_sha256":split["archive_sha256"],"split_sha256":hashlib.sha256(splitpath.read_bytes()).hexdigest(),"files":manifest}
(OUT/"extraction_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="files"}),flush=True)
