"""Apply the pinned official post-download updates to a separate subset copy."""
import difflib
import hashlib
import json
import shutil
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid03"
RAW = ROOT / "work/grid03/formal_env"
DEST = ROOT / "work/grid03/formal_env_initialized"
COMMIT = "7020a556eab049366e74d28c23c9e2091ff33145"
BASE = f"https://raw.githubusercontent.com/Grid2Op/grid2op-datasets/{COMMIT}/"
assert not DEST.exists(), "Do not overwrite an initialized environment"
started = time.monotonic()
session = requests.Session()
received = 0

def fetch(url):
    global received
    r = session.get(url, timeout=45)
    r.raise_for_status()
    received += len(r.content)
    assert received < 2 * 1024**2, "Small metadata acquisition ceiling"
    return r.content

def sha(b):
    return hashlib.sha256(b).hexdigest()

index_bytes = fetch(BASE + "updates.json")
entry = json.loads(index_bytes)["l2rpn_idf_2023"]
assert set(entry) == {"alerts_info.json", "difficulty_levels.json", "config.py"}
updates = []
for name, spec in entry.items():
    assert spec["base_url"] == "https://api.github.com/repos/bdonnot/grid2op-datasets/contents/updates/"
    assert spec["filename"] == "l2rpn_idf_2023_" + name
    url = BASE + "updates/" + spec["filename"]
    data = fetch(url)
    old = (RAW / name).read_bytes() if (RAW / name).exists() else None
    updates.append((name, data, {"path": name, "source_url": url,
        "raw_sha256": sha(old) if old is not None else None,
        "initialized_sha256": sha(data), "bytes": len(data),
        "diff": "".join(difflib.unified_diff(old.decode("utf-8").splitlines(True) if old else [],
            data.decode("utf-8").splitlines(True), fromfile="raw/"+name, tofile="official_update/"+name))}))

extract = json.loads((OUT / "extraction_manifest.json").read_text(encoding="utf-8"))
DEST.mkdir()
copied = 0
for row in extract["files"]:
    src, dst = RAW / row["path"], DEST / row["path"]
    assert src.resolve().is_relative_to(RAW.resolve()) and dst.resolve().is_relative_to(DEST.resolve())
    assert sha(src.read_bytes()) == row["sha256"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += row["size"]
for name, data, _ in updates:
    (DEST / name).write_bytes(data)

(OUT / "official_updates_pinned.json").write_bytes(index_bytes)
manifest = {"passed": True, "commit": COMMIT, "source_index": BASE+"updates.json",
    "updates_index_sha256": sha(index_bytes), "raw_root": str(RAW), "initialized_root": str(DEST),
    "copied_files": len(extract["files"]), "copied_payload_bytes": copied,
    "metadata_response_body_bytes": received, "updated_files": [x[2] for x in updates],
    "raw_preserved": True, "global_grid2op_update_called": False,
    "basis": "Official DownloadDataset._aux_download invokes UpdateEnv._update_files after extraction. bdonnot/grid2op-datasets resolves to Grid2op/grid2op-datasets.",
    "official_complete_env_hash_claimed": False, "wall_seconds": time.monotonic()-started}
for p in ["Download/DownloadDataset.py", "MakeEnv/UpdateEnv.py"]:
    source = ROOT / "work/grid00/.venv/Lib/site-packages/grid2op" / p
    manifest.setdefault("installed_initialization_sources", {})[p] = {"path": str(source), "sha256": sha(source.read_bytes())}
(OUT / "official_initialization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in manifest.items() if k != "updated_files"},indent=2))
