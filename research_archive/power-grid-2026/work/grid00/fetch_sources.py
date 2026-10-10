"""Fetch small, pinned source files for inspection, never execute them."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "outputs" / "grid00"
SRC = ROOT / "sources"
SHA = "ca0637eab9f098be7f206ed0e46a3900cd4deec0"
FILES = ["README.MD", "LICENSE", "requirements.txt", "make_agent.py", "LJNAgent.py",
         "modules/base_module.py", "modules/topology_heuristic.py",
         "modules/topology_nn_policy.py", "modules/convex_optim.py", "evaluate.py"]
manifest = []
for name in FILES:
    url = f"https://raw.githubusercontent.com/lajavaness/l2rpn-2023-ljn-agent/{SHA}/{name}"
    target = SRC / "ljn" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(url, timeout=30).read()
    target.write_bytes(data)
    manifest.append(dict(source=url, path=str(target.relative_to(ROOT)), bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest()))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "source_manifest.json").write_text(json.dumps(dict(repo="lajavaness/l2rpn-2023-ljn-agent",
    commit=SHA, files=manifest, inspection_only=True), indent=2), encoding="utf-8")
print(json.dumps(dict(files=len(manifest), bytes=sum(x["bytes"] for x in manifest), commit=SHA)))
