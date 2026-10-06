"""Fetch arbitrary files from a GitHub repo into a local mirror (handles the >1MB case)."""
from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

HDR = {"User-Agent": "probe"}


def api(repo, path, ref=None):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    if ref:
        url += f"?ref={ref}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60) as r:
        return json.load(r)


def fetch(repo, path, out_root, ref=None):
    dest = Path(out_root) / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    j = api(repo, path, ref)
    if isinstance(j, list):
        print(f"  {path}: directory with {len(j)} entries")
        return dest
    if j.get("content"):
        data = base64.b64decode(j["content"])
        how = "inline"
    else:
        with urllib.request.urlopen(urllib.request.Request(j["download_url"], headers=HDR),
                                    timeout=300) as r:
            data = r.read()
        how = "download_url"
    dest.write_bytes(data)
    print(f"  {path:<50} {len(data):>10,} bytes ({how})")
    return dest


if __name__ == "__main__":
    repo, out_root = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    for p in sys.argv[4:] if len(sys.argv) > 4 else []:
        try:
            fetch(repo, p, out_root, ref)
        except Exception as exc:
            print(f"  FAIL {p}: {exc}")
