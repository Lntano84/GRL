"""Fetch specific files from the LimeQO repo into a local mirror, preserving paths."""
from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "zixy17/LimeQO"
OUT = HERE / "limeqo_mirror"
HDR = {"User-Agent": "probe"}


def api(path):
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch(path):
    dest = OUT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    j = api(path)
    if j.get("content"):
        data = base64.b64decode(j["content"])
        how = "inline"
    else:
        # files over 1 MB come back with content=None; use the download_url instead
        req = urllib.request.Request(j["download_url"], headers=HDR)
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        how = "download_url"
    dest.write_bytes(data)
    print(f"  {path:<46} {len(data):>10,} bytes  ({how})")
    return dest


def main() -> int:
    targets = sys.argv[1:]
    if not targets:
        print("usage: q01_fetch.py <repo/path> [...]")
        return 2
    for t in targets:
        try:
            fetch(t)
        except Exception as exc:
            print(f"  FAIL {t}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
