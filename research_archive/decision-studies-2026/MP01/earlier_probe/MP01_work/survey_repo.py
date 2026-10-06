"""Survey a repo branch: which files exist, and where does #include <boost/...> appear?

Decides whether replacing Boost is a small, contained patch or a rewrite.
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HDR = {"User-Agent": "probe"}


def api(repo, path, ref=None):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    if ref:
        url += f"?ref={ref}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60) as r:
        return json.load(r)


def walk(repo, path, ref, out):
    """Recursively list files; returns list of (path, size)."""
    j = api(repo, path, ref)
    files = []
    if isinstance(j, list):
        for e in j:
            if e["type"] == "dir":
                files += walk(repo, e["path"], ref, out)
            elif e["type"] == "file":
                files.append((e["path"], e.get("size", 0)))
    return files


def get_text(repo, path, ref):
    j = api(repo, path, ref)
    if not j.get("content"):
        with urllib.request.urlopen(urllib.request.Request(j["download_url"], headers=HDR),
                                    timeout=120) as r:
            return r.read().decode("utf-8", "replace")
    return base64.b64decode(j["content"]).decode("utf-8", "replace")


def main() -> int:
    repo, ref = sys.argv[1], sys.argv[2]
    print("=" * 100)
    print(f"  SURVEY {repo} @ {ref}")
    print("=" * 100)
    files = walk(repo, "", ref, {})
    src = [f for f in files if f[0].endswith((".cpp", ".h", ".hpp", ".cc"))]
    print(f"  total files {len(files)}, source files {len(src)}")
    print(f"  total source bytes {sum(s for _, s in src):,}")

    boost_inc = Counter()
    nloh_inc = Counter()
    eigen_inc = Counter()
    per_file = {}
    for path, size in src:
        try:
            t = get_text(repo, path, ref)
        except Exception as exc:
            print(f"    skip {path}: {exc}")
            continue
        hits = []
        for line in t.splitlines():
            ls = line.strip()
            if ls.startswith("#include") and "boost/" in ls:
                hits.append(ls)
                lib = ls.split("boost/")[1].split(".")[0].split(">")[0]
                boost_inc[lib] += 1
            if ls.startswith("#include") and "nlohmann" in ls:
                nloh_inc[ls] += 1
            if ls.startswith("#include") and "Eigen" in ls:
                eigen_inc[ls] += 1
        if hits:
            per_file[path] = hits

    print(f"\n  BOOST INCLUDES -- {sum(boost_inc.values())} occurrences in "
          f"{len(per_file)} files")
    for lib, n in boost_inc.most_common():
        print(f"    boost/{lib:<28} {n}")
    print(f"\n  NLOHMANN INCLUDES: {dict(nloh_inc)}")
    print(f"  EIGEN INCLUDES: {dict(eigen_inc)}")
    print(f"\n  FILES USING BOOST:")
    for p, hits in sorted(per_file.items()):
        print(f"    {p}")
        for h in sorted(set(hits)):
            print(f"        {h}")

    # find boost:: symbols actually referenced (not just included)
    print(f"\n  boost:: SYMBOL USAGE")
    sym = Counter()
    for path, _ in src:
        try:
            t = get_text(repo, path, ref)
        except Exception:
            continue
        for line in t.splitlines():
            if "boost::" in line:
                for tok in line.split():
                    if tok.startswith("boost::"):
                        sym[tok.strip("();,")] += 1
    for s, n in sym.most_common(20):
        print(f"    {s:<48} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
