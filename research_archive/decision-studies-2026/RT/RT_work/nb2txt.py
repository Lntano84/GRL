"""Extract source from a Jupyter notebook into a readable text file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
nb = json.loads(src.read_text(encoding="utf-8"))
lines = []
for i, cell in enumerate(nb.get("cells", [])):
    kind = cell.get("cell_type")
    s = "".join(cell.get("source", []))
    if kind == "markdown":
        lines.append(f"\n### [md {i}] " + s.replace("\n", "\n# "))
    elif kind == "code":
        lines.append(f"\n=== [code {i}] ===")
        lines.append(s)
out.write_text("\n".join(lines), encoding="utf-8")
print(f"cells: {len(nb.get('cells', []))}  -> {out}  ({out.stat().st_size:,} bytes)")
