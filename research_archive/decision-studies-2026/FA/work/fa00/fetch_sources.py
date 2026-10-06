"""Download only the frozen source modules and ASP data needed for FA00."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = "7a5727651a92fd2fa4960dcbbd7f6dab94130028"
BASE = f"https://raw.githubusercontent.com/stacs-cp/JAIR2026-FrugalAS/{COMMIT}/"
FILES = ["Algorithm_Selection_Pareto.py", "ActiveRFModel.py", "PassiveRFModel.py",
         "PassiveRFRegressor.py", "README.md"]
FILES += [f"DATASETS/ASP-POTASSCO/{n}" for n in
          ["algorithm_runs.arff", "feature_values.arff", "feature_costs.arff",
           "feature_runstatus.arff", "cv.arff", "description.txt", "readme.txt", "citation.bib"]]

manifest = []
for name in FILES:
    path = ROOT / "upstream" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(BASE + name, timeout=45) as response:
            path.write_bytes(response.read())
    blob = path.read_bytes()
    manifest.append({"path": name, "url": BASE + name, "bytes": len(blob),
                     "sha256": hashlib.sha256(blob).hexdigest()})
    print(name, len(blob), flush=True)
(ROOT / "source_manifest.json").write_text(json.dumps({"commit": COMMIT, "files": manifest},
                                                       indent=2), encoding="utf-8")
