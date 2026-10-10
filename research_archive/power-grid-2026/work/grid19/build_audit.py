"""Reuse the audited independent checks; remove inapplicable old-week pairing."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
text = (ROOT / "work/grid18/audit.py").read_text().replace('outputs/grid18', 'outputs/grid19')
start = text.index('        baseline_path = ')
end = text.index('        steps += len(rows)', start)
text = text[:start] + text[end:]
text = text.replace('"scope": "Independent saved-data action-mask/score/accounting checks, no independent AC solver verification. Borrowed baseline only."',
                    '"scope": "Independent saved-data mask/choice/accounting for new teacher collection; no old-week comparisons, no model fits or independent AC rerun."')
(ROOT / "work/grid19/audit.py").write_text(text, encoding="utf-8")
