"""Continue the already frozen fresh-date matrix after the audit-only recovery."""
from pathlib import Path
import json,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid68'
assert json.loads((ROOT/'outputs/grid67/review.json').read_text())['passed_saved_data']
try:
 if not (OUT/'finished.json').exists():subprocess.run([sys.executable,str(ROOT/'work/grid67/run.py'),'--stage','68'],cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
 if not (OUT/'review.json').exists():subprocess.run([sys.executable,str(ROOT/'work/grid67_review/audit_corrected.py'),'--stage','68'],cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
 (OUT/'continued_pipeline_finished.json').write_text(json.dumps({'passed':True,'new_physical_replays_repeated':0,'original_pipeline_error_preserved':True}),encoding='utf-8')
except BaseException:
 (OUT/'continued_pipeline_error.json').write_text(json.dumps({'traceback':traceback.format_exc()}),encoding='utf-8');raise
