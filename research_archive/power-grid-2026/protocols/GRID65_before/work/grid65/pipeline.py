"""One detached sequential experiment pipeline, no concurrent learning during timing."""
from pathlib import Path
import json,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid65'
try:
    for name in ['run.py','audit.py']:
        p=subprocess.run([sys.executable,'-u',str(ROOT/'work/grid65'/name)],cwd=ROOT)
        if p.returncode:raise RuntimeError(f'{name}: {p.returncode}')
    (OUT/'pipeline_finished.json').write_text(json.dumps({'passed':True,'runs':70}),encoding='utf-8')
    print('GRID65_REPLAY_AND_AUDIT_COMPLETE',flush=True)
except BaseException:
    (OUT/'pipeline_error.json').write_text(json.dumps({'traceback':traceback.format_exc()}),encoding='utf-8');raise
