"""Keep the authorized serial run and post-run audit in one managed process."""
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
for script in ['run.py','audit.py']:
 r=subprocess.run([sys.executable,str(HERE/script)],cwd=HERE.parents[1])
 if r.returncode:raise SystemExit(r.returncode)
print('GRID63_REPLAY_AND_AUDIT_COMPLETE',flush=True)
