"""Sequential finite experiments then audits. No automatic tuning or retries."""
from pathlib import Path
import json,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[2]
def main():
 try:
  for stage in (67,68):
   out=ROOT/f'outputs/grid{stage}'
   if not (out/'finished.json').exists():subprocess.run([sys.executable,str(ROOT/'work/grid67/run.py'),'--stage',str(stage)],cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
   if not (out/'review.json').exists():subprocess.run([sys.executable,str(ROOT/'work/grid67/audit.py'),'--stage',str(stage)],cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
  (ROOT/'outputs/grid68/pipeline_finished.json').write_text(json.dumps({'passed':True,'stages':[67,68],'new_fits':0}),encoding='utf-8')
 except BaseException:
  (ROOT/'outputs/grid68/pipeline_error.json').write_text(json.dumps({'traceback':traceback.format_exc()}),encoding='utf-8');raise
if __name__=='__main__':main()
