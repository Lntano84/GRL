"""Import-path-only audit recovery. Frozen fitting and evaluation code remain unchanged."""
import datetime,hashlib,importlib.util,json,shutil,traceback
from pathlib import Path
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent
spec=importlib.util.spec_from_file_location('g168_unchanged_eval',W/'run.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

def main():
    S.verify_trained();O=S.O;assert not (O/'recovery01.json').exists()
    assert (O/'evaluation_started.json').exists() and not (O/'state_scores.npz').exists() and not (O/'review.json').exists()
    dest=O/'interrupted_attempts/audit_import_01';dest.mkdir(parents=True,exist_ok=False)
    shutil.move(str(O/'evaluation_started.json'),str(dest/'evaluation_started.json'))
    error="ModuleNotFoundError: No module named 'common' while importing grid113/audit.py, before data/model forward."
    (dest/'error.txt').write_text(error+'\n',encoding='utf8')
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason=error,fix='Initialize the existing G.old_api() import paths BEFORE importing unchanged independent NumPy audit.',frozen_sources_unchanged=True,trained_models_unchanged=True,new_fits=0,new_data=0,new_physics=0,first_attempt_forward_rows=0,wrapper_sha256=S.sha(Path(__file__)),frozen_run_sha256=S.sha(W/'run.py'))
    S.save('recovery01.json',receipt)
    S.G.old_api()  # This was already called later in evaluate(); only its import order changes.
    S.evaluate();S.verify_trained();print('GRID168_IMPORT_RECOVERY_EVALUATION_COMPLETE')

if __name__=='__main__':main()
