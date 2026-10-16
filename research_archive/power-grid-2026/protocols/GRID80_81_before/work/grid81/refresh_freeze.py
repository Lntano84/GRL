"""Pre-execution engineering-only freeze completion; preserve original receipt."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid81'
def main():
    assert not (OUT/'manifest.json').exists() and not (OUT/'RUN.lock').exists()
    p=OUT/'code_freeze.json';original=p.read_bytes();backup=OUT/'code_freeze_pre_engineering_v0.json'
    assert not backup.exists();backup.write_bytes(original)
    hashes=json.loads(original);changed={}
    for x in list((ROOT/'work/grid81').glob('*.py'))+list((ROOT/'work/grid80').glob('*.py')):
        name=x.relative_to(ROOT).as_posix();h=hashlib.sha256(x.read_bytes()).hexdigest()
        if name in hashes and h!=hashes[name]:changed[name]=dict(before=hashes[name],after=h)
        hashes[name]=h
    p.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    receipt=dict(reason='Complete audit/report files and require generator qualification before any physical replay; no model, random rate, design, dates, actions or thresholds changed.',changed=changed,new_fits=0,closed_loop_runs=0)
    (OUT/'pre_execution_engineering.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
