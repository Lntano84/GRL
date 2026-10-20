"""Serialization-only recovery. Frozen extractor remains byte-identical."""
from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
import numpy as np
import build as B
ROOT=B.ROOT;OUT=B.OUT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    B.check_freeze();assert not (OUT/'features.json').exists()
    before={p.name:sha(p) for p in (OUT/'contexts').glob('*.json')};assert len(before)==56
    archive=OUT/'interrupted_attempts/serialization01';archive.mkdir(parents=True,exist_ok=False)
    for n in ['build_stdout.log','build_stderr.log','freeze.json']:shutil.copy2(OUT/n,archive/n)
    with zipfile.ZipFile(archive/'completed_contexts.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (OUT/'contexts').glob('*.json'):z.write(p,p.name)
    def default(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    def save(name,data):
        p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=default)+'\n',encoding='utf8')
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='First extraction reached S055 then failed saving static template metadata: numpy.int64 is not JSON serializable.',
        frozen_build_sha256=sha(Path(B.__file__)),recovery_source_sha256=sha(Path(__file__)),change='Only replace JSON serializer with numpy scalar/item and array/list conversion; all frozen feature code unchanged.',
        first_attempt_completed_contexts_sha256=before,physical_step_or_forecast_calls=0,first_attempt_per_case_times='in-memory data lost on final metadata serialization; not fabricated; second extraction times separately saved',
        retry='Repeat saved-data extraction and all checks only; no fits/predictions/physical rollouts. Two total static template initializations including interrupted attempt.')
    save('recovery_01.json',receipt)
    B.save=save;B.main()
    after={p.name:sha(p) for p in (OUT/'contexts').glob('*.json')};assert after==before
    receipt.update(passed=True,completed_contexts_identical=True,features_or_label_definitions_changed=False,freeze_unchanged=True)
    B.check_freeze();save('recovery_01.json',receipt)
if __name__=='__main__':main()
