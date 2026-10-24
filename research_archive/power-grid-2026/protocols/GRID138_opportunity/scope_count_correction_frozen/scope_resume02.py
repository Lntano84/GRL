"""Correct the diagnostic count assertion;retain original scripts and logs."""
from run import *
from scope_resume01 import v

def main():
    v();assert not (O/'historical_timing_runs.json').exists();assert not (O/'scope_count_correction.json').exists()
    src=W/'scope_resume01.py';code=src.read_text(encoding='utf8')
    old="assert len(rows)==25 and len(different)==44"
    new="assert len(rows)==25 and len(different)==sum(r['calls'] for r in rows if r['cohort']=='G2')"
    assert code.count(old)==1;code=code.replace(old,new)
    d=O/'interrupted_checks/timing_count02';d.mkdir(parents=True,exist_ok=False)
    for p in [src,O/'scope_resume01.log']:shutil.copy2(p,d/p.name)
    save('scope_count_correction.json',dict(reason='Recovery checker hardcoded44 mismatch calls from manual count;all six G2 references actually contain48. Replace hardcoded count with independent sum of their recorded call counts.',
        scientific_estimator_unchanged=True,original_code_logs_preserved=True,no_completed_outputs_repeated=True,fits=0,physical=0,
        script_sha256=sha(Path(__file__)),old_script_sha256=sha(src),replacement=new))
    frozen=A/'scope_count_correction_frozen';frozen.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),O/'scope_count_correction.json']:
        shutil.copy2(p,frozen/p.name)
    shutil.copytree(d,frozen/'interrupted_check')
    (frozen/'archive_manifest.json').write_text(json.dumps({p.relative_to(frozen).as_posix():sha(p) for p in frozen.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
    sys.argv=[str(src),'run'];exec(compile(code,str(src),'exec'),{'__name__':'__main__','__file__':str(src)})
    v();assert sha(Path(__file__))==read(O/'scope_count_correction.json')['script_sha256'];print('SCOPE_COUNT_CORRECTION_VERIFIED')
if __name__=='__main__':main()
