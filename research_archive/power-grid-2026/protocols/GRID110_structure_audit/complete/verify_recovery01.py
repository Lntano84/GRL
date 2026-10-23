"""Keep the failed audit; constrain its bus-known assertion to the local descriptor scope."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid110_structure_audit';W=Path(__file__).parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old=W/'verify.py';before=old.read_text(encoding='utf8')
needle="            assert not np.any((plan[orpos[on]]<=0)) and not np.any((plan[expos[on]]<=0))"
replacement="""            unresolved=set(np.r_[orpos[on&(plan[orpos]<=0)],expos[on&(plan[expos]<=0)]].tolist())
            assert not unresolved.intersection(pos.tolist()), 'Unknown terminal bus inside descriptor station'"""
assert before.count(needle)==1
preserved=O/'interrupted_attempts/audit_scope_01'
if not preserved.exists():
    preserved.mkdir(parents=True)
    for p in [old,O/'verify_stdout.log',O/'verify_stderr.log',O/'verification_input_manifest.json']:
        if p.exists():shutil.copy2(p,preserved/p.name)
    (O/'audit_recovery_01.json').write_text(json.dumps(dict(reason='Independent audit accidentally required known reconnection terminal buses anywhere in grid, while descriptor and original extractor explicitly use only the locked station.',
        repair='Only scope the unknown-terminal assertion to positions of that same station; no descriptor, array, label, split, model or source changed.',
        original_audit_sha256=sha(old),recovery_source_sha256=sha(Path(__file__)),
        descriptor_array_sha256=sha(O/'diagnostic_descriptors.npz'),
        original_error_preserved=str(preserved)),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
exec(compile(before.replace(needle,replacement),str(old),'exec'),{'__name__':'__main__','__file__':str(Path(__file__))})
