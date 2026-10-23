"""Finish publication only, after adding the missing json import; never rerun analysis."""
from pathlib import Path
import json,shutil,textwrap
import finalize as F
from common111 import R,O,W,read,save,sha
REPO=R/'work/GRL-results-integration'
assert read(O/'review.json')['passed'] and read(O/'analysis_receipt.json')['passed']
assert 'GRID111 action-feedback association diagnosis' not in (REPO/'docs/RESEARCH_STATE.md').read_text(encoding='utf8')
complete=read(O/'complete.json');archive=Path(complete['archive']);collab=Path(complete['collaboration'])
save('publication_recovery_01.json',dict(error="NameError: name 'json' is not defined in final archive-manifest serialization.",
    scope='Report and copies were already written; state documents/push had not run. Added import json, finish publication tail only. No scientific data, labels, metrics, fit, or source inputs changed.',
    original_source_sha256=sha(O/'interrupted_attempts/publication_import_01/finalize_v0.py'),fixed_source_sha256=sha(W/'finalize.py')))
for p in O.rglob('*'):
    if p.is_file():d=archive/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
for p in W.glob('*.py'):shutil.copy2(p,archive/p.name)
source=(W/'finalize.py').read_text(encoding='utf8');marker="    (archive/'archive_manifest.json').write_text"
assert source.count(marker)==1
tail=source[source.index(marker):source.index("if __name__=='__main__':main()")]
namespace=dict(F.__dict__);namespace.update(archive=archive,collab=collab)
exec(compile(textwrap.dedent(tail),str(W/'finalize.py'),'exec'),namespace)
