"""Resume the frozen collector after user shutdown; retain spent wall budget."""
import hashlib
from pathlib import Path
from bulk_common import ROOT, OUT, W, read, verify, test_sealed

record = read(OUT / 'recovery_01.json')
assert record['passed'] and record['reason'] == 'user_shutdown_resume'
verify(read(OUT / 'bulk_code_freeze.json'))
test_sealed()
source = W / 'bulk_pipeline.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == record['original_pipeline_sha256']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == record['resume_source_sha256']
code = source.read_text(encoding='utf8')
marker = "    plan = read(OUT / 'bulk_plan.json'); started = time.monotonic()"
assert code.count(marker) == 1
spent = record['previous_active_seconds']
replacement = ("    plan = read(OUT / 'bulk_plan.json').copy(); started = time.monotonic()\n"
               f"    plan['total_wall_seconds'] -= {spent!r}\n"
               f"    plan['generation_wall_seconds'] -= {spent!r}\n"
               "    assert plan['total_wall_seconds'] > 0 and plan['generation_wall_seconds'] > 0")
code = code.replace(marker, replacement)
exec(compile(code, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
