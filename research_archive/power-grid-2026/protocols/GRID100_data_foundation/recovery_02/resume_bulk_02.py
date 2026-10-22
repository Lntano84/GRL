"""Resume after a diagnostic-only logger failure, with exact prefix check."""
import gzip
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from bulk_common import ROOT, OUT, W, RUNTIME, read, save, verify, utc


def prefix_check():
    plan = read(OUT / 'recovery_02_plan.json')
    old = ROOT / plan['preserved_failed_folder']
    new = RUNTIME / 'runs/r0__2037-06-01_100__FULL_TEACHER'
    def rows(path):
        with gzip.open(path, 'rt', encoding='utf8') as f:
            return [json.loads(line) for line in f]
    before = rows(old / 'partial_steps.jsonl.gz')
    after = rows(new / 'steps.jsonl.gz')
    assert len(before) == 1157 and len(after) >= len(before)
    fields = ['step', 'before_hash', 'action_hash', 'after_hash', 'raw_cost', 'done', 'complete',
              'illegal', 'ambiguous', 'exceptions', 'simulations']
    for a, b in zip(before, after):
        assert all(a[k] == b[k] for k in fields), a['step']
    with np.load(old / 'partial_vectors.npz') as a, np.load(new / 'vectors.npz') as b:
        for k in ['action', 'observation']:
            assert np.ascontiguousarray(a[k]).tobytes() == np.ascontiguousarray(b[k][:len(before)]).tobytes()
    save(OUT / 'recovery_02_prefix_check.json', dict(passed=True, unchanged_physical_steps=1157,
         unchanged_action_and_observation_vectors=True, no_new_model_fits=True,
         scope='Exact deterministic comparison of the previously saved prefix; no equality claim after the former crash point.', utc=utc()))


plan = read(OUT / 'recovery_02_plan.json')
verify(plan['supplemental_source_hashes'])
verify(read(OUT / 'bulk_code_freeze.json'))
assert read(OUT / 'recovery_02_preflight.json')['passed']
source = W / 'bulk_pipeline.py'
code = source.read_text(encoding='utf8')
marker = "    plan = read(OUT / 'bulk_plan.json'); started = time.monotonic()"
assert code.count(marker) == 1
spent = plan['previous_active_seconds']
code = code.replace(marker, "    plan = read(OUT / 'bulk_plan.json').copy(); started = time.monotonic()\n"
                    f"    plan['total_wall_seconds'] -= {spent!r}\n"
                    f"    plan['generation_wall_seconds'] -= {spent!r}")
marker = "str(W / 'bulk_teacher_worker.py')"
assert code.count(marker) == 1
code = code.replace(marker, "str(W / 'teacher_logger_recovery.py')")
# Keep a failed attempt distinct from the39 completed new teacher episodes.
marker = "        summarize()"
assert code.count(marker) == 1
code = code.replace(marker, marker + "\n        append_recovery_note()")
namespace = {'__file__': str(source), '__name__': 'grid100_recovered_pipeline'}
exec(compile(code, str(source), 'exec'), namespace)
original_run = namespace['run_child']


def checked_run(command, *args, **kwargs):
    original_run(command, *args, **kwargs)
    if str(W / 'teacher_logger_recovery.py') in command and command[-1] == '16':
        prefix_check()


def append_recovery_note():
    assert read(OUT / 'recovery_02_prefix_check.json')['passed']
    path = OUT / 'GRID100_report.md'
    path.write_text(path.read_text(encoding='utf8') + '\n工程记录：第17个教师场景曾在1157步后因优化器缺少日志属性中断，失败轨迹与错误完整保留。仅将原模块logger接入该属性后重跑未完成项，前1157步动作、观测、成本与查询逐项重现；已完成16项未重跑。没有更改数值求解或控制规则。新教师完整条目39之外另有1次失败的部分尝试。\n', encoding='utf8')
    review = read(OUT / 'bulk_review.json')
    review.update(preserved_partial_teacher_failures=1, diagnostic_logger_recovery=True,
                  original_prefix_steps_reproduced=1157)
    save(OUT / 'bulk_review.json', review)


namespace['run_child'] = checked_run
namespace['append_recovery_note'] = append_recovery_note
try:
    namespace['main']('--check' in sys.argv)
except BaseException:
    if '--check' not in sys.argv:
        save(OUT / 'bulk_error.json', dict(utc=utc(), traceback=traceback.format_exc()))
        namespace['state']('stopped_preserving_error')
    raise
