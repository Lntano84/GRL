"""Supply the optimizer's existing module logger; no numeric/control change."""
import sys
from bulk_common import ROOT, OUT, W, read, verify

record = read(OUT / 'recovery_02_plan.json')
verify(record['supplemental_source_hashes'])
source = W / 'bulk_teacher_worker.py'
code = source.read_text(encoding='utf8')
marker = 'code = code.replace(marker, patch + marker)'
assert code.count(marker) == 1
addition = ('patch += "    import sys\\n'
            '    optimizer_module = sys.modules[OptimModule.__module__]\\n'
            '    assert not hasattr(OptimModule, \'logger\')\\n'
            '    OptimModule.logger = optimizer_module.logger\\n"\n')
code = code.replace(marker, addition + marker)
exec(compile(code, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
