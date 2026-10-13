"""Audit-only repair: use the functioning workspace PowerShell7 event-log host."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid55b'
source_path = ROOT / 'work/grid55b/audit.py'
source = source_path.read_text(encoding='utf-8')
host = shutil.which('pwsh.exe')
assert host and Path(host).is_file()
assert source.count("['powershell.exe'") == 2
modified = source.replace("['powershell.exe'", '[' + repr(host))
old_failure = {
    'stage': 'power_log_access_probe_before_scientific_audit',
    'error': "WindowsPowerShell5 Get-WinEvent autoload failed: Microsoft.PowerShell.Diagnostics could not be loaded. Filtered empty event list was NOT accepted as proof; system-log access guard stopped audit.",
    'old_audit_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
    'runs_repeated': 0, 'model_or_data_changed': False,
    'repair': 'Use already-installed workspace PowerShell7 for both event-log subprocesses; all scientific checks and selection code unchanged.'}
record = OUT / 'audit_v0_power_host_failure.json'
if not record.exists(): record.write_text(json.dumps(old_failure, indent=2), encoding='utf-8')
meta = {'only_source_substitution': 'Two powershell.exe hosts -> installed pwsh.exe', 'host_path': host,
    'original_audit_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
    'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'old_frozen_audit_retained': True, 'new_solver_runs': 0, 'new_fits': 0}
(OUT / 'audit_revision.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
namespace = {'__file__': str(source_path), '__name__': 'grid55b_audit_power_v1'}
try:
    exec(compile(modified, str(source_path), 'exec'), namespace)
    namespace['main']()
except BaseException:
    (OUT / 'audit_v1_failure.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
    raise
