"""Guard before any model/environment import. Physical receipt must be published."""
from pathlib import Path
import json,hashlib,subprocess
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid291_control_preparation';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID291_control_preparation'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_receipt(auth,design_hash,freeze_hash):
 if auth.get('approved_by_user') is not True or not auth.get('user_instruction') or not auth.get('user_message_reference'):raise PermissionError('explicit scoped user approval missing')
 if auth.get('design_sha256')!=design_hash or auth.get('source_freeze_sha256')!=freeze_hash:raise PermissionError('frozen scope differs')
 if auth.get('runs')!=192 or auth.get('wall_cap_s')!=14400 or auth.get('worker_cap_s')!=1200:raise PermissionError('physical budget differs')
def require_authorization(*,remote=False):
 p=O/'physical_authorization.json'
 if not p.exists():raise PermissionError('physical authorization receipt missing; no models or environment loaded')
 auth=read(p);validate_receipt(auth,sha(O/'design.json'),sha(O/'source_freeze.json'))
 def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
 head=git('rev-parse','HEAD').decode().strip()
 if git('show',head+':'+(A/'authorization/physical_authorization.json').relative_to(REPO).as_posix())!=p.read_bytes():raise PermissionError('authorization Git blob differs')
 if remote and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head:raise PermissionError('authorization not remotely published')
 if git('show',head+':'+(A/'source_launch_recovery02/source_freeze.json').relative_to(REPO).as_posix())!=(O/'source_freeze.json').read_bytes():raise PermissionError('source freeze Git blob differs')
 return auth

def verify_frozen(*,data=False):
 for rel,h in read(O/'source_freeze.json').items():
  if sha(R/rel)!=h:raise PermissionError('frozen source/model changed: '+rel)
 if data:
  for rel,h in read(O/'data_freeze.json').items():
   if sha(R/rel)!=h:raise PermissionError('frozen data changed: '+rel)


def manager_ancestor_matches(expected_pid):
 """Allow only direct parent or one Windows python venv redirector."""
 import os,ctypes
 parent=os.getppid()
 if parent==expected_pid:return True
 if os.name!='nt':return False
 class Entry(ctypes.Structure):
  _fields_=[('dwSize',ctypes.c_ulong),('cntUsage',ctypes.c_ulong),('th32ProcessID',ctypes.c_ulong),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',ctypes.c_ulong),('cntThreads',ctypes.c_ulong),('th32ParentProcessID',ctypes.c_ulong),('pcPriClassBase',ctypes.c_long),('dwFlags',ctypes.c_ulong),('szExeFile',ctypes.c_wchar*260)]
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateToolhelp32Snapshot.argtypes=[ctypes.c_ulong,ctypes.c_ulong];k.CreateToolhelp32Snapshot.restype=ctypes.c_void_p;k.Process32FirstW.argtypes=[ctypes.c_void_p,ctypes.POINTER(Entry)];k.Process32NextW.argtypes=[ctypes.c_void_p,ctypes.POINTER(Entry)];k.CloseHandle.argtypes=[ctypes.c_void_p]
 handle=k.CreateToolhelp32Snapshot(2,0)
 if handle in [None,ctypes.c_void_p(-1).value]:return False
 try:
  e=Entry();e.dwSize=ctypes.sizeof(e);ok=k.Process32FirstW(handle,ctypes.byref(e))
  while ok:
   if e.th32ProcessID==parent:return e.th32ParentProcessID==expected_pid and e.szExeFile.lower() in ['python.exe','pythonw.exe']
   ok=k.Process32NextW(handle,ctypes.byref(e))
  return False
 finally:k.CloseHandle(handle)
