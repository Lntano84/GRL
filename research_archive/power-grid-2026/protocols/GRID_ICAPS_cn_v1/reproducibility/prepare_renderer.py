"""Task-local official rendering dependency; administrative extraction, no desktop install."""
from pathlib import Path
import hashlib,json,time,urllib.request,subprocess
HERE=Path(__file__).resolve().parent
DEP=(HERE/'render_dependency').resolve();DEP.mkdir(parents=True,exist_ok=True)
assert DEP.is_relative_to(HERE.resolve())
URL='https://download.documentfoundation.org/libreoffice/stable/26.8.1/win/x86_64/LibreOffice_26.8.1_Win_x86-64.msi'
EXPECTED='c6298b10cfa1748bcbedad94c62ce771e240e5021bfd6d7a967cd571b04b5312'
MSI=DEP/'LibreOffice_26.8.1_Win_x86-64.msi'
DEST=(DEP/'extracted').resolve()
assert DEST.is_relative_to(DEP)
if not MSI.exists():
    req=urllib.request.Request(URL,headers={'User-Agent':'GRL-document-render/1.0'})
    start=time.monotonic();n=0
    with urllib.request.urlopen(req,timeout=40) as r, MSI.with_suffix('.part').open('wb') as f:
        final=r.geturl()
        while chunk:=r.read(1024*1024):
            f.write(chunk);n+=len(chunk)
            if time.monotonic()-start>240:raise TimeoutError('Rendering dependency download cap')
        print(json.dumps({'downloaded':n,'final_url':final}),flush=True)
    MSI.with_suffix('.part').replace(MSI)
h=hashlib.sha256(MSI.read_bytes()).hexdigest()
assert h==EXPECTED,('Official checksum mismatch',h)
existing=list(DEST.rglob('soffice.exe')) if DEST.exists() else []
if not existing:
    result=subprocess.run(['msiexec.exe','/a',str(MSI),'/qn','TARGETDIR='+str(DEST),'/L*v',str(DEP/'extract.log')],timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,('MSI extraction failed',result.returncode)
    existing=list(DEST.rglob('soffice.exe'))
assert len(existing)==1,existing
receipt=dict(official_url=URL,sha256=h,bytes=MSI.stat().st_size,renderer=str(existing[0]),scope='Task-local administrative extraction only. No use of user desktop LibreOffice, no installation or file-association change.',reference='https://download.documentfoundation.org/libreoffice/stable/26.8.1/win/x86_64/LibreOffice_26.8.1_Win_x86-64.msi.mirrorlist')
(DEP/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
