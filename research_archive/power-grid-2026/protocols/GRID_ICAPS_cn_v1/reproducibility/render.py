"""Run the packaged DOCX renderer with an isolated task-local official dependency."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid_icaps_cn_v1'
deps=Path('C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies')
helper=Path('C:/Users/windows/.codex/plugins/cache/openai-primary-runtime/documents/26.1011.11113/skills/documents/render_docx.py')
rec=json.loads((Path(__file__).resolve().parent/'render_dependency/receipt.json').read_text())
exe=Path(rec['renderer'])
assert exe.resolve().is_relative_to(Path(__file__).resolve().parent)
env=dict(os.environ);env['PATH']=str(exe.parent)+os.pathsep+str(deps/'native/poppler/Library/bin')+os.pathsep+env.get('PATH','')
env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
cmd=[str(deps/'python/python.exe'),str(helper),str(OUT/'ICAPS2027_中文完整初稿.docx'),'--output_dir',str(OUT/'qa'),'--emit_pdf','--dpi','105','--verbose']
with (OUT/'render.log').open('w',encoding='utf-8') as f:
    r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
print((OUT/'render.log').read_text(encoding='utf-8')[-5000:])
if r.returncode:raise SystemExit(r.returncode)
