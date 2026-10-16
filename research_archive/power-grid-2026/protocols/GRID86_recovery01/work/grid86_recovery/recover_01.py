"""Add required timing-description metadata before the first decision loop."""
from pathlib import Path
import ast,datetime as dt,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid86';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'recovery_01.json').exists() and not (OUT/'RUN.lock').exists() and not (OUT/'pipeline.lock').exists()
    assert not (OUT/'manifest.json').exists() and not list(OUT.glob('worker_result_*.json'))
    partial=OUT/'runs/r0__2036-01-07_9__FRAME_GUARD'
    assert partial.is_dir() and not any(partial.iterdir())
    assert "KeyError: 'timing'" in (OUT/'worker_00.log').read_text(encoding='utf-8')
    assert read(OUT/'runner_error.json')['runs']==0
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/name).items():assert sha(ROOT/p)==h,p
    original=OUT/'interrupted_attempts/attempt01';original.mkdir(parents=True,exist_ok=False)
    for name in ['code_freeze.json','design.json','pipeline_error.json','runner_error.json','worker_00.log','pipeline_state.json','pipeline_stderr.log','run_pipeline.log','RUN_STATE.md']:
        if (OUT/name).exists():shutil.copy2(OUT/name,original/name)
    source=ROOT/'work/grid86/prepare.py';shutil.copy2(source,original/'prepare.py')
    before=read(OUT/'design.json');after=dict(before,timing=read(ROOT/'outputs/grid82/design.json')['timing'])
    assert {k:v for k,v in after.items() if k!='timing'}==before
    (OUT/'design.json').write_text(json.dumps(after,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    text=source.read_text(encoding='utf-8');needle="quality_threshold=old['quality_threshold'],**{k:old[k] for k in ['full_reference']},"
    assert text.count(needle)==1
    source.write_text(text.replace(needle,"quality_threshold=old['quality_threshold'],timing=old['timing'],**{k:old[k] for k in ['full_reference']},"),encoding='utf-8')
    frozen=read(OUT/'code_freeze.json')
    for p in [source,OUT/'design.json']:frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    (OUT/'code_freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    # This folder is verified empty and both paths are within the intended output root.
    for p in [partial,original]:assert p.resolve().is_relative_to(OUT.resolve())
    partial.replace(original/'empty_episode_directory')
    for name in ['worker_00.log','runner_error.json','pipeline_error.json']:
        assert sha(OUT/name)==sha(original/name);(OUT/name).unlink()
    # Check the worker's required top-level design schema without running control.
    tree=ast.parse((ROOT/'work/grid55/worker.py').read_text(encoding='utf-8'))
    keys={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='design' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)}
    assert keys<=set(after),keys-set(after)
    receipt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),reason='Required descriptive design.timing field missing; failed during metadata construction before the first decision/step.',
                 completed_items=0,physical_decision_steps=0,only_added_design_field='timing',scientific_protocol_unchanged=True,
                 worker_required_design_keys=sorted(keys),empty_partial_preserved=True,new_training=0,
                 original_design_sha256=sha(original/'design.json'),new_design_sha256=sha(OUT/'design.json'),
                 original_code_freeze_sha256=sha(original/'code_freeze.json'),new_code_freeze_sha256=sha(OUT/'code_freeze.json'))
    (OUT/'recovery_01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    dest=REPO/'research_archive/power-grid-2026/protocols/GRID86_recovery01';assert not dest.exists()
    items=[OUT/'recovery_01.json',OUT/'design.json',OUT/'code_freeze.json',source,Path(__file__).resolve()]+[p for p in original.iterdir() if p.is_file()]
    for p in items:
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(target)==sha(p)
    note='GRID86首worker在元数据构造时因缺少descriptive timing字段退出，尚未进入首个控制步，完成项0且回放目录为空。原日志/空目录/设计/源码/冻结清单完整保留；只补入与GRID82相同的计时说明字符串，科学方案与控制逻辑不变。核对worker设计必需键后按原方案续跑。'
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-16 — GRID86 recovery01\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **工程恢复：** '+note+'\n'+''.join(lines[1:]),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
