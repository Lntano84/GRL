"""Publish audited inputs and correct report-only initialization count."""
from pathlib import Path
import json,hashlib,shutil
import finish
ROOT=finish.ROOT;OUT=finish.OUT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    finish.main()
    p=OUT/'GRID98_report.md';before=p.read_text(encoding='utf8');old='仅构造一次已有Grid2Op向量解码模板，没有调用reset、step或simulate'
    assert before.count(old)==1
    after=before.replace(old,'每次仅构造已有Grid2Op向量解码模板；包括保存故障后的重提取，共初始化两次，没有调用reset、step或simulate')
    after+='\n## 工程记录\n\n第一次提取已完成全部56例隔离检查，在保存静态模板的NumPy整数时停止。冻结源码未改，只通过recovery_01.py补上JSON类型转换后重提取。两次生成的56份可见输入文件逐字节相同；原错误、第一次输入压缩包和恢复收据均保留。首次计时随进程退出丢失，没有编造或混入；extraction_timing.json仅记录成功重提取的实测时间。\n'
    a=Path(json.loads((OUT/'complete.json').read_text(encoding='utf8'))['archive']);c=Path(json.loads((OUT/'complete.json').read_text(encoding='utf8'))['collaboration'])
    original=OUT/'interrupted_attempts/report_draft_before_count_correction.md';original.write_text(before,encoding='utf8')
    receipt=dict(change='Report-only correction to count two vector template initializations including serialization retry; no feature, label, or extractor changes.',before_sha256=sha(original),after_sha256=hashlib.sha256(after.encode('utf8')).hexdigest(),source_sha256=sha(Path(__file__)))
    p.write_text(after,encoding='utf8');(OUT/'report_count_correction.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for src in OUT.rglob('*'):
        if src.is_file() and src.name not in ['complete.json','final_verification.json']:
            dest=a/src.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for name in ['GRID98_report.md','report_count_correction.json','recovery_01.json']:shutil.copy2(OUT/name,c/name)
    d=json.loads((OUT/'complete.json').read_text(encoding='utf8'));d['archive_sha256']={f.relative_to(a).as_posix():sha(f) for f in a.rglob('*') if f.is_file() and f.name!='complete.json'};d['report_sha256']=sha(p)
    (OUT/'complete.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');shutil.copy2(OUT/'complete.json',a/'complete.json');shutil.copy2(OUT/'complete.json',c/'complete.json')
if __name__=='__main__':main()
