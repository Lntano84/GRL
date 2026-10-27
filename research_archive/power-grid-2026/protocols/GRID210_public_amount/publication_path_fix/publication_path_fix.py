"""Repair only publication pointers; retain frozen scientific sources and error."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def main():
    C.trained();assert not (C.O/'publication_path_fix.json').exists() and not (C.O/'delivery_receipt.json').exists();old=C.A/'audited/audit_delivery.py';assert old.exists();frozen_before=C.sha(C.O/'trained_freeze.json');p=Path(__file__).parent/'audit_delivery.py';text=p.read_text(encoding='utf8');needle="('grid210_amount_depth','GRID210_amount_depth','FRAME_收益深度对照_GRID210')";assert text.count(needle)==1;p.write_text(text.replace(needle,"('grid210_amount_depth',C.A.name,'FRAME_收益深度对照_GRID210')"),encoding='utf8',newline='\n');r=C.REPO/'research_archive/power-grid-2026/README.md';text=r.read_text(encoding='utf8');wrong='protocols/GRID210_amount_depth/';right='protocols/'+C.A.name+'/';assert wrong in text;r.write_text(text.replace(wrong,right),encoding='utf8',newline='\n')
    C.save('publication_error_01.json',dict(kind='publication_path_mismatch',error='AssertionError in audit_delivery.delivery: no matching archive bytes under nonexistent GRID210_amount_depth.',actual_existing_archive=C.A.name,original_delivery_checker_sha256=C.sha(old),scientific_results_failure=False,old_frozen_sources_and_archives_preserved=True))
    C.trained();assert frozen_before==C.sha(C.O/'trained_freeze.json');C.save('publication_path_fix.json',dict(passed=True,fix='Currentstage archive path resolved from frozen run.C.A.name;README links corrected. No directory move,training/evaluation rerun,or frozenrun edit.',actual_archive=C.A.as_posix(),original_checker_preserved_at=old.as_posix(),trained_manifest_sha256_before=frozen_before,trained_manifest_sha256_after=C.sha(C.O/'trained_freeze.json'),new_fits=0,new_predictions=0,new_physics=0));C.archive('publication_path_fix');dest=C.R/'论文协作资料/FRAME_收益深度对照_GRID210'
    for n in ['publication_error_01.json','publication_path_fix.json']:shutil.copy2(C.O/n,dest/n)
    msg='GRID210交付核验曾因汇总检查器使用GRID210_amount_depth而失败，实际封存目录为冻结源码指定的GRID210_public_amount。仅修复检查器和README链接，保留原错误/检查器/全部冻结源码，权重清单哈希前后完全相同；未训练、重评或移动目录。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID210发布路径核验修复\n\n'+msg+'\n',encoding='utf8',newline='\n')
    print('GRID210_PUBLICATION_POINTERS_REPAIRED',C.A.name)
if __name__=='__main__':main()
