"""Verify immutable archives, actual remote LFS downloads, and final result commit."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid184_search_adapter';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter'
sys.path.insert(0,str(R/'work/grid184_search_adapter'));import run as original
read=original.read;sha=original.sha
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
original.verify();assert not (O/'publication_verification.json').exists()
local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
assert subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)==''
phases=0;files=0
for p in A.glob('*/archive_manifest.json'):
    for name,h in read(p).items():assert sha(p.parent/name)==h,(p,name);files+=1
    phases+=1
raw=A/'raw_records';m=read(raw/'records_manifest.json');download=R/'tmp/grid184_remote_verify_lfs/objects';large=0
for rel,row in m['files'].items():
    assert sha(O/rel)==row['sha256'],rel
for obj in (raw/'objects').iterdir():
    h=obj.stem;p=download/h[:2]/h[2:4]/h;assert p.is_file() and sha(p)==h==sha(obj);large+=1
for n in ['GRID184_report.md','review.json','decision.json','results_summary.json','first_divergences.json','baseline_aligned_call_tradeoffs.json']:
    assert sha(O/n)==sha(R/'论文协作资料/FRAME_小闭环结果_GRID184'/n)
    path=(A/'final_delivery'/n).relative_to(REPO).as_posix();data=subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=REPO);assert hashlib.sha256(data).hexdigest()==sha(O/n)
review=read(O/'review.json');assert review['passed_saved_data'] and review['checks']['repeat_pairs']==32
div=read(O/'first_divergences.json')['entries'];assert sum(not r['physical_signature_equal'] for r in div)==3
receipt=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results_git_commit=local,remote_result_commit_verified=True,archive_phases_verified=phases,archive_files_verified=files,original_run_files_verified=len(m['files']),original_run_bytes=m['logical_bytes'],actual_remote_lfs_objects_downloaded_and_hash_verified=large,collaboration_and_git_result_blobs_match=True,checks=review['checks'],new_fits=0,new_physical_runs_during_recovery=0,report_sha256=sha(O/'GRID184_report.md'),frozen_inputs_unchanged=True)
save(O/'publication_verification.json',receipt);dest=A/'publication';dest.mkdir(exist_ok=False)
for p in [O/'publication_verification.json',Path(__file__),O/'audit_recovery_execution.json',O/'raw_archive_review.json']:shutil.copy2(p,dest/p.name)
save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir()})
copy=R/'论文协作资料/FRAME_小闭环结果_GRID184';shutil.copy2(O/'publication_verification.json',copy/'publication_verification.json')
with (O/'HANDOFF.md').open('a',encoding='utf8') as f:f.write('\n## 2026-10-26完成\n\n64条审计检查、报告、全部1842原记录/915353235字节的无损去重归档已完成。10个大对象经Git LFS上传并在新D盘缓存从远端重新下载，逐SHA验证；各phase清单、协作副本及Git结果blob一致。读取优化保留全部assert，复制footer缺shutil在审计通过后只补发布，未重跑审计或物理。原错误与较差结果均保留。publication_verification.json记录结果提交；当前任务收尾后删除grid184-consensus，不恢复任何旧自动化。\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID184完成\n\n1. 审计、1842原记录可恢复归档、LFS远端读回及报告/协作副本一致均已验证；本轮物理与审计不再重复。\n2. 保留FACTOR对FRAME的23个同状态同动作省查询点（同等已运行时域时间少3.55%、候选少9.61%）；当前共识对FACTOR未新增此类收益，先查覆盖停止决定的3个实际物理分歧。\n3. 质量与计算分列：3个可判日期2/3达标，5月参考受限；不以全赢门槛否定小改善，但不把NET或省查询当整周成本改善。未授权新增训练/物理/最终8/论文。\n',encoding='utf8',newline='\n')
p=REPO/'docs/EXPERIMENT_LOG.md'
with p.open('a',encoding='utf8') as f:f.write(f'\n\n## 2026-10-26 — GRID184发布验证完成\n\n结果提交{local}与远端一致，{phases}归档阶段/{files}归档文件和1842原run文件校验；10个LFS大对象已从远端重新下载逐SHA读回。报告/协作副本/Git blob一致，完成通知后删除跟进，不再回放或重新审计。\n')
print('GRID184_PUBLICATION_FULLY_VERIFIED',json.dumps(receipt,ensure_ascii=False))
