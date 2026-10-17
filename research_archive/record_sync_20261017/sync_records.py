"""Content-verified snapshot of research records, including failed attempts and logs."""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import subprocess, hashlib, json, zipfile, gzip, os, shutil

ROOT=Path('D:/Codex/GRL研究').resolve()
REPO=ROOT/'work/GRL-results-integration'
DEST=REPO/'research_archive/record_sync_20261017'
LOCAL=ROOT/'outputs/research_records_sync_20261017'
GIT=Path('C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe')
assert not (DEST/'sync_summary.json').exists(),'Completed snapshot exists; do not duplicate.'
DEST.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
SKIP={'official','upstream','node_modules','site-packages','generator_packages','render_dependency',
      '.venv','venv','__pycache__','.git','qa','中文写作范例'}
TEXT={'.json','.md','.csv','.yaml','.yml','.toml','.txt','.log','.bib','.tex','.py','.ps1','.cmd','.sh','.ini'}
WORK={'.py','.ps1','.cmd','.sh','.json','.md','.yaml','.yml','.toml','.tex','.bib'}
authored_assets={'.docx','.xlsx','.svg','.png','.drawio','.pdf'}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def wanted(p,scope):
    rel=p.relative_to(ROOT)
    if any(x in SKIP for x in rel.parts):return False
    if scope=='collaboration':
        if 'sources' in rel.parts and p.suffix.lower() in {'.pdf','.html','.txt'}:return False
        return True
    if p.suffix.lower() in TEXT:return True
    if p.name.endswith('.jsonl.gz'):return True
    if p.suffix.lower()=='.docx':return True
    if p.suffix.lower() in authored_assets and not any(x in rel.parts for x in ('sources','reading','参考材料')):
        return True
    return False

candidates=[];excluded=Counter()
for directory,scope in [(ROOT/'outputs','outputs'),(ROOT/'论文协作资料','collaboration')]:
    for p in directory.rglob('*'):
        if not p.is_file():continue
        if p.is_relative_to(LOCAL) or p.is_relative_to(ROOT/'outputs/frame_cn_merged_20261017_v1/qa'):continue
        if wanted(p,scope):candidates.append(p)
        else:excluded[p.suffix.lower() or '[no extension]']+=1
# Preserve task-owned entry points; vendor code and environments stay outside this snapshot.
for stage in (ROOT/'work').iterdir():
    if not stage.is_dir() or stage==REPO:continue
    if stage.name.startswith(('grid','fa','frame','icaps','lg')):
        candidates.extend(p for p in stage.iterdir() if p.is_file() and p.suffix.lower() in WORK)
candidates=sorted(set(p.resolve() for p in candidates))
assert all(p.is_relative_to(ROOT) for p in candidates)

tracked=subprocess.check_output([str(GIT),'-C',str(REPO),'ls-files','-z']).decode('utf-8').split('\0')
repo_files=[REPO/x for x in tracked if x and (REPO/x).is_file() and (REPO/x).stat().st_size<100*1024*1024]
sizes={p.stat().st_size for p in candidates}
repo_files=[p for p in repo_files if p.stat().st_size in sizes]
lookup={}
with ThreadPoolExecutor(max_workers=4) as pool:
    for p,h in zip(repo_files,pool.map(digest,repo_files)):
        lookup.setdefault((p.stat().st_size,h),p.relative_to(REPO).as_posix())

index=[];groups=defaultdict(list);newbytes=0;existing=0
with ThreadPoolExecutor(max_workers=4) as pool:
    hashes=list(pool.map(digest,candidates))
for p,h in zip(candidates,hashes):
    n=p.stat().st_size;rel=p.relative_to(ROOT).as_posix()
    item={'source':rel,'bytes':n,'sha256':h}
    if (n,h) in lookup:
        item.update(status='existing_identical',repository_path=lookup[(n,h)]);existing+=1
    else:
        parts=p.relative_to(ROOT).parts
        stage=('collaboration/'+parts[1]) if parts[0]=='论文协作资料' else '/'.join(parts[:2])
        groups[stage].append((p,item));newbytes+=n
    index.append(item)

archives=[];count=0
for stage,items in sorted(groups.items()):
    batches=[];current=[];size=0
    for p,item in items:
        if current and size+item['bytes']>32*1024*1024:
            batches.append(current);current=[];size=0
        current.append((p,item));size+=item['bytes']
    if current:batches.append(current)
    for batchno,batch in enumerate(batches,1):
        out=DEST/'records'/stage/f'part-{batchno:03d}.zip'
        out.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p,item in batch:
                data=p.read_bytes()
                assert hashlib.sha256(data).hexdigest()==item['sha256'],('Changed during snapshot',str(p))
                mode=zipfile.ZIP_STORED if p.suffix.lower() in {'.gz','.docx','.xlsx','.png','.pdf'} else zipfile.ZIP_DEFLATED
                z.writestr(item['source'],data,compress_type=mode)
                item.update(status='new_archive',repository_path=out.relative_to(REPO).as_posix(),member=item['source'])
        assert out.stat().st_size<95*1024*1024,('Archive exceeds per-file admission',out)
        # CRC plus original SHA-256 verification, without rerunning any experiment.
        with zipfile.ZipFile(out) as z:
            assert z.testzip() is None
            for _,item in batch:
                assert hashlib.sha256(z.read(item['source'])).hexdigest()==item['sha256']
        archives.append({'path':out.relative_to(REPO).as_posix(),'files':len(batch),
                         'bytes':out.stat().st_size,'sha256':digest(out)})
        count+=len(batch)
    print(json.dumps({'stage':stage,'new_files':len(items),'completed_new_files':count},ensure_ascii=False),flush=True)

with gzip.open(DEST/'records_index.jsonl.gz','wt',encoding='utf-8') as f:
    for item in index:f.write(json.dumps(item,ensure_ascii=False)+'\n')
summary={
 'snapshot_utc':datetime.now(timezone.utc).isoformat(),'source_root':str(ROOT),
 'source_files':len(index),'existing_identical':existing,'newly_archived':count,
 'all_indexed_files_accounted_for':all(x.get('status') in ('existing_identical','new_archive') for x in index),
 'original_record_bytes':sum(x['bytes'] for x in index),'new_original_bytes':newbytes,
 'archive_bytes':sum(x['bytes'] for x in archives),'archives':archives,
 'excluded_file_counts_by_extension':dict(excluded),
 'scope':'Research outputs and per-query JSON, per-step JSONL.GZ, audits, failures and logs, manuscript assets, collaboration materials, task-owned source entry points; excludes environments, vendor/source-paper reading files, large binary vectors/checkpoints and raw input datasets.',
 'all_new_archive_members_crc_and_sha256_verified':True,'experiment_reruns':0}
assert summary['all_indexed_files_accounted_for'] and existing+count==len(index)
(DEST/'sync_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copy2(DEST/'sync_summary.json',LOCAL/'sync_summary.json')
shutil.copy2(DEST/'records_index.jsonl.gz',LOCAL/'records_index.jsonl.gz')
(DEST/'README.md').write_text(
 '# 研究记录补齐归档 2026年10月17日\n\n'
 '本目录补齐当前D盘研究工作区尚未归档的研究记录。索引逐项保存原始路径、字节数、SHA256及仓库位置。'
 '已经与仓库文件逐字节相同的记录引用原有归档；新增记录按实验批次压缩，包含正常结果、较差结果、工程失败、恢复日志和逐次查询/逐步记录。\n\n'
 '恢复新记录：在一个空目录中解压records下的ZIP，内部路径与原工作区一致。原始文件没有改写；各成员已经校验CRC和SHA256。'
 'records_index.jsonl.gz覆盖所有本次纳入的记录；sync_summary.json给出数量、大小和各压缩包校验值。\n\n'
 '范围：outputs研究文本/JSON/CSV/日志/逐步JSONL.GZ、论文和图源、论文协作资料，以及work下各任务自有入口源码。'
 'Python环境、第三方源码/下载的原论文、大型状态向量、模型二进制和原始输入数据保留本地；此前已有模型/输入哈希和来源记录继续保留。'
 '这次是归档补齐，不是新的实验或再次独立物理验证。\n',encoding='utf-8')
shutil.copy2(Path(__file__),DEST/'sync_records.py')
print(json.dumps({k:v for k,v in summary.items() if k not in ['archives','excluded_file_counts_by_extension']},ensure_ascii=False,indent=2),flush=True)
