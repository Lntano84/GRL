"""Extract only frozen primary episodes with checked paths and resumable file hashes."""
import hashlib,json,os,tarfile,time
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid52';DEST=ROOT/'work/grid03/formal_env_initialized';ARCHIVE=ROOT/'work/grid03/data/l2rpn_idf_2023.tar.bz2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=read(OUT/'design.json');assert not (OUT/'extraction_complete.json').exists()
for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
chosen=set(D['weeks']); index=read(ROOT/'outputs/grid03/archive_index.json')
expected={r['name']:r['size'] for r in index['entries'] if r['file'] and len((p:=PurePosixPath(r['name'])).parts)>=4 and p.parts[1]=='chronics' and p.parts[2] in chosen}
assert len(expected)>0;assert {PurePosixPath(n).parts[2] for n in expected}==chosen
stamp=time.perf_counter();records=[];total=0;base=DEST.resolve()
with tarfile.open(ARCHIVE,'r|bz2') as tf:
    for m in tf:
        p=PurePosixPath(m.name)
        assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='l2rpn_idf_2023'
        if m.name not in expected:continue
        assert m.isfile() and m.size==expected[m.name] and m.size<=128*1024**2
        dst=(DEST/Path(*p.parts[1:])).resolve();assert dst.is_relative_to(base)
        dst.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();written=0
        # Existing complete files may be reused only after archive-byte equality.
        if dst.exists():
            assert dst.stat().st_size==m.size
            with tf.extractfile(m) as src:
                for b in iter(lambda:src.read(1024**2),b''):h.update(b);written+=len(b)
            assert sha(dst)==h.hexdigest()
        else:
            temp=dst.with_suffix(dst.suffix+'.grid52.partial');assert not temp.exists()
            with tf.extractfile(m) as src,temp.open('xb') as f:
                for b in iter(lambda:src.read(1024**2),b''):f.write(b);h.update(b);written+=len(b)
                f.flush();os.fsync(f.fileno())
            assert written==m.size;os.replace(temp,dst)
        total+=written;assert total<=1024**3
        records.append({'path':dst.relative_to(ROOT).as_posix(),'bytes':written,'sha256':h.hexdigest()})
        if len(records)%24==0:print(json.dumps({'files':len(records),'wall_s':time.perf_counter()-stamp}),flush=True)
assert len(records)==len(expected) and total==sum(expected.values())
R={'passed':True,'files':records,'payload_bytes':total,'wall_s':time.perf_counter()-stamp,
   'source_archive_sha256_from_qualified_index':index['archive_sha256'],'scope':'Frozen8primary episodes only. No agent/score/feature extraction during acquisition.'}
(OUT/'extraction_complete.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('EXTRACTED FROZEN8PRIMARY EPISODES:96serial confirmation runs pending.\n',encoding='utf-8')
print(json.dumps({'passed':True,'files':len(records),'payload_bytes':total,'wall_s':R['wall_s']}))
