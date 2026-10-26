"""Verify archived originals, or restore all logical paths into a NEW D: folder."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--destination',type=Path);args=ap.parse_args();root=args.archive.resolve();m=json.loads((root/'records_manifest.json').read_text(encoding='utf8'))
    if args.destination:
        dest=args.destination.resolve();assert dest.drive.lower()=='d:','Keep project materials on D:';dest.mkdir(parents=True,exist_ok=False)
    checked={}
    with zipfile.ZipFile(root/'small_files.zip') as z:
        for rel,row in m['files'].items():
            name=PurePosixPath(rel);assert not name.is_absolute() and '..' not in name.parts
            where=row['location'];h=row['sha256'];iszip=where.startswith('small_files.zip!')
            if h not in checked:
                data=z.read(h) if iszip else (root/where).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==h,(rel,'Run git lfs pull if blob is a pointer.');checked[h]=True
            if args.destination:
                target=dest.joinpath(*name.parts);assert target.resolve().is_relative_to(dest);target.parent.mkdir(parents=True,exist_ok=True)
                if iszip:target.write_bytes(z.read(h))
                else:
                    import shutil
                    shutil.copyfile(root/where,target)
                assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    print(json.dumps(dict(passed=True,logical_files=len(m['files']),unique_objects=len(checked),restored=bool(args.destination))))
if __name__=='__main__':main()
