from pathlib import Path
import hashlib,json,collections
ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    groups=[]
    for stage in [70,81]:
        folders=sorted((ROOT/f'outputs/grid{stage}/runs').glob('r0__*__FRAME_GUARD'))
        calls=[p for f in folders for p in sorted(f.glob('call[0-9][0-9][0-9].json'))]
        groups.append(dict(stage=stage,dates=len(folders),calls=len(calls),phases=dict(collections.Counter(r['phase'] for p in calls for r in read(p)['outcomes'])),
                           first_call_keys=list(read(calls[0])) if calls else []))
    for stage in [65,68,81]:
        fs=sorted((ROOT/f'outputs/grid{stage}/runs').glob('r0__*__FULL'))
        print(json.dumps(dict(stage=stage,full_dates=[f.name.split('__')[1] for f in fs],full_calls=sum(len(list(f.glob('call[0-9][0-9][0-9].json'))) for f in fs)),ensure_ascii=False))
    print(json.dumps(groups,ensure_ascii=False))
if __name__=='__main__':main()
