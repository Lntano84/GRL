"""Append-only new-scenario and gate-ablation protocol/results publication."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['before','qualified','complete']);a=ap.parse_args()
    dest=ARCH/'protocols'/('GRID80_81_'+a.phase);assert not dest.exists()
    assert read(ROOT/'outputs/grid81/preflight.json')['passed']
    if a.phase=='complete':assert read(ROOT/'outputs/grid81/review.json')['passed_saved_data']
    if a.phase=='qualified':assert read(ROOT/'outputs/grid80/qualification.json')['passed']
    files=list((ROOT/'work/grid80').glob('*.py'))+list((ROOT/'work/grid81').glob('*.py'))+[Path(__file__).resolve()]
    for stage in [80,81]:
        folder=ROOT/f'outputs/grid{stage}'
        files.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv'])
        files.extend((folder/'engineering_errors').glob('*.txt'))
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,scope='Six prospectively fixed official-generator 2036 scenarios; five unchanged or calibrated-rate arms; no training, no resampling; full trajectories and third-party packages remain in D workspace.'),ensure_ascii=False,indent=2),encoding='utf-8')
    note='按用户一二优先：现有2035官方52日期家族全部已暴露，改为预先锁定2036交替月份首个周一六场景，用官方Chronix2Grid、固定生成种子20261016和独立依赖目录。GRID81比较FULL、当前门控、始终核查和随机门控0/1，随机范围内跳过概率4/13只用旧校准集确定，范围外均委托核查；两次反向计时共60条。755项既有训练/校准付费反馈检查通过；不训练、不改模型/阈值、不加入SoftGNN新对照。'
    if a.phase=='complete':
        d=read(ROOT/'outputs/grid81/decision.json');note+=' 已完成保存数据审计。新场景联合筛查='+str(d['fresh_confirmation_screen'])+'，两随机种子识别增量筛查='+str(d['targeting_screen_both_random_seeds'])+'；全部较差结果和失败保留。'
    elif a.phase=='qualified':note+=' 六个生成场景及全部输入哈希已完成资格检查，即将启动串行闭环；没有按控制结局筛选场景。'
    else:note+=' 方案冻结，先通过生成数据资格再执行闭环；生成工程失败不解释为方法失败。'
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-16 — GRID80/81 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+('阅读GRID81_report.md，按完整新场景与随机门控结果组织贡献论证；不按中间结果选日期或种子。' if a.phase=='complete' else '完成GRID80官方时序生成/资格检查；按冻结GRID81执行60条串行回放，运行期间不训练、安装或进行重型审计；所有已完成条目不重跑。')+'\n2. 数据与运行错误保留；六生成场景属于相同电网的新时序，两次计时不算独立场景。\n3. 论文先中文，用户手写稿不覆盖；不自动扩大基线/模型/参数。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID80/81 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps(dict(phase=a.phase,files=len(entries))))
if __name__=='__main__':main()
