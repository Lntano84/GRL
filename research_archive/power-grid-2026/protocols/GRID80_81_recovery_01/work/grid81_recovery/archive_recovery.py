"""Preserve the interrupted generation receipt without changing any frozen source."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
def main():
    dest=ARCH/'protocols/GRID80_81_recovery_01';assert not dest.exists()
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'outputs/grid81/recovery_01.json',ROOT/'outputs/grid81/pipeline_error.json',ROOT/'outputs/grid81/pipeline_state.json']
    entries=[]
    for p in files:
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==h;entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=h,bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,scope='Engineering-only recovery; completed January generated data retained byte-for-byte, original exact seed stream used for five missing dates; zero closed-loop runs repeated.'),indent=2),encoding='utf-8')
    note='GRID80原生成进程无Python异常记录地结束，具体原因未定，GRID81 pipeline因缺完整六场景receipt安全停止，尚未有闭环回放。已逐项核对全部冻结哈希、原始主种子还原的逐日期三种子，以及已完成2036-01-07_9的完整文件/数值/元数据；原场景保留13个文件SHA-256，不重做。续跑脚本直接使用官方add_data所调用的相同wrapper与全部原参数，只生成剩余五日期，固定场景ID9，避免再次调用自动递增ID的add_data。原错误与不完整计算耗时保留，不解释为方法失败、不改模型/阈值/日期/种子。生成进程改为持久隐藏启动，后续仍由原冻结pipeline接管。'
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-16 — GRID80 safe generation continuation\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **运行恢复：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    print(json.dumps(dict(files=len(entries),completed_generated_dates_retained=1,closed_loop_runs_repeated=0)))
if __name__=='__main__':main()
