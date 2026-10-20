"""Freeze analysis code and provenance before any classifier fit."""
from pathlib import Path
import shutil,datetime
from data import ROOT,W,OUT,read,save,sha,prepare


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'fits.json').exists() and not (OUT/'code_freeze.json').exists()
    shutil.copy2(W/'design.json',OUT/'design.json')
    prepare()
    pre=read(OUT/'preflight.json')
    files=dict(pre['sources'])
    for p in W.glob('*.py'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [W/'design.json',OUT/'dataset.json',OUT/'dataset.csv',OUT/'preflight.json',OUT/'data_summary.json']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        frozen_before_fitting=True,no_new_simulations=True))
    repo=ROOT/'work/GRL-results-integration'
    dst=repo/'research_archive/power-grid-2026/protocols/GRID94_predictability/frozen';dst.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[W/'design.json']+[OUT/n for n in ['code_freeze.json','preflight.json','data_summary.json','dataset.csv','dataset.json']]:
        shutil.copy2(p,dst/p.name);assert sha(p)==sha(dst/p.name)
    key='GRID94_SKIP_PREDICTABILITY_frozen'
    note='按用户授权先检验何时不启动补查。标签仅为原补查前后返回同一代表动作，不把动作变化标为长期有益，也不把整周反例贴给每次调用。固定15项补查前可见特征、逻辑回归和深度2树，按日历日期留组及old13到G1/G2检验；0.95主阈值、0.5诊断全部保留，不调参或部署，无新仿真/RL。'
    for n in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
        p=repo/n;s=p.read_text(encoding='utf-8');p.write_text(s.rstrip()+'\n\n## 2026-10-20 — '+key+'\n\n'+note+'\n',encoding='utf-8')
    print('FROZEN',len(files),'files; no classifier fit yet')


if __name__=='__main__':main()
