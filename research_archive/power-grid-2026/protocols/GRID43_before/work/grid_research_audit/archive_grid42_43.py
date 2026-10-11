"""Preserve the import-only failure, bounded completion probe, and common pilot."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
phase=argparse.ArgumentParser();phase.add_argument('--phase',choices=['before','pilot','complete'],required=True);phase=phase.parse_args().phase
assert read(ROOT/'outputs/grid42b/audit.json')['passed']
if phase=='before':assert not (ROOT/'outputs/grid43/manifest.json').exists()
elif phase=='pilot':assert len(read(ROOT/'outputs/grid43/manifest.json')['summaries'])==6
else:assert read(ROOT/'outputs/grid43/review.json')['passed_saved_data']
old={}
for manifest in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(manifest).get('files',[]):
        p=manifest.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
def archive(label,source,output,raw=False):
    dest=ARCH/'protocols'/label;assert not dest.exists();dest.mkdir(parents=True)
    files=list(source.glob('*.py'))+[Path(__file__).resolve()]
    files += [p for p in output.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv']]
    if raw:
        for folder in (output/'runs').iterdir():
            files += [folder/name for name in ['summary.json','metadata.json','components.json'] if (folder/name).exists()]
    entries=[]
    for p in sorted(set(files)):
        rel=p.relative_to(ROOT);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,q);assert sha(p)==sha(q);entries.append({'path':rel.as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    (dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),
        'scope':'Development observed-reward tie completion,common to every ranker. No new model or final dates. Large raw response/vector records stay local with original manifests.'},indent=2),encoding='utf-8')
    return len(entries)
counts={}
if phase=='before':
    counts['GRID42_import_failure']=archive('GRID42_import_failure',ROOT/'work/grid42',ROOT/'outputs/grid42')
    counts['GRID42B_completed']=archive('GRID42B_completed',ROOT/'work/grid42b',ROOT/'outputs/grid42b')
counts['GRID43_'+phase]=archive('GRID43_'+phase,ROOT/'work/grid43',ROOT/'outputs/grid43',phase!='before')
if phase=='before':
    note='GRID41后续机制检查：SoftGNN三月578步的rho/FP32即时奖励与FULL完全并列但物理动作不同。GRID42导入冲突仅预检失败、保留；GRID42B独立复算130条缓存序列及隐藏反馈扰动通过。一条共同补查：完成已观测最佳奖励并列涉及的变电站，含安全前缀。原GNN0/1、SoftGNN0/1均找回578/1733，代价为查询增加；不是长期质量或图专属贡献。GRID43冻结6条三月试点，质量通过后18条其他开发日期；不重训、不打开最终8日期。'
    nxt='1. 完成GRID43六条三月实际回放和保存质量门槛；通过才执行18条其余开发周。\n2. 所有排序器共享并列补查，原权重/策略/失败结果均保留；不扩大主对照清单。\n3. 最终8日期封存、LG01暂停；继续到确需人工帮助。\n'
elif phase=='pilot':
    A=read(ROOT/'outputs/grid43/pilot_audit.json')
    note='GRID43共同并列补查三月六条试点完成，质量门槛'+('全部通过' if A['all_rankers_qualified'] else '未全部通过')+'；'+ '；'.join(f"{r['method']}成本差{r['cost_regret']:.6f}、查询{r['queries']}" if r['cost_regret'] is not None else f"{r['method']}不可比" for r in A['results'])+'。结果保留，不选最佳种子。'
    nxt='1. '+('按冻结协议执行其余18条开发回放并统一核验。' if A['all_rankers_qualified'] else '保留试点，解释未通过项；不自动执行其余日期。')+'\n2. 原模型和全部对照保留；最终8日期封存、LG01暂停。\n'
else:
    R=read(ROOT/'outputs/grid43/review.json')
    note='GRID43共同并列补查24条闭环核验完成；'+'；'.join(f"{r['method']}质量{r['quality_qualified_pairs']}/4、控制{r['control_s']:.2f}秒、查询{r['queries']}" for r in R['summaries'])+'。不把共同修复归于GNN，不作封存泛化或完整SoftGNN论文复现。'
    nxt='1. 汇总GRID43实际质量—时间，确定下一步封存协议或有依据的改进。\n2. 保留三类主对照和所有种子，不增菜单；最终8日期暂封存、LG01暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — common search quality repair\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['EXPERIMENT_LOG.md','DECISIONS.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — GRID42/43 '+phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'files':counts,'historical_preserved':len(old)}))
