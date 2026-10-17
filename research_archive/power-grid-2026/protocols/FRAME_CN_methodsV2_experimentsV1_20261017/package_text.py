"""Package text-only manuscript sections; remove only enumerated new figures/tables."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone

ROOT = Path('D:/Codex/GRL研究').resolve()
OUT = ROOT/'outputs/frame_paper_sections_20261017'
COLLAB = ROOT/'论文协作资料/FRAME_论文逐节重写_20261017'
REPO = ROOT/'work/GRL-results-integration'
PROTO = REPO/'research_archive/power-grid-2026/protocols'
ARCHIVE = PROTO/'FRAME_CN_methodsV2_experimentsV1_20261017'
METHOD = Path('04_方法/FRAME_中文方法_文字修订_V2.md')
EXPERIMENT = Path('05_实验/FRAME_中文实验_文字初稿_V1.md')
NOTE = Path('05_实验/写作依据与实验取舍说明.md')

def safe(p):
    p = p.resolve()
    assert p.is_relative_to(ROOT), p
    return p

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

removed, changed = [], []
# These are this section-writing session's newly created figures and optional table.
targets = []
for base in (OUT, COLLAB):
    targets += [base/'04_方法/figures'/('FRAME_方法流程'+ext) for ext in ('.drawio','.svg','.png')]
    targets += [base/'03_相关工作/相关研究路线对照表_可选.md']
targets += [PROTO/'FRAME_CN_section04_methods_20261017/figures'/('FRAME_方法流程'+ext) for ext in ('.drawio','.svg','.png')]
targets += [PROTO/'FRAME_CN_section03_related_work_20261017/相关研究路线对照表_可选.md',
            PROTO/'FRAME_CN_section04_methods_20261017/make_diagram.py',
            ROOT/'work/frame_methods_section_20261017/make_diagram.py']
for p in targets:
    p = safe(p)
    if p.exists():
        removed.append({'path':str(p.relative_to(ROOT)), 'sha256':sha(p), 'bytes':p.stat().st_size})
        p.unlink()

# Preserve the earlier text draft while withdrawing its figure and symbol table.
oldcopies = [OUT/'04_方法/FRAME_中文方法_逐节重写_V1.md',
             COLLAB/'04_方法/FRAME_中文方法_逐节重写_V1.md',
             PROTO/'FRAME_CN_section04_methods_20261017/FRAME_中文方法_逐节重写_V1.md']
for p in oldcopies:
    p = safe(p)
    if not p.exists():
        continue
    original = p.read_text(encoding='utf-8')
    revised = original.replace('图1给出了整体流程。','')
    revised = '\n'.join(line for line in revised.splitlines()
                        if not line.startswith(('![FRAME方法流程]','图1：','|')))
    marker = '> 历史文字稿。本轮按用户要求移除图及符号表；请以同目录文字修订V2为当前方法稿。完整原版保存在Git历史。\n\n'
    if not revised.startswith(marker):
        revised = marker + revised
    if revised != original:
        changed.append({'path':str(p.relative_to(ROOT)), 'before_sha256':sha(p)})
        p.write_text(revised.rstrip()+'\n',encoding='utf-8')
        changed[-1]['after_sha256'] = sha(p)

def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def compare(name,method,baseline):
    data = rows(name)
    a = {x['date']:x for x in data if x['method']==method}
    b = {x['date']:x for x in data if x['method']==baseline}
    assert a.keys()==b.keys()
    n = len(a)
    time = sum(1-float(a[k]['controller_s'])/float(b[k]['controller_s']) for k in a)/n
    active = [k for k in a if int(b[k]['queries'])>0]
    query = sum(1-int(a[k]['queries'])/int(b[k]['queries']) for k in active)/len(active)
    return dict(dates=n,active_dates=len(active),time_percent=time*100,query_percent=query*100)

metrics = {}
metrics['GRID81_FRAME_vs_FULL'] = compare('outputs/grid81/per_date.csv','FRAME_GUARD','FULL')
assert round(metrics['GRID81_FRAME_vs_FULL']['time_percent'],2)==14.98
assert round(metrics['GRID81_FRAME_vs_FULL']['query_percent'],2)==65.10
metrics['GRID88_FRAME_vs_FIXED512'] = compare('outputs/grid88/per_date.csv','FRAME_GUARD','FIXED_512')
assert round(metrics['GRID88_FRAME_vs_FIXED512']['time_percent'],2)==3.88
assert round(metrics['GRID88_FRAME_vs_FIXED512']['query_percent'],2)==33.57
metrics['GRID86_FRAME_vs_FIXED512'] = compare('outputs/grid86/per_date.csv','FRAME_GUARD','FIXED_512')
assert round(metrics['GRID86_FRAME_vs_FIXED512']['time_percent'],2)==7.10
assert round(metrics['GRID86_FRAME_vs_FIXED512']['query_percent'],2)==40.89
metrics['GRID86_FRAME_vs_FIXED256'] = compare('outputs/grid86/per_date.csv','FRAME_GUARD','FIXED_256')
assert round(metrics['GRID86_FRAME_vs_FIXED256']['time_percent'],2)==-2.02
metrics['GRID84_META_seeds'] = [compare('outputs/grid84/per_date.csv','FRAME_GUARD',f'META_CONTEXT{i}') for i in range(5)]
metrics['GRID84_seed_mean_time_percent'] = sum(x['time_percent'] for x in metrics['GRID84_META_seeds'])/5
assert round(metrics['GRID84_seed_mean_time_percent'],2)==2.23

for rel in (METHOD, EXPERIMENT):
    text = (OUT/rel).read_text(encoding='utf-8')
    assert not re.search(r'^\s*\|',text,re.M),rel
    assert '![' not in text and '<svg' not in text,rel
    assert not re.search(r'(图|表)\s*[0-9一二三四五六七八九]',text),rel
    assert '不是' not in text and '而不是' not in text,rel
    assert '下一控制时刻重新建立' not in text,rel

receipts = dict(
    created_utc=datetime.now(timezone.utc).isoformat(),
    task='Text-only Methods V2 and Experiments V1; waiting for user review',
    removed_current_section_assets=removed,
    modified_earlier_reading_copies=changed,
    historical_experiment_outputs_and_user_drafts='unchanged',
    new_training=0,new_physical_replays=0,new_timing=0,
    metrics_recomputed_from_existing_csv=metrics,
    source_hashes={},deliverables={})

sources = ['outputs/grid81/per_date.csv','outputs/grid84/per_date.csv','outputs/grid86/per_date.csv',
           'outputs/grid87/GRID87_report.md','outputs/grid87/first_divergences.json',
           'outputs/grid88/per_date.csv','outputs/grid88/review.json',
           'outputs/grid85_mechanism/GRID85_report.md','outputs/grid85_mechanism/计算来源细分.md',
           'work/grid70/policies.py','work/grid81/mechanism_adapter.py']
for name in sources:
    receipts['source_hashes'][name]=sha(ROOT/name)

for rel in (METHOD, EXPERIMENT, NOTE):
    src=OUT/rel
    for base in (COLLAB, ARCHIVE):
        dest=safe(base/rel);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest)
    receipts['deliverables'][str(rel)]=sha(src)

combined = '# FRAME：方法与实验文字修订稿\n\n方法V2、实验V1；供通读修改。摘要、引言与相关工作不变。本稿不含图表。\n\n' + (OUT/METHOD).read_text(encoding='utf-8') + '\n\n---\n\n' + (OUT/EXPERIMENT).read_text(encoding='utf-8')
for base in (OUT, COLLAB, ARCHIVE):
    dest=safe(base/'06_本轮通读/FRAME_方法与实验_文字通读稿.md')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(combined,encoding='utf-8')
    (dest.parent/'revision_receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

nextsteps = REPO/'docs/NEXT_STEPS.md'
nextsteps.write_text('# Next steps\n\n1. 等待用户逐节审阅中文方法文字V2和实验文字V1，合并通读稿位于FRAME_CN_methodsV2_experimentsV1_20261017/06_本轮通读。不覆盖用户编辑稿，已确认摘要V3保持原文。\n2. 本阶段只做文字。按用户要求删除本次逐节写作新增的方法图和可选相关工作表；正文无图表及引用，历史实验数据和用户旧稿保留。图表待用户重新授权后统一规划。\n3. 实验主文围绕FULL计算收益、固定预算、会议适配五种子、分阶段机制和计算分解。保留固定256生成组竞争力、META小幅差距及同站补查反向结果；不将旧版敏感性冒充最终版证据。\n4. 不自动新增实验、模型、预算或种子。待用户确认这两节后再继续其指定部分。\n',encoding='utf-8')
log=REPO/'docs/EXPERIMENT_LOG.md'
with log.open('a',encoding='utf-8') as f:
    f.write('\n## 2026-10-17 — 方法文字V2与实验文字V1\n\n按用户Methods五部分和Experiments问题导向要求重写。先阅读ICAPS2022、AAAI2024实验及中国科学信息科学2024中文实验表达；方法保留定义、三条公式、完整伪代码，删去工程审计式叙述。实验按FULL/固定预算/会议适配、机制、案例、计算分解组织，全部核心对照的不利结果保留。用既有CSV复核65.10%/14.98%、33.57%/3.88%、40.89%/7.10%、固定256负向及五种子2.23%。按指示移除当前逐节写作新增图、可选表及旧阅读稿中的图表，Git历史可恢复；不动用户编辑稿或历史实验。三份文件与合并通读稿归档，同步D盘协作资料。无新增训练、物理回放或计时，等待用户审阅。\n')

print(json.dumps({'removed_assets':len(removed),'updated_earlier_copies':len(changed),
                  'metrics':metrics,'combined':str(COLLAB/'06_本轮通读/FRAME_方法与实验_文字通读稿.md')},
                 ensure_ascii=False,indent=2))
