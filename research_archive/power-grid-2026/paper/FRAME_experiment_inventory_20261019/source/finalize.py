from pathlib import Path
from collections import defaultdict
from statistics import mean
from datetime import datetime,timezone
import csv,hashlib,json,math,shutil,zipfile
import xml.etree.ElementTree as ET
ROOT=Path('D:/Codex/GRL研究'); OUT=ROOT/'outputs/frame_experiment_inventory_20261019'
WORK=ROOT/'work/frame_experiment_inventory_20261019'
REPO=ROOT/'work/GRL-results-integration'
COLLAB=ROOT/'论文协作资料/FRAME_全部实验分类汇总_20261019'
ARCH=REPO/'research_archive/power-grid-2026/paper/FRAME_experiment_inventory_20261019'
XLSX=OUT/'FRAME_全部实验分类与质量速度指标_20261019.xlsx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def main():
    data=json.loads((OUT/'experiment_tables.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(XLSX) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS)]
        def cells(index):
            root=ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml'))
            result={}
            for c in root.findall('.//m:sheetData/m:row/m:c',NS):
                v=c.find('m:v',NS)
                value=v.text if v is not None else None
                t=c.get('t')
                if t=='s':value=strings[int(value)]
                elif t=='inlineStr':
                    node=c.find('m:is',NS);value=''.join(node.itertext()) if node is not None else ''
                elif t=='e':raise AssertionError(('formula error',c.get('r'),value))
                elif value is not None and t not in ['str','s']:
                    try:value=float(value)
                    except ValueError:pass
                result[c.get('r')]=value
            return root,result
        contents={}
        for i in range(1,7):
            root,cs=cells(i);contents[i]=cs
            assert root.find('m:tableParts',NS) is not None
            pane=root.find('.//m:pane',NS)
            assert pane is not None and float(pane.get('xSplit',0))==2 and float(pane.get('ySplit',0))==6,pane.attrib
        summary=contents[2]; checked=0
        for i in range(60):
            r=i+7; key=(summary[f'A{r}'],summary[f'B{r}'])
            ds=[d for d in data['dates'] if (d['batch'],d['method'])==key]
            assert ds,key
            assert summary[f'D{r}']==f'{sum(d["quality"] for d in ds)}/{len(ds)}'
            assert summary[f'E{r}']==f'{sum(d["complete"] for d in ds)}/{len(ds)}'
            expected={'C':len(ds),'I':sum(d['queries'] for d in ds),'K':mean(d['controller'] for d in ds)}
            costs=[d['cost_delta'] for d in ds if d['cost_delta'] is not None]
            if costs:expected.update(G=mean(costs),H=max(costs))
            q=[];t=[];p95=[];sp95=[]
            for d in ds:
                b=next((b for b in data['dates'] if b['batch']==d['batch'] and b['date']==d['date'] and b['method']==d['baseline']),None)
                if b and b['steps']==d['steps']:
                    if b['queries']>0:q.append(1-d['queries']/b['queries'])
                    t.append(1-d['controller']/b['controller'])
                    if b['all_p95'] and d['all_p95'] is not None:p95.append(1-d['all_p95']/b['all_p95'])
                    if b['search_p95'] and d['search_p95'] is not None:sp95.append(1-d['search_p95']/b['search_p95'])
            for letter,vs in [('J',q),('L',t),('V',p95),('W',sp95)]:
                if vs:expected[letter]=mean(vs)
                else:assert summary[f'{letter}{r}']=='未测'
            for letter,field,mul in [('N','forecast',1),('O','optimizer',1),('P','search_other',1),('Q','other',1),
                                    ('R','all_p95',1000),('S','all_p99',1000),('T','search_p95',1000),('U','search_p99',1000)]:
                vs=[d[field] for d in ds if d[field] is not None]
                if vs:expected[letter]=mean(vs)*mul
                else:assert summary[f'{letter}{r}']=='未测'
            expected['M']=mean(d['forecast'] for d in ds)+mean(d['search_other'] for d in ds)
            for c,v in expected.items():
                assert math.isclose(summary[f'{c}{r}'],v,rel_tol=1e-9,abs_tol=1e-9),(key,c,summary[f'{c}{r}'],v)
                checked+=1
        for src,receipt in data['source_receipts'].items():
            assert sha(Path(src))==receipt['sha256'],src
        formula_count=sum(len(ET.fromstring(z.read(n)).findall('.//m:f',NS)) for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
        assert formula_count>1000,formula_count
    # Plain reading version, keeping classifications and source references.
    parts=['# FRAME实验分类总表\n','整理日期：2026年10月19日。\n',
           'Excel包含贡献总表、质量速度汇总、逐日期数据、逐轮记录、原报告指标和对照口径。覆盖96轮记录；这些轮次包含实验、拟合、离线分析及工程资格，不能作为96个独立评价样本。\n',
           '当前论文FRAME对应最终FRAME_GUARD。质量达标、整周完成、成本变化分别列出；速度同时查看查询、搜索、完整控制、延迟与重复波动。没有重新运行物理实验。\n']
    def safe(x):return str(x).replace('|','／').replace('\n','<br>')
    for category in dict.fromkeys(r[0] for r in data['main']):
        parts+=['\n## '+category+'\n',
                '| 实验与场景 | 和谁比较 | 实验目的 | 结果 | 证据说明 |\n|---|---|---|---|---|\n']
        for r in data['main']:
            if r[0]!=category:continue
            first=f'[{r[1]}]({r[8].splitlines()[0]})：{r[2]}<br>{r[4]}'
            parts.append('| '+' | '.join(safe(x) for x in [first,r[3],r[5],r[6],r[7]])+' |\n')
    parts+=['\n## 全部实验记录\n',
            '| 编号 | 类别 | 实验 | 目的 | 结果 | 与当前稿的关系 |\n|---|---|---|---|---|---|\n']
    for r in data['history']:
        parts.append('| '+' | '.join(safe(r[k]) for k in ['编号','类别','实验名称与版本','实验目的：检验什么','结果摘要','与当前论文的关系'])+' |\n')
    parts+=['\n## 指标口径\n']
    for kind,name,meaning in data['dictionary']:
        if kind in ['质量','速度','统计']:parts.append(f'- **{name}**：{meaning}\n')
    (OUT/'FRAME_实验分类总表_阅读版.md').write_text(''.join(parts),encoding='utf-8')
    with (OUT/'逐轮实验记录.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data['history'][0]));w.writeheader();w.writerows(data['history'])
    with (OUT/'全部原报告指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['编号','报告文件','原表与章节','原行上下文','原指标','原值','指标类别','来源']);w.writerows(data['raw_metrics'])
    review={'utc':datetime.now(timezone.utc).isoformat(),'workbook_sha256':sha(XLSX),'all_sheets_reviewed':True,
        'visual_review':['main contribution purpose/results','quality/control summary','component/latency summary','per-date table','all-stage history','raw reported metrics','definitions'],
        'source_files_unchanged':len(data['source_receipts']),'stages':96,'contribution_rows':38,'normalized_method_batches':60,
        'per_date_rows':354,'reported_metric_cells':4710,'formula_cache_checks_against_independent_python':checked,
        'formulas_in_export':formula_count,'xlsx_error_cells':0,'tables_and_filter_controls':6,
        'frozen_rows':6,'frozen_columns':2,'no_new_experiments':True,
        'validation_scope':'Artifact Tool recalculation/inspection, rendered sheet review, exported XML/cache cross-check. No native Excel/WPS execution claimed.'}
    (OUT/'validation.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    readme='''# FRAME全部实验分类与质量速度指标

先看Excel的“贡献总表”，再看“质量速度汇总”。其余页面可核对逐日期数据、完整轮次、原报告指标与方法定义。可用每页表头筛选类别、编号、批次或方法；顶部与前两列冻结。

- 贡献总表：38项贡献证据，说明和谁比、检验什么、结果及适用范围。
- 质量速度汇总：60个方法×批次，质量达标/完整周数、终止步数范围、平均/最大可比完整周成本、查询、完整控制/搜索耗时、互斥分解、全步骤和搜索步骤P95/P99、计时重复及CV。
- 逐日期数据：354条方法×日期记录，不重复累计时间重复中的相同查询。
- 逐轮记录：GRID00–89加6个B/C补充轮，共96轮，含早期探索、资格、部分完成和失败版本。
- 原报告指标：4710个原表单元，保留原始列名与来源版本。
- 对照与口径：版本、场景、单位、计数、基线及比例方向。

质量与成本按冻结FULL定义；速度使用同批参照列。当前FRAME是FRAME_GUARD，旧GNN/旧FRAME/学习门控不混入。没有测量的留空或“未测”，单次计时不伪造波动。原版、全部不利结果、待机异常RAW与统一补测均保留。GRID50旧报告文字与数值表冲突已标注。

所有新文件在D盘。本次仅整理已有报告、CSV和收据，没有训练、回放、物理求解或新增科学审计。核验记录见validation.json，源文件哈希见source_receipts.json。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    deliver=['FRAME_全部实验分类与质量速度指标_20261019.xlsx','FRAME_实验分类总表_阅读版.md',
             '逐轮实验记录.csv','全部原报告指标.csv','experiment_tables.json','source_receipts.json','validation.json','README.md','workbook_build.json']
    for target in [COLLAB,ARCH]:
        target.mkdir(parents=True,exist_ok=True)
        for name in deliver:
            shutil.copy2(OUT/name,target/name);assert sha(OUT/name)==sha(target/name)
        (target/'source').mkdir(exist_ok=True)
        for p in WORK.iterdir():
            if p.suffix in ['.py','.mjs','.tsv']:shutil.copy2(p,target/'source'/p.name)
    index=REPO/'docs/PAPER_MATERIALS_INDEX.md';txt=index.read_text(encoding='utf-8')
    if 'FRAME_experiment_inventory_20261019' not in txt:
        txt+='''

## 2026年10月19日 全部实验分类与质量速度指标

应作者要求整理全部GRID00–89及六个B/C轮次，共96轮。按8类组织38项贡献证据，另列60个方法批次的质量/速度多指标、354条逐日期记录及4710个原报告指标。明确早期GNN/原FRAME/最终FRAME_GUARD/学习门控、质量与完整周、宏平均与汇总比。原始结果、失败版本和待机RAW均保留；未新增实验或训练。

- [阅读与文件说明](../research_archive/power-grid-2026/paper/FRAME_experiment_inventory_20261019/README.md)
- [分类阅读版](../research_archive/power-grid-2026/paper/FRAME_experiment_inventory_20261019/FRAME_实验分类总表_阅读版.md)
- D盘协作入口：D:/Codex/GRL研究/论文协作资料/FRAME_全部实验分类汇总_20261019。
'''
        index.write_text(txt,encoding='utf-8')
    print(json.dumps(review,ensure_ascii=False))
    print('COLLAB '+str(COLLAB))
if __name__=='__main__':main()
