from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil, unicodedata

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_paper_sections_20261017/07_参考文献'
COLLAB=ROOT/'论文协作资料/FRAME_论文逐节重写_20261017'
REPO=ROOT/'work/GRL-results-integration'
ARCHIVE=REPO/'research_archive/power-grid-2026/protocols/FRAME_CN_references_20261017'

def parse_bib(s):
    """Parse the braced-value subset deliberately used in the final bibliography."""
    entries=[];i=0
    def ws(i):
        while i<len(s) and s[i].isspace():i+=1
        return i
    while (i:=ws(i))<len(s):
        m=re.match(r'@(\w+)\{([a-z0-9]+),',s[i:])
        assert m,('Bad entry header',s[i:i+80])
        kind,key=m.groups();i+=m.end();fields={}
        while True:
            i=ws(i)
            if s[i]=='}':i+=1;break
            f=re.match(r'(\w+)\s*=\s*\{',s[i:]);assert f,s[i:i+60]
            name=f.group(1);assert name not in fields
            i+=f.end();start=i;depth=1
            while depth:
                assert i<len(s),'Unclosed field'
                if s[i]=='\\':i+=2;continue
                if s[i]=='{':depth+=1
                if s[i]=='}':depth-=1
                i+=1
            fields[name]=s[start:i-1]
            i=ws(i)
            if i<len(s) and s[i]==',':i+=1
        entries.append(dict(type=kind,key=key,fields=fields))
    return entries

s=(OUT/'references.bib').read_text(encoding='utf-8')
entries=parse_bib(s)
assert len(entries)==9
keys=[e['key'] for e in entries];assert len(keys)==len(set(keys))
allowed={'author','title','booktitle','journal','volume','number','pages','year'}
seen_titles=set()
for e in entries:
    f=e['fields'];assert set(f)<=allowed
    assert {'author','title','year'}<=set(f)
    assert e['type'] in {'inproceedings','article','misc'}
    assert re.fullmatch(r'[a-z]+\d{4}[a-z][a-z0-9]*',e['key'])
    assert f['year'] in e['key']
    assert ' and ' in f['author'] or e['type']=='misc'
    assert not re.search(r'\bet al\b|\bothers\b|\b\d{4}\b',f['author'])
    norm=re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',f['title']).lower())
    assert norm not in seen_titles;seen_titles.add(norm)
    if e['type']=='inproceedings':
        assert {'booktitle','pages'}<=set(f)
        assert 'journal' not in f and 'volume' not in f and 'number' not in f
        assert re.fullmatch(r'\d+--\d+',f['pages'])
    if e['type']=='article':
        assert {'journal','volume','pages'}<=set(f)
        assert 'booktitle' not in f
        assert 'number' not in f # These two verified journals have no issue metadata.
    if e['key']=='lan2020topology':assert '{AI}' in f['title'] and '{ATCs}' in f['title']
    if e['key']=='tolpin2013heuristics':assert '{A*}' in f['title']
    if e['key']=='donnot2020grid2op':assert '{Grid2Op}' in f['title']
    if e['key']=='budd2024planner':assert '{Stop! Planner Time}' in f['title']
assert 'zhou2021' not in s and 'arcticle' not in s and 'DBLP:' not in s

E={e['key']:e for e in entries}
crossrefs={'bhatia2022tuning':'bhatia','budd2024planner':'budd',
           'hassouna2026topology':'hassouna','lan2020topology':'lan',
           'lehna2023managing':'lehna','liu2024progressive':'liu','svegliato2018online':'svegliato'}
metadata_checks=[]
for key,source in crossrefs.items():
    m=json.loads((OUT/'sources'/f'{source}_crossref.json').read_bytes())['message']
    f=E[key]['fields']
    if m.get('page'):assert f['pages']==m['page'].replace('--','-').replace('-','--')
    authors=[a['family']+', '+a['given'] for a in m['author']]
    actual=f['author'].replace(r'{\"u}','ü')
    assert actual==' and '.join(authors),(key,actual,authors)
    year=m.get('published-print',m['published'])['date-parts'][0][0]
    assert f['year']==str(year),(key,year)
    if m.get('volume') and E[key]['type']=='article':assert f['volume']==m['volume']
    metadata_checks.append({'key':key,'source':f'{source}_crossref.json','authors_pages_year_match':True})

old=ROOT/'outputs/frame_paper_sections_20261017/03_相关工作/FRAME_中文相关工作_逐节重写_V1.md'
new=old.with_name('FRAME_中文相关工作_引文修订_V2.md')
assert old.exists() and new.exists() and '周毅' not in new.read_text(encoding='utf-8')
assert '周毅' in old.read_text(encoding='utf-8')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt={
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'reference_count':len(entries),
 'types':{t:sum(e['type']==t for e in entries) for t in ('inproceedings','article','misc')},
 'keys':keys,'checks':{'syntax_braces':True,'unique_keys_titles':True,'allowed_fields':True,
 'publication_types':True,'page_ranges_or_verified_article_numbers':True,'protected_names':True,
 'all_paper_authors_kept':True,'publication_years_checked':True},
 'crossref_field_checks':metadata_checks,
 'manual_checks':['Tolpin original proceedings title, author order, pages 674--680',
 'Grid2Op official suggested author B. Donnot and year 2020',
 'Zhou 2021 retraction on current journal page and BibTeX export',
 'Springer citation year 2026 versus conference year 2025',
 'Lehna published author order differs from arXiv'],
 'compiler_validation':'BibTeX subset syntax checked; no ICAPS template/BST compilation in this writing stage',
 'dbpl_download_status':'Direct BibTeX downloads blocked; visible DBLP records plus publisher/Crossref used, no challenge page treated as a valid BibTeX entry',
 'source_files':{},'deliverable_hashes':{},
 'old_drafts_preserved':True,'new_training':0,'new_simulation':0}
for p in sorted((OUT/'sources').glob('*')):
    if p.is_file():receipt['source_files'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
for name in ['references.bib','FRAME_参考文献_核定稿_V1.md','格式规则与逐篇核对记录.md','实验软件与实现入口.md']:
    receipt['deliverable_hashes'][name]=sha(OUT/name)
receipt['related_work_v1_sha256']=sha(old)
receipt['related_work_v2_sha256']=sha(new)
(OUT/'reference_check.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

for base in [COLLAB/'07_参考文献',ARCHIVE/'07_参考文献']:
    base.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,base/p.name)
    # Metadata is small and useful for provenance; full paper PDFs stay in the D: reading archive.
    (base/'sources').mkdir(exist_ok=True)
    for p in (OUT/'sources').iterdir():
        if p.is_file() and p.suffix in ('.json','.bib'):
            shutil.copy2(p,base/'sources'/p.name)
    for name in receipt['deliverable_hashes']:
        assert sha(OUT/name)==sha(base/name)
for p in [COLLAB/'03_相关工作'/new.name,ARCHIVE/'03_相关工作'/new.name]:
    p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(new,p)
for name in ['fetch_metadata.py','check_and_package.py']:
    shutil.copy2(ROOT/'work/frame_references_20261017'/name,ARCHIVE/name)

(ARCHIVE/'README.md').write_text('# 中文参考文献核定稿\n\n参考文献阅读版和正式整理BibTeX在07_参考文献。9项包括8篇论文和1项软件。相关工作V2仅修订引文：撤稿周毅2021退出，替换为已核对的Lan2020；旧版保留。Lehna采用正式期刊作者顺序；SoftGNN采用Springer引文年2026。\n\n每项来源、数据库获取边界及格式选择详见逐篇核对记录。主要字段与Crossref、正式页面核对，reference_check.json保存哈希及语法检查。未编译ICAPS模板，未新增图表或实验。PDF全文仅保存在D盘阅读目录，仓库只保留元数据和写作材料。\n',encoding='utf-8')

with (REPO/'docs/EXPERIMENT_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## 2026-10-17 — 参考文献核定与撤稿引用纠正\n\n逐篇打开当前技术引用的出版方/官方论文来源，手动统一9条参考文献（6会议、2期刊、1软件）的key、类型、作者、题名保护、会议全称及页码。发现周毅2021当前期刊记录已撤稿，旧全文页未标；撤出有效引用，另核Lan2020原论文并保存相关工作引文V2，旧稿保留。Lehna改用正式期刊作者顺序；SoftGNN会议2025/正式引文2026分开；文章编号不造页码或期号。DBLP直接导出被验证页阻挡，明确记录并用可见记录、官方导出与Crossref交叉核对，不伪称全部原始DBLP BibTeX已取得。语法/键/字段/大小写/作者页码年份检查通过，无ICAPS样式编译、新训练或仿真。材料归档FRAME_CN_references_20261017。\n')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 等待用户审阅方法文字V2、实验文字V1及参考文献核定V1。当前参考文献9项在FRAME_CN_references_20261017；BibTeX按用户规则整理。\n2. 当前相关工作采用引文修订V2：周毅2021已撤稿，替换为经核对Lan2020；旧V1和用户编辑稿保留。Lehna采用正式版作者顺序，SoftGNN正式引文年2026。\n3. 仍只做文字。图表、新实验、模型和预算不自动追加；后续按用户指定章节推进。\n4. 全文内容确定后统一正文引用键，并在ICAPS正式模板下检查参考文献排版。当前仅书目字段与BibTeX语法核对，不把这一步称模板编译完成。\n',encoding='utf-8')
print(json.dumps({'reference_count':len(entries),'types':receipt['types'],
                  'crossref_checks':len(metadata_checks),'output':str(COLLAB/'07_参考文献'),
                  'related_work_v2':str(COLLAB/'03_相关工作'/new.name)},ensure_ascii=False,indent=2))
