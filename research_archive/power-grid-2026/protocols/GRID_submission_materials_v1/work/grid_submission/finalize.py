"""Finalize author materials and local raw bundle without changing experiments."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/grid_submission_materials_v1'
REPO = ROOT/'work/GRL-results-integration'
DEST = REPO/'research_archive/power-grid-2026/protocols/GRID_submission_materials_v1'
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p:json.loads(p.read_text(encoding='utf-8'))


def main():
    assert not (OUT/'validation.json').exists() and not DEST.exists()
    assert read(OUT/'reproducibility/material_audit.json')['passed']
    assert read(ROOT/'outputs/grid_submission_table_check_v1/validation.json')['passed']
    sources = [
        {'id':'CCF-AI','title':'CCF artificial-intelligence category','url':'https://www.ccf.org.cn/Academic_Evaluation/AI/','checked':'2026-10-13','status':'official indexed page; direct fetch405','facts':'ECAI and ICAPS listed as B; category view may update separately from downloadable directory.'},
        {'id':'CCF-DM','title':'CCF database/data-mining category','url':'https://www.ccf.org.cn/Academic_Evaluation/DM_CS/','checked':'2026-10-13','status':'official indexed page; direct fetch405','facts':'ECML-PKDD listed as B.'},
        {'id':'CCF-2026','title':'Seventh directory release announcement','url':'https://www.ccf.org.cn/Academic_Evaluation/By_category/','checked':'2026-10-13','status':'official indexed announcement; direct fetch is script challenge; full PDF not downloaded','facts':'Released2026-03-31; announcement distinguishes formal Full/Regular long papers from shorts/demos/workshops. Institutional applicable version remains to confirm.'},
        {'id':'ECAI27','title':'ECAI2027 official homepage','url':'https://ecai2027.org/','checked':'2026-10-13','status':'current official live page','facts':'Full-paper date2027-04-14; conference2027-10-02 to07; includes PAIS. Detailed2027 CFP/format not retrieved.'},
        {'id':'ECAI25','title':'ECAI2025 main CFP','url':'https://ecai2025.org/call-for-papers/','checked':'2026-10-13','status':'historical official requirements','facts':'Planning/Search in scope; 7 content+atmost1 reference page; double blind. Historical AI-text versus polishing policy, not automatically2027.'},
        {'id':'PAIS25','title':'PAIS2025 CFP','url':'https://ecai2025.org/call-for-pais-papers/','checked':'2026-10-13','status':'historical official requirements','facts':'Emerging prototypes/simulations eligible;7+1 pages; single blind; publication together with ECAI. 2027 policy and local recognition unresolved.'},
        {'id':'ICAPS27','title':'ICAPS2027 CFP','url':'https://icaps27.icaps-conference.org/calls/cfp/','checked':'2026-10-13','status':'current official live CFP','facts':'Abstract2026-12-07/paper2026-12-14 at23:59UTC-12; long8 content pages plus references, AAAI format, double blind. Substantive planning relevance required.'},
        {'id':'ICAPS27-TRACK','title':'Planning Under a Different Name','url':'https://icaps27.icaps-conference.org/calls/special_tracks/','checked':'2026-10-13','status':'current official track scope','facts':'Neighbor-community search/deliberation can fit; same novelty/significance standards, sequential ML alone insufficient.'},
        {'id':'ECML27','title':'ECMLPKDD2027 homepage','url':'https://ecmlpkdd.org/2027/','checked':'2026-10-13','status':'current official homepage','facts':'Conference2027-08-30 to09-03; ADS mentioned; detailed2027 submission deadline not located.'},
        {'id':'ECML26-ADS','title':'ECMLPKDD2026 ADS CFP','url':'https://ecmlpkdd.org/2026/submissions-ads-track/','checked':'2026-10-13','status':'historical official requirements','facts':'Application relevance and insight, method novelty not mandatory;16 LNCS pages incl references; actual generativeAI usage declared. Not2027 frozen rules.'},
        {'id':'SOFT-PUB','title':'SoftGNN official publication','url':'https://link.springer.com/chapter/10.1007/978-3-032-06129-4_8','checked':'2026-10-13','status':'publisher metadata/preview','facts':'ECMLPKDD2025 ADS/Demo combined volume16022,129-146; detailed methods checked in author v2, not subscription full-text page-by-page.'},
        {'id':'SOFT-V2','title':'SoftGNN author v2','url':'https://arxiv.org/abs/2503.15190v2','checked':'2026-10-13','status':'author primary metadata and open method text','facts':'Metadata explicitly says accepted ADS; formal DOI matches. First-safe/.9/disconnected-endpoint/N1-extension mechanics credited.'},
        {'id':'LJN','title':'LJN official implementation','url':'https://github.com/lajavaness/l2rpn-2023-ljn-agent','checked':'2026-10-13','status':'official source','facts':'Existing hybrid and ranking components credited; local compatibility receipts retained.'},
        {'id':'DEJONG-V1','title':'Generalizable GNNs old v1','url':'https://arxiv.org/html/2501.07186v1','checked':'2026-10-13','status':'author fixed old version','facts':'Method discussion uses v1 section4.9.'},
        {'id':'DEJONG-V3','title':'Transmission-grid GNNs current v3','url':'https://arxiv.org/html/2501.07186v3','checked':'2026-10-13','status':'author new title/version','facts':'Same work/three authors; not a second independent paper; old section numbers not assumed to transfer.'},
        {'id':'BHATIA22','title':'Anytime planning metareasoning','url':'https://ojs.aaai.org/index.php/ICAPS/article/view/19842','checked':'2026-10-13','status':'official proceedings metadata/abstract','facts':'2022,32,556-564,DOI10.1609/icaps.v32i1.19842; existing stopping/hyperparameter-control concept.'},
        {'id':'MILLI17','title':'Bounded-optimal metareasoning','url':'https://ojs.aaai.org/index.php/AAAI/article/view/11156','checked':'2026-10-13','status':'official proceedings metadata/abstract','facts':'2017,31,DOI10.1609/aaai.v31i1.11156; conceptual context, not new empirical comparator.'}]
    (OUT/'literature/official_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'literature/official_sources.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(sources[0]));w.writeheader();w.writerows(sources)
    venues=[
        {'venue':'ECAI2027 main','role':'default positioning','paper_date':'2027-04-14','deadline_timezone':'not verified in detailed2027 CFP','format':'2027 not retrieved; historical2025 7+1','ccf_source':'CCF-AI','note':'Framework/search/application contribution; no acceptance probability assigned.'},
        {'venue':'ECMLPKDD2027 ADS','role':'applied alternative','paper_date':'not published/located','deadline_timezone':'not verified','format':'2027 not retrieved; historical2026 LNCS16 incl refs','ccf_source':'CCF-DM','note':'Data-driven method relevance must be clear; do not force unsupported GNN increment.'},
        {'venue':'ICAPS2027 long','role':'earlier conditional option','paper_date':'2026-12-14 23:59','deadline_timezone':'UTC-12; Beijing2026-12-15 19:59','format':'8 content pages plus references;AAAI;double blind','ccf_source':'CCF-AI','note':'Planning/deliberation contribution required; short paper not substituted for CCF long target.'},
        {'venue':'ECAI2027 PAIS','role':'conditional emerging application','paper_date':'track-specific not verified','deadline_timezone':'not verified','format':'2027 track CFP not retrieved','ccf_source':'institution confirmation required','note':'2025 accepts prototypes; cannot transfer rules or guarantee local long-paper recognition.'}]
    with (OUT/'venue_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(venues[0]));w.writeheader();w.writerows(venues)
    ElementTree.parse(OUT/'figures/framework.svg')
    shutil.copy2(ROOT/'outputs/grid_submission_table_check_v1/validation.json',OUT/'reproducibility/portable_table_check.json')
    # One current-data receipt must be kept with this offline material version.
    payloads=read(OUT/'reproducibility/raw_payload_manifest.json')
    data=OUT/'data_archives';data.mkdir()
    raw=data/'GRID61_GRID62_完整原始记录.zip'
    assert shutil.disk_usage(ROOT).free>2*sum(x['bytes'] for x in payloads)
    with zipfile.ZipFile(raw,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for x in payloads:
            p=ROOT/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
            z.write(p,x['path'])
        for stage in ('grid61','grid62'):
            for p in (ROOT/'outputs'/stage).iterdir():
                if p.is_file() and p.suffix in ('.json','.md','.csv'):
                    z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(raw) as z:assert z.testzip() is None
    receipt={'path':raw.relative_to(ROOT).as_posix(),'bytes':raw.stat().st_size,'sha256':sha(raw),
        'raw_files':len(payloads),'crc_checked':True,'scope':'Original GRID61/62 output evidence, no new runs or model fits; local large-data bundle, not committed or published to Git.'}
    (OUT/'reproducibility/raw_archive_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    validated={'passed':True,'portable_table_rows':8,'date_method_rows':28,'claims':14,'official_sources':len(sources),
        'bibliography_entries':6,'svg_xml_parsed':True,'figures_visual_status':'Scientific PNG/PDF inspected; framework SVG rendered with bundled sharp and its preview visually inspected for labels/arrows.',
        'raw_zip_crc_checked':True,'raw_zip_bytes':raw.stat().st_size,'new_simulator_runs':0,'new_model_fits':0,
        'original_outputs_unchanged':True,'legacy_estimator_preserved':True,
        'limitations':'No full-paper native typesetting, submission, complete standalone simulator installation, new scientific confirmation, or acceptance prediction.'}
    for x in read(OUT/'reproducibility/source_receipts.json'):assert sha(ROOT/x['path'])==x['sha256']
    (OUT/'validation.json').write_text(json.dumps(validated,indent=2),encoding='utf-8')
    files=[p for p in OUT.rglob('*') if p.is_file() and 'data_archives' not in p.parts and '__pycache__' not in p.parts]
    for p in (ROOT/'work/grid_submission').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files.append(p)
    DEST.mkdir()
    receipts=[]
    for src in files:
        dest=DEST/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest);receipts.append({'path':src.relative_to(ROOT).as_posix(),'bytes':src.stat().st_size,'sha256':sha(src)})
    (DEST/'manifest.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    light=OUT/'投稿定位与重写材料_轻量包.zip'
    with zipfile.ZipFile(light,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            relative=p.relative_to(OUT) if p.is_relative_to(OUT) else Path('preparation_scripts')/p.relative_to(ROOT/'work/grid_submission')
            z.write(p,relative.as_posix())
    with zipfile.ZipFile(light) as z:assert z.testzip() is None
    note='完成一次正式投稿定位与作者重写素材包：ECAI框架/搜索主方向、ECML2027 ADS应用备选、ICAPS2027较早但规划范围有条件，PAIS待当届细则与单位标准。官方来源17条、8行统一主表/28日期方法行、14项主张证据、6条BibTeX、代码映射/算法/性质、SVG与科学图、标准与AI使用事实说明已准备。按先平均重复再算比例统一GRID61 FRAME为13.1121%，原13.0588%保留，不改科学裁决。独立便携表格复算通过；769MB原始记录另建本地raw包并CRC/哈希校验，未推入Git。轻量附件齐全但尚不是匿名投稿或完整独立仿真安装包。无新运行/训练；两个离线材料目录/换行错误留档，未损及历史结果。'
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — submission positioning and author materials\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';p.write_text('# Research state — author materials ready\n\n- **当前：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
    p=REPO/'docs/NEXT_STEPS.md';p.write_text('# Next steps\n\n1. 投稿定位与可供作者重写的材料已经准备并验证；主方法/全部结果保留，无待跑实验。\n2. 等用户给出单位/稿件标准和自主重写稿，再按目标track精修；ECAI/ECML2027详细CFP还需发布后刷新。ICAPS2027全文2026-12-14 UTC-12，但不只为赶截止选它。\n3. 保留GNN增量未建立、首个安全参照竞争力、无搜索日期和计时变异；不追加模型、基线或封存调参。raw证据包在本地，Git只有轻量材料及哈希收据。\n',encoding='utf-8')
    p=REPO/'docs/GRID_PAPER_EVIDENCE.md';p.write_text('# Grid paper evidence — submission material package\n\n'+note+'\n\n[材料入口](../research_archive/power-grid-2026/protocols/GRID_submission_materials_v1/outputs/grid_submission_materials_v1/README.md) · [投稿定位](../research_archive/power-grid-2026/protocols/GRID_submission_materials_v1/outputs/grid_submission_materials_v1/01_投稿定位.md)\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
    print(json.dumps({'passed':True,'archived_small_files':len(files),'light_zip_bytes':light.stat().st_size,'raw_zip_bytes':raw.stat().st_size}))


if __name__=='__main__':main()
