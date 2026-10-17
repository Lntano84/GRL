"""Deliver visually approved V2 and update research/material indices."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review';REPO=ROOT/'work/GRL-results-integration';DOC=OUT/'manuscript_v2'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
 Q=read(OUT/'visual_review.json');word=DOC/'FRAME_ICAPS2027_中文合稿_V2.docx';assert Q['all_pages_reviewed'] and Q['docx_sha256']==hashlib.sha256(word.read_bytes()).hexdigest()
 R=read(DOC/'合稿来源与修改说明.json');R.update(visual_review_pending=False,visual_review=Q);(DOC/'合稿来源与修改说明.json').write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding='utf-8')
 (DOC/'交付完成.json').write_text(json.dumps(Q,ensure_ascii=False,indent=2),encoding='utf-8')
 (DOC/'阅读说明.md').write_text('''# 中文合稿V2 审稿意见补强

本稿保留V1和用户编辑稿，补充候选顺序来源、算法性质、停止依据、META25维状态及非零成本比较、候选换序、最终版参数检查、独立生成种子评价、等预算闭环及搜索步骤延迟。全部新实验和较差结果均保留。

完整研究报告与逐日期CSV位于outputs/grid89_review；论文仍为中文阅读修改版，无新增图表，尚未套用英文投稿模板。公式和伪代码可编辑，已逐页检查。请结合新增实验决定最终贡献表述。
''',encoding='utf-8')
 collab=ROOT/'论文协作资料/FRAME_中文合稿_审稿补强_V2_20261017';repo=REPO/'research_archive/power-grid-2026/paper/FRAME_CN_review_V2_20261017'
 files=['FRAME_ICAPS2027_中文合稿_V2.docx','FRAME_ICAPS2027_中文合稿_V2.md','references.bib','合稿来源与修改说明.json','交付完成.json','阅读说明.md']
 for target in [collab,repo]:
  target.mkdir(parents=True,exist_ok=True)
  for name in files:shutil.copy2(DOC/name,target/name)
  shutil.copy2(OUT/'GRID89_report.md',target/'GRID89_report.md')
 index=REPO/'docs/PAPER_MATERIALS_INDEX.md';old=index.read_text(encoding='utf-8')
 intro='''## 最新中文V2 审稿补强

[Word全文](../research_archive/power-grid-2026/paper/FRAME_CN_review_V2_20261017/FRAME_ICAPS2027_中文合稿_V2.docx) · [在线阅读全文](../research_archive/power-grid-2026/paper/FRAME_CN_review_V2_20261017/FRAME_ICAPS2027_中文合稿_V2.md) · [新增实验完整报告](../research_archive/power-grid-2026/protocols/GRID89_review/complete/GRID89_report.md)

完成冻结的换序、参数、非零成本META与独立生成种子比较；全部结果和原稿保留。当前为中文全文，仍无新增图表。

'''
 if '## 最新中文V2 审稿补强' not in old:index.write_text(old.replace('\n\n','\n\n'+intro,1),encoding='utf-8')
 D=read(OUT/'decision.json');sealed=next(c for c in D['comparisons'] if c['phase']=='sealed' and c['method']=='FRAME_GUARD' and c['baseline']=='FULL')
 state=f"2026-10-17 GRID89审稿补强完成：246条新回放、3个非零成本种子，保存数据审计及中文V2逐页复核通过。独立生成种子场景FRAME质量{D['sealed_frame_quality_dates']}/6；对FULL控制时间宏平均变化{sealed['macro_controller_reduction']*100:+.3f}%（正值为减少），冻结同质量5%筛查={sealed['stable_5percent_screen']}。完整不利结果保留，不自动调参或追加场景。"
 p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');a,b=s.split('\n',1);p.write_text(a+'\n\n- **'+state+'**\n'+b,encoding='utf-8')
 with (REPO/'docs/EXPERIMENT_LOG.md').open('a',encoding='utf-8') as f:f.write('\n## GRID89 审稿意见有限补强完成\n\n'+state+' 候选编号证实承接作者教师频次降序，前128占原记录频次44.38%；顺序和参数不据新结果重选。META基线采用原核查委托权限、固定非零费用0.0025，旧0和全部种子保留。新主种子20261017与旧20261016区分，同电网、同日历日期的随机时序验证。新稿另存，完整记录ZIP与SHA256归档。\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 用户通读审稿补强后的中文V2，结合GRID89完整结果确定贡献范围。\n2. 保留V1、旧实验与全部较差结果，不自动新增种子、预算、模型、场景或训练。\n3. 按用户修改意见精修正文，再决定英文模板和配图。\n',encoding='utf-8')
 p=REPO/'docs/RECORDS_INDEX.md';s=p.read_text(encoding='utf-8');s=s.replace('\n\n','\n\n[GRID89审稿补强完整记录](../research_archive/power-grid-2026/protocols/GRID89_review/complete/GRID89_report.md)：246条回放、3次拟合、审计和中文V2，作为原快照后的新增归档。\n\n',1);p.write_text(s,encoding='utf-8')
 print(state,flush=True)
if __name__=='__main__':main()
