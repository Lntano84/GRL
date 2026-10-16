"""Complete the requested method/evidence packet after GRID82, retain pre-run materials."""
from pathlib import Path
import csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid82_method';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    R=read(ROOT/'outputs/grid82/review.json');assert R['passed_saved_data'];C=R['comparisons'];D=read(ROOT/'outputs/grid82/decision.json')
    text=(OUT/'最终方法_伪代码与证据对应.md').read_text(encoding='utf-8')
    text=text.replace('GRID82最终固定核查版同批36条待运行','GRID82最终固定核查版同批36条已完成')
    text=text.replace('当前尚未由最终版本的该组比较裁决','GRID82中两会议种子均通过质量条件；相对种子0通过同质量提速筛查，相对种子1未建立稳定提速')
    addition='\n\n## 9. GRID82完成后的主对照结论\n\n'
    addition+='36条正式回放及保存数据审计完成，实际回放约38.0分钟，新增拟合0。固定核查版与两个会议适配种子均满足6/6场景质量条件，均完成4/6场景，另外两场景终止长度相同。固定核查最大完整周成本增幅0.0866%，会议种子0为0.3776%，种子1为0.0866%。\n\n'
    for c,d in zip(C,D['decisions']):
        addition+=f"- 相对{c['baseline']}：控制时间宏平均减少{c['macro_controller_reduction']*100:.3f}%，活跃同终止长度场景查询宏平均减少{c['macro_query_reduction_equal_horizons']*100:.3f}%，冻结判读{d['category']}。\n"
    addition+='\n论文可据此说明固定核查流程具有竞争力：对一个冻结种子取得稳定计算收益，对另一个种子没有建立额外提速。两个种子全部保留，不能以较弱种子的胜利概括整类会议方法。相对FULL的同批计算证据仍引用GRID81，不能用GRID82新计时与GRID81旧FULL计时拼接新的提速比例。\n\n本完成版补齐原预运行材料中的待验证项，旧材料和用户手写稿均保留。正式逐场景表见outputs/grid82/GRID82_report.md。\n'
    complete=OUT/'最终方法_伪代码与证据对应_完成版.md';assert not complete.exists();complete.write_text(text+addition,encoding='utf-8')
    with (OUT/'贡献与证据对应表.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        if row['claim']=='最终版本优于既有会议策略':
            row.update(evidence='GRID82：36条最终固定核查版同批比较，六场景、两冻结种子、两次计时',result='三个方法均6/6通过质量；相对会议种子0控制时间宏平均减少6.435%、查询减少38.475%；相对种子1时间减少0.017%、查询相同，未建立稳定优势',paper_role='最终版本外部适配主对照',boundary='任务适配且原校准质量缺口保留；两个种子全报，不能概括为普遍胜过会议方法；六场景已暴露')
    complete_csv=OUT/'贡献与证据对应表_完成版.csv'
    with complete_csv.open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    collab=ROOT/'论文协作资料/FRAME_最终方法与会议对照_20261016'
    for p in [complete,complete_csv,ROOT/'outputs/grid82/GRID82_report.md']:
        assert not (collab/p.name).exists();shutil.copy2(p,collab/p.name)
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID82_method_complete';assert not archive.exists();entries=[]
    for p in [complete,complete_csv,Path(__file__)]:
        dest=archive/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (archive/'manifest.json').write_text(json.dumps(dict(files=entries,scope='Post-run completion of requested Chinese method and evidence packet; benchmark sources and all older/user drafts unchanged.'),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID82已完成，按用户工作1/2补齐中文最终方法和证据表完成版，已放入协作文件夹。相对两个冻结会议种子的控制宏平均减少6.435%/0.017%；均6/6质量通过，前者通过提速筛查，后者无稳定增量。旧预运行稿和原手写稿不覆盖；不新增实验或训练。'
    p=REPO/'docs/EXPERIMENT_LOG.md';p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-16 — GRID82 requested materials complete\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps(dict(materials_completed=True,new_fits=0,new_runs=0,files=len(entries))))
if __name__=='__main__':main()
