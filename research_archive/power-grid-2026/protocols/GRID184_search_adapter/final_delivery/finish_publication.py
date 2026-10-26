"""Finalize bounded GRID184 evidence without new models or physical predictions."""
from pathlib import Path
import collections,datetime,hashlib,json,shutil,sys,zipfile
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid184_search_adapter';W=Path(__file__).parent;REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter'
sys.path.insert(0,str(R/'work/grid184_search_adapter'));import run as original
read=original.read;sha=original.sha
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def prepare():
    original.verify();assert read(O/'review.json')['passed_saved_data'];assert read(O/'raw_archive_review.json')['passed'];assert not (O/'publication_preflight.json').exists()
    raw=A/'raw_records';manifest=read(raw/'records_manifest.json');stats=collections.Counter();pairs=[]
    rs=read(O/'review.json')['runrows'];lookup={(r['repeat'],r['week'],r['method']):r for r in rs}
    def calls(row):
        folder=O/'runs'/f"r{row['repeat']}__{row['week']}__{row['method']}";out={}
        for p in folder.glob('call[0-9][0-9][0-9].json'):
            c=read(p)
            with np.load(p.with_name(p.stem+'_state.npz')) as z:h=hashlib.sha256(z['base_action'].tobytes()).hexdigest()
            out[c['step'],c['state_hash'],h]=c
        return out
    for seed in [0,1,2]:
        for week in sorted({r['week'] for r in rs}):
            ca=calls(lookup[0,week,f'CONSENSUS_seed{seed}']);cb=calls(lookup[0,week,f'FACTOR_seed{seed}'])
            for k in sorted(set(ca)&set(cb)):
                a,b=ca[k],cb[k];same=a['chosen_pool_id']==b['chosen_pool_id'];delta=a['public_queries']-b['public_queries'];direction='fewer' if delta<0 else 'more' if delta>0 else 'same'
                name=direction+('_same_chosen' if same else '_different_chosen');stats[name]+=1
                def chosen(c):return next(v for v in c['outcomes'] if v['pool_id']==c['chosen_pool_id']) if c['chosen_pool_id'] is not None else None
                ac=chosen(a);bc=chosen(b);item=dict(seed=seed,week=week,step=k[0],same_current_observation_and_base=True,class_name=name,query_delta=delta,consensus_queries=a['public_queries'],factor_queries=b['public_queries'],consensus_chosen=a['chosen_pool_id'],factor_chosen=b['chosen_pool_id'],selected_reward_delta=None if ac is None or bc is None else float(np.float32(ac['reward']))-float(np.float32(bc['reward'])))
                pairs.append(item)
    save(O/'aligned_call_tradeoffs.json',dict(counts=dict(stats),pairs=pairs,scope='Repeat0only,3pairedseeds/all4dates;ONLYsameactualobservation+baseaction. Selectedrewardswereactuallypaidinowntrace,no newprediction. Queriesfewerandsameaction islocalcomputesaving,notnewindependentdateorwholeweekcausalproof.'))
    # Preserve the original recovery report; improve signs in the user-facing text.
    report=(O/'GRID184_report.md').read_text(encoding='utf8');shutil.copy2(O/'GRID184_report.md',O/'GRID184_report_prepublication.md')
    replacements={'候选查询减少-1.03%':'候选查询增加1.03%','总查询减少-0.40%':'总查询增加0.40%','控制时间减少-1.62%':'控制时间增加1.62%','候选查询减少-2.26%':'候选查询增加2.26%','总查询减少-1.14%':'总查询增加1.14%','控制时间减少-1.76%':'控制时间增加1.76%','全步骤P95减少-0.52%':'全步骤P95增加0.52%','搜索步骤P95减少-1.33%':'搜索步骤P95增加1.33%'}
    for a,b in replacements.items():assert a in report,a;report=report.replace(a,b)
    useful=stats['fewer_same_chosen'];waste=stats['more_same_chosen'];different=stats['more_different_chosen']+stats['fewer_different_chosen']+stats['same_different_chosen']
    report+=f'\n\n## 保留有价值的小改善\n\n在第一次计时、全部三种子四日期中，仅配对真实观测和基础动作完全相同的搜索调用：有{useful}个调用中共识查询更少且最终动作相同，也有{waste}个调用查询更多但最终动作相同；{different}个调用选择不同。完整计数、查询增量和双方实际已查询动作的即时奖励差见aligned_call_tradeoffs.json。前者是可以继续研究的具体省查询机会，但这些调用不独立、不能替代整周结果；不因整体组合暂未胜出就删掉这些有利点。\n\n## 原始记录恢复\n\nraw_records/records_manifest.json保留1842个原始文件路径和SHA256。相同字节仅按SHA256去重，10个较大唯一对象由Git LFS保存，其余内容位于small_files.zip；每个唯一对象已经读回验证。restore_records.py可在新建D盘目录完整恢复原路径，不覆盖现有数据。\n'
    (O/'GRID184_report.md').write_text(report,encoding='utf8',newline='\n')
    dest=R/'论文协作资料/FRAME_小闭环结果_GRID184';assert sha(dest/'GRID184_report.md')==sha(O/'GRID184_report_prepublication.md');(dest/'GRID184_report.md').rename(dest/'GRID184_report_prepublication.md');shutil.copy2(O/'GRID184_report.md',dest/'GRID184_report.md')
    for n in ['aligned_call_tradeoffs.json','raw_archive_review.json']:shutil.copy2(O/n,dest/n)
    summary=read(O/'results_summary.json');c=summary['consensus_comparisons'];msg=f"GRID184全64条/32重复轨迹审计通过，关机只中断审计；缓存I/O保持29处assert，复制footer缺shutil仅做发布修复。所有方法3/4整周，3个可判日期2/3达标，5月FULL未完参考受限；3月各方法成本失败。共识对FRAME同等已运行时域时间少{100*c['FRAME_GUARD']['control_reduction']:.2f}%/候选少{100*c['FRAME_GUARD']['candidate_query_reduction']:.2f}%；对同种子F时间多{-100*c['FACTOR']['control_reduction']:.2f}%/候选多{-100*c['FACTOR']['candidate_query_reduction']:.2f}%。保留{useful}个同动作省查询调用与全部反例，不做同整周质量优势结论，不自动加实验。1842原记录按SHA去重+LFS归档可完整恢复，零新增拟合/物理/最终8/论文。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID184真实闭环审计结果\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID184结果已核对\n\n1. 完成LFS/报告/原记录归档上传和远端SHA核对，通知并删除grid184-consensus跟进。\n2. 保留原FRAME上的小计算节省，及同状态同动作省查询调用；结合3月种子0/1和10月种子0的首次分歧，先用已有反馈解释继续查询何时有益/浪费。\n3. 不因局部NET提升宣称整周成本改善，不以全赢门槛否定小改善；当前不自动重训、换规则、扩物理场景/最终8或修改论文。下一步实验需新范围。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID184真实闭环审计\r\n\r\n[报告](protocols/GRID184_search_adapter/final_delivery/GRID184_report.md)；64条核验通过，计算小改善与成本反例分列，全部原记录可恢复。\r\n\r\n'.encode('utf8')+p.read_bytes())
    save(O/'publication_preflight.json',dict(passed=True,checks=read(O/'review.json')['checks'],files=1842,raw_record_manifest_sha256=sha(raw/'records_manifest.json'),no_changed_benchmark_records=True,small_savings_calls=useful,same_action_wasted_queries_calls=waste,new_physics=0,new_fits=0,report_sha256=sha(O/'GRID184_report.md')))
    final=A/'final_delivery';final.mkdir(exist_ok=False)
    for p in [*O.iterdir(),*W.glob('*.py')]:
        if p.is_file():shutil.copy2(p,final/p.name)
    (final/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in final.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    original.verify();print('GRID184_FINAL_PUBLICATION_PREPARED',json.dumps(read(O/'publication_preflight.json')))
if __name__=='__main__':globals()[sys.argv[1]]()
