"""Before any physics, make repeat two the exact reverse of repeat one."""
import copy,json,shutil
from run import R,W,O,read,save,sha,verify,archive
def main():
    verify();assert read(O/'offline_review.json')['passed'];assert not (O/'physical_authorization.json').exists() and not (O/'benchmark_started.json').exists() and not (O/'runs').exists()
    assert not (O/'readiness_review.json').exists();old=read(O/'design.json');before=sha(O/'design.json');data=read(O/'data_freeze.json');code=read(O/'code_freeze.json')
    # The frozen/qualified archive retains the original proposal verbatim.
    assert sha(R/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID184_search_adapter/qualified/design.json')==before
    new=copy.deepcopy(old);first=[v for v in old['execution_order'] if v['repeat']==0];assert len(first)==32
    new['execution_order']=first+[dict(v,repeat=1) for v in reversed(first)];new['execution_order_construction']='First32interleaveddate/armentries,thenstrictfullreverse32; correctedbeforeanyphysicalrun,notbasedonresults.'
    assert {(v['repeat'],v['week'],v['method']) for v in old['execution_order']}=={(v['repeat'],v['week'],v['method']) for v in new['execution_order']}
    for k in old:
        if k!='execution_order':assert old[k]==new[k]
    save('design.json',new);data['outputs/grid184_search_adapter/design.json']=sha(O/'design.json');save('data_freeze.json',data)
    code[(W/'finalize_preparation.py').relative_to(R).as_posix()]=sha(W/'finalize_preparation.py');save('code_freeze.json',code);verify()
    save('protocol_order_receipt.json',dict(passed=True,reason='Ensurestrictreversetiming,notmerelyreverserotatedperdateordering',before_design_sha256=before,after_design_sha256=sha(O/'design.json'),changed_only_execution_order=True,all64identities_unchanged=True,all_previously_qualified_code_unchanged=True,new_physics=0,new_fits=0,previousproposal_preserved_in='frozen/andqualified/archive'))
    report=(O/'GRID184_preparation_report.md').read_text(encoding='utf8')+'\n顺序复核补充：第二次32条严格采用第一次32条的完整逆序。该调整在任何物理回放前完成，64条身份、日期、模型、参数和控制器不变；原提案及哈希保留在frozen/qualified归档。\n'
    (O/'GRID184_preparation_report_v2.md').write_text(report,encoding='utf8',newline='\n')
    (O/'HANDOFF.md').write_text('# GRID184待用户决定新物理范围\n\n182六次完整训练小模型拟合及三旧F权重复用、183输入/补查核心、184独立模型与Search.choose保存状态资格均已通过。最新方案design.json经protocol_order_receipt.json将第二重复改为第一32条严格逆序，全部64身份与参数不变，旧案已保存。\n\n当前没有physical_authorization.json、benchmark_started.json或runs目录，不能启动pipeline/worker。用户确认具体64条范围后，才可记录真实用户授权及当前design SHA，使用work/grid02/.venv/Scripts/python.exe启动唯一pipeline.py。不得重训/生成/恢复旧运行器。\n\n四个已暴露开发日期2037-03-02/05-04/08-03/10-19，各scenario100。FRAME/BATCH1/F三种子/共识三种子，64串行，总窗口4h、每条1200s，保留部分输出与全部不利结果。首条计划内FRAME承担实际接口检查；尚无真实控制器/求解器资格运行，发生工程故障先保留再核对，不能当方法失败。最终8不生成，无论文改动。\n\nauthorization_guard_stderr.log是有意验证缺少授权时阻止worker的异常，只是资格测试，不是实验故障。offline_review不是整周质量结果。训练集工程前向不当新泛化成绩。\n',encoding='utf8',newline='\n')
    save('readiness_review.json',dict(passed=True,offline_model_rows=32238,offline_search_calls=1776,strict_reverse_execution_order=True,physical_authorization_absent=True,physical_runs_started=False,worker_guard_passed=read(O/'authorization_guard_test.json')['passed'],upstream_archives_passed=read(O/'upstream_archive_review.json')['passed'],original_qualified_code_unchanged=True,current_design_sha256=sha(O/'design.json'),reviewable_scripts=['worker.py','pipeline.py','audit.py']))
    dest=R/'论文协作资料/FRAME_小闭环准备_GRID184'
    for n in ['GRID184_preparation_report_v2.md','HANDOFF.md','protocol_order_receipt.json','readiness_review.json']:shutil.copy2(O/n,dest/n)
    # Preserve the original collaboration design rather than replacing a draft.
    shutil.copy2(O/'design.json',dest/'design_strict_reverse.json');archive('readiness');print('GRID184_READY_FOR_SCOPE_DECISION')
if __name__=='__main__':main()
