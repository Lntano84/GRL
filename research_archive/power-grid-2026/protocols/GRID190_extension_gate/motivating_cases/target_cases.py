"""Post hoc match to the motivating GRID188 cases,not new evaluation data."""
from pathlib import Path
import shutil,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def main():
    C.trained();O=C.O;R=C.R;assert not (O/'motivating_cases.json').exists();p=R/'outputs/grid188_repeated_cases/unique_divergence_states.json';old=C.read(p);tr=C.read(O/'oof_trajectories.json');by={(v['method'],v['sample_id']):v for v in tr};restore={v['sample_id']:v for v in old if v['category']=='RESTORED'};lost={v['sample_id']:v for v in old if v['category']=='LOST'};details=[]
    for sid,v in restore.items():
        seeds=[]
        for s in range(3):
            n=by[f'NET_seed{s}',sid];g=by[f'GATE_seed{s}',sid];assert not n['action_match'];full=g['action_match'];seeds.append(dict(seed=s,restored_full_local_action=full,appended=g['append'],added_gain=g['gain']-n['gain'],added_queries=g['queries']-n['queries']))
        details.append(dict(sample_id=sid,group=v['physical_state_group'],date=v['date'],seeds=seeds))
    protected=[]
    for sid,v in lost.items():
        for s in range(3):
            n=by[f'NET_seed{s}',sid];g=by[f'GATE_seed{s}',sid];assert n['action_match'] and g['action_match'];assert g['gain']>=n['gain'];protected.append(dict(sample_id=sid,seed=s,group=v['physical_state_group']))
    anycase=[v for v in details if any(x['restored_full_local_action'] for x in v['seeds'])];allcase=[v for v in details if all(x['restored_full_local_action'] for x in v['seeds'])]
    rec=dict(posthoc=True,source_sha256=C.sha(p),old_restore_pairs=72,newly_restored_pairs=sum(x['restored_full_local_action'] for v in details for x in v['seeds']),restored_unique_starts=len(anycase),restored_groups=len({v['group'] for v in anycase}),all3restored_starts=len(allcase),all3restored_groups=len({v['group'] for v in allcase}),prior_lost_pairs_preserved=len(protected),details=details,protected=protected,new_fits=0,new_model_forwards=0,new_physics=0)
    C.save('motivating_cases.json',rec)
    text=f'''# GRID190：对最初改善与退化案例的复核

这是主训练、全量折外结果完成后的同案例匹配，不是新增独立实验。原GRID188的24个重复恢复起点/8个物理组，在本轮对应72个种子—起点配对。

本轮学习追加找回其中{rec['newly_restored_pairs']}/72个配对的完整同站动作，涉及{rec['restored_unique_starts']}个起点、{rec['restored_groups']}个物理组；其中{rec['all3restored_starts']}个起点/{rec['all3restored_groups']}个物理组在全部三个种子中均恢复。其余未恢复案例继续保留，不据此改阈值。

原GRID188的28个重复退化起点/17个物理组，共84个配对，在本轮均保持原NET的完整同站动作。这个局部保护主要来自原NET冻结、不能提前删除其查询的结构，不能单独当作门控学得的能力，也不等于整周成本保证。

因此，这次既没有重新引入原纯二元替换的这些局部退化，也追回了部分原有小改善。总成绩仍按全部16日期/3种子汇报，不只按这些目标案例算。

主报告中的全部局部改选为84个配对/61个起点/29个物理组；这里的{rec['newly_restored_pairs']}个是与原24个起点重合的子集，两个数字不能混为一谈。收益幅度与额外查询仍见完整报告，尚无新的物理闭环。
'''
    (O/'原改善与退化案例复核.md').write_text(text,encoding='utf8',newline='\n');C.archive('motivating_cases');dest=R/'论文协作资料/FRAME_NET停止后学习追加_GRID190'
    for n in ['motivating_cases.json','原改善与退化案例复核.md']:shutil.copy2(O/n,dest/n);assert C.sha(O/n)==C.sha(dest/n)
    msg=f"GRID190事后原案例复核：原24恢复起点/8组的72配对找回{rec['newly_restored_pairs']}，{rec['restored_unique_starts']}起点/{rec['restored_groups']}组；全3种子恢复{rec['all3restored_starts']}起点/{rec['all3restored_groups']}组。原28退化起点/17组的84配对全保留NET局部完整动作，主要是结构保护而非学得安全。全部成绩不筛案例；仍有额外查询和幅度识别不足，不新训练或物理。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n### GRID190原案例复核\n\n'+msg+'\n',encoding='utf8',newline='\n')
    print('GRID190_CASES',C.json.dumps({k:v for k,v in rec.items() if k not in ['details','protected']}))
if __name__=='__main__':main()
