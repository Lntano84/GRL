"""Independent saved-trajectory checks and complete first-prototype report."""
import collections,csv,json,pickle,shutil
import numpy as np
from common import ROOT,W,OUT,ARCHIVE,read,save,sha,verify,test_sealed


def main():
    verify();test_sealed();D=read(OUT/'design.json');results=read(OUT/'validation_results.json')
    with (OUT/'cases.pkl').open('rb') as f:cases={c['sample_id']:c for c in pickle.load(f)}
    rows=read(OUT/'validation_trajectories.json');assert len(rows)==7*590
    checks=0
    for r in rows:
        c=cases[r['sample_id']];assert c['partition']=='validation'
        position=0;done=False;utility=0.;seen=set()
        for action in r['actions']:
            assert not done and action in (0,1)
            if action==0:done=True
            else:
                assert position<len(c['queries'])
                utility+=(float(c['gains'][position])-D['cost_per_query']*int(c['queries'][position]))/D['gain_scale']
                position+=1;done=position==len(c['queries'])
        assert done and position==r['batches']
        assert int(c['queries'][:position].sum())==r['queries']
        assert float(c['gains'][:position].sum())==r['gain']
        assert abs(utility-r['net_utility'])<1e-9
        assert c['chosen'][position]==r['chosen_pool_id']
        upper=max([0.]+[(c['gains'][:j].sum()-D['cost_per_query']*c['queries'][:j].sum())/D['gain_scale'] for j in range(1,len(c['queries'])+1)])
        assert abs(upper-r['oracle_utility'])<1e-9 and utility<=upper+1e-9
        checks+=1
    for method in ['STOP','FULL_LOCAL','ORACLE','COST_LOGISTIC','DQN0','DQN1','DQN2']:
        subset=[r for r in rows if r['method']==method]
        assert len(subset)==590
        date_values=[np.mean([r['net_utility'] for r in subset if r['date']==d]) for d in sorted({r['date'] for r in subset})]
        assert abs(float(np.mean(date_values))-results[method]['macro']['mean_net_utility'])<1e-12
    review=dict(passed=True,trajectory_checks=checks,date_macro_checks=7,model_fits=4,seeds=D['seeds'],
        fixed_final_checkpoints=True,new_physical_calls=0,final_test_materialized=False,
        scope='Saved-search policy validation; not independent physical AC re-solving or week-level control.')
    save(OUT/'review.json',review)
    lines=['# GRID102 成本感知补查：小型DQN首轮','',
       '使用现有FULL教师反馈搭建逐批揭示的搜索环境。全局前缀、初始补查站及站内候选顺序固定；每次最多查询64个候选，策略选择停止或继续。没有新的电网回放。',
       '',f"训练{D['cases']['train']}个起始前缀、16日期；验证{D['cases']['validation']}个起始前缀、4日期。多个前缀及局部状态相关，不是新增独立电网场景。41项输入只含已支付反馈与公开结构，37项继承GRID101，4项表达补查进度。",
       '', f"奖励为（最佳已观察奖励增量－每查询成本×查询数）/增益尺度。每查询成本{D['cost_per_query']:.10g}、尺度{D['gain_scale']:.10g}均只从训练数据确定；成本定价是启发式查询代理，不是实测秒数或货币。没有按验证结果挑成本系数。",
       '', 'DQN使用现有Stable-Baselines3实现，网络两层各64节点；种子0/1/2分别训练30000步，均采用最终检查点，不选最好种子。成本逻辑回归使用训练轨迹事后最优停止标签，具有相同状态、动作与成本；ORACLE是已知全部未来的上界，不是可部署方法。',
       '', '## 验证结果：4日期宏平均','',
       '| 策略 | 标准化净效用 | 查询占完整补查 | 即时增益覆盖 | 立即停止比例 |',
       '|---|---:|---:|---:|---:|']
    for method in ['STOP','FULL_LOCAL','COST_LOGISTIC','DQN0','DQN1','DQN2','DQN_SEED_MEAN','ORACLE']:
        r=results[method]['macro'];lines.append(f"| {method} | {r['mean_net_utility']:.5f} | {100*r['query_fraction']:.2f}% | {100*r['gain_capture']:.2f}% | {100*r['stop_immediately']:.2f}% |")
    lines+=['','净效用越高越好；查询比例相对查完该初始站所有剩余候选，不是相对整个电网控制器。这里是实际重放的搜索策略查询数，而非GRID101前20%记录排序，但仍不等于闭环实测控制时间或整周质量。',
      '', '## 逐日期净效用','']
    for date in sorted(results['STOP']['per_date']):
        lines.append('- '+date+': '+'; '.join(f"{method}={results[method]['per_date'][date]['mean_net_utility']:.5f}" for method in ['FULL_LOCAL','COST_LOGISTIC','DQN0','DQN1','DQN2','ORACLE']))
    rl=results['DQN_SEED_MEAN']['macro']['mean_net_utility'];simple=results['COST_LOGISTIC']['macro']['mean_net_utility']
    lines+=['','## 判读','',
      (f'三个DQN种子平均净效用{rl:.5f}，高于成本逻辑回归{simple:.5f}。' if rl>simple else f'三个DQN种子平均净效用{rl:.5f}，未超过成本逻辑回归{simple:.5f}。'),
      '只按本轮固定查询定价和四日期开发验证判读，不外推其他成本系数、独立测试或整周运行。全部种子、学习曲线、逐前缀轨迹、模型和基线保留；没有追加训练救结果。',
      '', '当前学习控制的是单个物理状态内的补查深度，不控制下个电网时刻，也没有学习整周后果。只有在学习策略接入后续独立闭环并计入特征/推理开销后，才能判断实际应用价值。最终8测试场景仍未生成，本轮结束不自动追加神经网络、第二电网或新成本实验。']
    (OUT/'GRID102_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    with (OUT/'validation_trajectories.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dest=ARCHIVE/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in OUT.rglob('*'):
        if p.is_file():
            q=dest/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for p in W.glob('*.py'):shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
    collab=ROOT/'论文协作资料/FRAME_补查DQN首轮_GRID102_20261022';collab.mkdir(exist_ok=False)
    for name in ['GRID102_report.md','design.json','review.json','validation_results.json','validation_trajectories.csv']:
        shutil.copy2(OUT/name,collab/name)
    print(json.dumps(review,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
