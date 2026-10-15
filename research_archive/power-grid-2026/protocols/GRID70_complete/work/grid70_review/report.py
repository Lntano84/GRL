"""Read-only report and cross-batch identical-state feedback checks."""
from pathlib import Path
import csv,gzip,hashlib,json,math,shutil,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid70'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    D=read(OUT/'design.json');A=read(OUT/'review.json');S=read(OUT/'finished.json')['summaries']
    assert A['passed_saved_data'] and len(S)==len(D['weeks'])==13
    assert {s['week'] for s in S}==set(D['weeks']) and not (OUT/'RUN.lock').exists()
    assert sum(s['steps'] for s in S)<=D['caps']['physical_steps']
    assert sum(s['public_forecasts'] for s in S)<=D['caps']['public_forecasts']
    source={};old={}
    for stage in [65,68]:
        for s in read(ROOT/f'outputs/grid{stage}/finished.json')['summaries']:
            if s['repeat']==0 and s['method'] in ['FULL','FRAME']:
                source[s['week'],s['method']]=ROOT/f"outputs/grid{stage}/runs/r0__{s['week']}__{s['method']}"
                if s['method']=='FRAME':old[s['week']]=s
    matched_states=matched_queries=0;details=[]
    for s in S:
        fdir=source[s['week'],'FULL'];full=read(fdir/'summary.json');truth={}
        for p in sorted(fdir.glob('call[0-9][0-9][0-9].json')):
            c=read(p)
            with np.load(p.with_name(p.stem+'_state.npz')) as z:key=(digest(z['observation']),digest(z['base_action']))
            assert key not in truth;truth[key]={r['pool_id']:r for r in c['outcomes']}
        folder=OUT/'runs'/f"r0__{s['week']}__{s['method']}";guard_calls=local_calls=unchanged=0
        for j,p in enumerate(sorted(folder.glob('call[0-9][0-9][0-9].json'))):
            c=read(p);guard_calls+=int(c['verification_guard']);local_calls+=int(c['local_queries']>0)
            unchanged+=int(c['verification_guard'] and c['pre_extra_choice']==c['shortlist_choice'])
            with np.load(folder/f'call{j:03d}_state.npz') as z:key=(digest(z['observation']),digest(z['base_action']))
            if key not in truth:continue
            matched_states+=1
            for r in c['outcomes']:
                t=truth[key][r['pool_id']]
                assert all(r[k]==t[k] for k in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous'])
                matched_queries+=1
        original=old[s['week']];complete=s['complete'] and full['complete'];old_complete=original['complete'] and full['complete']
        survive=s['complete'] if full['complete'] else s['steps']>=full['steps']
        delta=s['cost']/full['cost']-1 if complete else None
        old_delta=original['cost']/full['cost']-1 if old_complete else None
        details.append(dict(date=s['week'],full_steps=full['steps'],old_steps=original['steps'],new_steps=s['steps'],
            full_complete=full['complete'],old_complete=original['complete'],new_complete=s['complete'],
            old_cost_delta_full=old_delta,new_cost_delta_full=delta,
            old_quality=bool((original['complete'] if full['complete'] else original['steps']>=full['steps']) and (old_delta is None or old_delta<=.005)),
            new_quality=bool(survive and (delta is None or delta<=.005)),
            full_queries=full['n1_forecasts'],old_queries=original['n1_forecasts'],new_queries=s['n1_forecasts'],
            new_query_saving=1-s['n1_forecasts']/full['n1_forecasts'] if s['steps']==full['steps'] and s['complete']==full['complete'] and full['n1_forecasts'] else None,
            controller_s=s['controller_s'],controller_cpu_s=s['controller_cpu_s'],
            guarded_calls=guard_calls,unchanged_guard_incumbents=unchanged,completion_calls=local_calls))
    for x,y in zip(details,A['dates']):
        assert x['date']==y['date'] and x['new_quality']==y['quality_pass'] and x['new_cost_delta_full']==y['cost_delta_full']
    q=[r['new_query_saving'] for r in details if r['new_query_saving'] is not None]
    result=dict(passed=True,physical_runs=len(S),date_count=13,old_quality=sum(x['old_quality'] for x in details),
        new_quality=sum(x['new_quality'] for x in details),full_complete=sum(x['full_complete'] for x in details),
        old_complete=sum(x['old_complete'] for x in details),new_complete=sum(x['new_complete'] for x in details),
        max_complete_cost_delta=max(x['new_cost_delta_full'] for x in details if x['new_cost_delta_full'] is not None),
        macro_query_saving=statistics.mean(q),query_ratio_dates=len(q),
        cross_reference_states=matched_states,cross_reference_feedback=matched_queries,
        all_dates_preserved=True,new_training=0,independent_ac_resolve=False,
        timing='Single repair timing observations retained; no matched fresh FULL timing experiment, no speed claim from historical timing ratios.')
    save(OUT/'supplementary_review.json',result)
    with (OUT/'complete_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(details[0]));w.writeheader();w.writerows(details)
    pct=lambda x:'—' if x is None else f'{100*x:+.3f}%'
    lines=['# FRAME提前返回修复结果（GRID69—70）','',
        '本轮按作者要求修复提前返回，保留原版、第一版修复和所有不利结果。原环境、排序、控制器、缓存及模型均不变；没有训练。',
        '',f"第二版修复在全部13个已有日期中质量通过{result['new_quality']}/13；完整周{result['new_complete']}/13，FULL为{result['full_complete']}/13，原FRAME为{result['old_complete']}/13。共同完整周最坏成本变化{pct(result['max_complete_cost_delta'])}。",'',
        f"在{len(q)}个等时域且FULL实际执行搜索的日期中，逐日期等权查询减少{statistics.mean(q)*100:.3f}%。未执行搜索的日期不计查询比例，提前结束不作为省时收益。单次控制时间全部保留，修复版尚无同批平衡重复的整套控制器加速结论。",'',
        '## 具体改动','',
        '原版遇到128候选中的一步安全动作会直接返回。第二版对此增加原有64候选批次的核查，至少256候选且连续两批无改进后才退出。追加发现更好动作才对其所属变电站补全；未发现改进时保留原安全动作。初始前缀不安全的回退与补全精确沿用原版，前缀反馈缓存复用且不重复查询。',
        '', '## 全部日期','',
        '|日期|FULL/原版/修复版步数|原版成本变化|修复版成本变化|FULL/修复版查询|查询减少|质量筛查|',
        '|---|---:|---:|---:|---:|---:|---|']
    for x in details:lines.append(f"|{x['date']}|{x['full_steps']}/{x['old_steps']}/{x['new_steps']}|{pct(x['old_cost_delta_full'])}|{pct(x['new_cost_delta_full'])}|{x['full_queries']}/{x['new_queries']}|{pct(x['new_query_saving'])}|{'通过' if x['new_quality'] else '未通过'}|")
    lines += ['', '质量筛查沿用原0.5%成本尺度及不比FULL更早失败的条件。该尺度用于投资筛查；0.64%可同时如实描述为小幅成本增加，不改写GRID68旧裁决。未完成周不报完整周成本变化。',
        '', '## 第一版修复为何没有采用','',
        'GRID69统一延后退出并对所有安全前缀开启补全：七月恢复完整周，十二月成本略低于FULL，但三月成本增加1.374%。首次退步发生于原安全动作未被核查改进、仍新触发同站补全时。该版6条运行、审计与错误方向均留档，未继续扩大。GRID70将改动收窄后按同一规则评价全部13日期，没有逐日期挑选版本。',
        '', '## 核验与可复现边界','',
        f"GRID70保存向量、逐步成本账目、合法反馈、付费查询与策略重放通过；另从历史FULL在相同观测和基动作下逐项复核{matched_queries}个反馈（{matched_states}个状态）。这不是独立AC重求解。原策略308个搜索调用精确复现，隐藏后缀扰动和分支手算检查通过。",'',
        '全部日期已用于开发，修复结果属于已暴露日期的回归。旧实验和阈值保持原样。两版修复不按日期拼接，最终版本始终使用同一策略。完整控制时间需要在最终版本定型后同批计时，当前原始秒数不替代该检验。',
        '', '## 论文主线与创新性材料','',
        '主线聚焦反馈驱动的拓扑搜索计算分配：先检查高优先级候选，利用追加反馈控制退出，将必要预算分配到结构邻域。新修复明确区分了“已有一步安全动作”和“退出前已有足够核查”，也限制了无新信息时额外改变动作。',
        '', '可直接使用的证据包括全部日期的质量与完成情况、昂贵预测查询数量、原版与两版修复的机制差异。GNN继续作为可替换辅助；本轮未开发新GNN/RL。搜索、元推理和局部邻域已有成熟研究，论文应具体说明电网动作结构、合法反馈与在线控制集成带来的差异，不以通用模块组合本身宣称首创。',
        '', '正文呈现核心方法、完整质量摘要与查询结果；逐日期表、完整敏感性及工程审计放附录。保留原版不利案例作为修复动机，不用删日期改善表观结论。']
    (OUT/'GRID70_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    package=ROOT/'论文协作资料/FRAME提前返回修复_GRID69-70';package.mkdir(parents=True,exist_ok=True)
    for p in [OUT/'GRID70_report.md',OUT/'complete_comparison.csv',OUT/'supplementary_review.json',OUT/'review.json',OUT/'design.json',OUT/'preflight.json',ROOT/'outputs/grid69/pilot_review.json',ROOT/'outputs/grid69/march_regression_diagnostic.json']:
        target=package/('第一版_'+p.name if 'grid69' in p.parts else p.name);shutil.copy2(p,target)
    for stage in [69,70]:
        for p in (ROOT/f'work/grid{stage}').glob('*.py'):
            target=package/f'实现/grid{stage}'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (package/'先读.md').write_text('先读GRID70_report.md。原稿和历史记录未覆盖。完整原始回放向量在 D:/Codex/GRL研究/outputs/grid69 与 grid70/runs；本文件夹提供轻量结果与实现。修复版时间仅原始记录，不作同批加速声明。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
