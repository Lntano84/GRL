"""All five seeds reported, quality eligibility separated from speed."""
from pathlib import Path
from collections import Counter
import json,statistics,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid84'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    R=read(OUT/'review.json');D=read(OUT/'design.json');assert R['passed_saved_data'];decisions=[];behavior=[]
    for c in R['comparisons']:
        eligible=R['guard_quality_dates']==6 and c['quality_dates']==6
        category='GUARD_QUALITY_SCREEN_FAILED' if R['guard_quality_dates']!=6 else 'GUARD_PASS_META_QUALITY_FAIL' if c['quality_dates']!=6 else 'SUPPORTED_EQUAL_QUALITY_SPEED' if c['stable_5percent_screen'] else 'BOTH_PASS_TIMING_MIXED_OR_SMALL'
        decisions.append(dict(seed=int(c['baseline'][-1]),baseline=c['baseline'],guard_quality_dates=R['guard_quality_dates'],baseline_quality_dates=c['quality_dates'],equal_quality_eligible=eligible,same_quality_speed_supported=bool(eligible and c['stable_5percent_screen']),category=category,macro_controller_reduction=c['macro_controller_reduction'],repeat_macro_reductions=c['repeat_macro_reductions']))
        files=sorted((OUT/'runs').glob(f"r0__*__{c['baseline']}/call[0-9][0-9][0-9].json"));first=Counter();all_events=Counter();stops=Counter()
        for p in files:
            x=read(p);ev=x.get('meta_events',[]);first[ev[0]['action'] if ev else 'exhausted_before_decision']+=1
            all_events.update(e['action'] for e in ev);stops[x['stop_reason']]+=1
        behavior.append(dict(seed=int(c['baseline'][-1]),calls=len(files),first_actions={str(k):v for k,v in first.items()},all_actions={str(k):v for k,v in all_events.items()},delegate_first=first[2],delegate_share=first[2]/len(files) if files else None,stops=dict(stops)))
    a=[d['macro_controller_reduction'] for d in decisions];all_quality=all(d['equal_quality_eligible'] for d in decisions)
    overall=dict(all_five_equal_quality_eligible=all_quality,mean_seed_macro_controller_reduction=statistics.mean(a),median_seed_macro_controller_reduction=statistics.median(a),min_seed_macro_controller_reduction=min(a),max_seed_macro_controller_reduction=max(a),per_seed_speed_passes=sum(d['same_quality_speed_supported'] for d in decisions))
    overall['stronger_five_seed_support']=bool(all_quality and statistics.mean(a)>=.05 and overall['per_seed_speed_passes']>=3)
    result=dict(overall=overall,decisions=decisions,new_fits=3,all_seeds=[0,1,2,3,4],new_seeds=[2,3,4],old_costindex_fixed=2,cost_level=0,calibration_reselected=False,method='FRAME_GUARD',scope=D['scope'])
    save('decision.json',result);save('behavior.json',behavior)
    train=read(OUT/'training_finished.json')
    lines=['# GRID84：固定费用配置的五种子同批比较','',
           '保留原模型种子0/1，只新增2/3/4各一次拟合。六个已暴露生成场景，FRAME_GUARD与五个适配模型各两次反向计时，共72条串行回放。原GRID82结果保持原样。','',
           '| 方法 | 质量通过 | 完成整周 | 搜索查询合计¹ | 完整控制秒/场景² | 最大可比完整周成本增幅 |','|---|---:|---:|---:|---:|---:|']
    for m in D['methods']:
        gs=[g for g in R['groups'] if g['method']==m['method']];cost=[g['cost_delta_full'] for g in gs if g['cost_delta_full'] is not None]
        cs='不适用' if not cost else f'{max(cost)*100:+.4f}%'
        lines.append(f"|{m['method']}|{sum(g['quality_pass'] for g in gs)}/6|{sum(g['complete'] for g in gs)}/6|{sum(g['queries'] for g in gs)}|{statistics.mean(g['controller_s'] for g in gs):.3f}|{cs}|")
    lines+=['','¹每场景仅计一次查询。²各日期先平均两次计时再平均；时间相对变化另按日期比例宏平均。','',
            '| FRAME相对种子 | 控制时间宏平均减少 | 查询宏平均减少³ | 正向日期 | 两次计时分别 | 同质量提速判据 |','|---|---:|---:|---:|---|---|']
    for c,d in zip(R['comparisons'],decisions):
        q=c['macro_query_reduction_equal_horizons'];qs='不适用' if q is None else f'{q*100:+.3f}%'
        lines.append(f"|{d['seed']}|{c['macro_controller_reduction']*100:+.3f}%|{qs}|{c['positive_dates']}/6|"+' / '.join(f'{v*100:+.3f}%' for v in c['repeat_macro_reductions'])+f"|{d['category']}|")
    lines+=['','³只含同终止时域且对照搜索非零日期，完整分母见review.json。', '',
            f"五种子控制时间变化均值{statistics.mean(a)*100:+.3f}%，中位数{statistics.median(a)*100:+.3f}%，范围[{min(a)*100:+.3f}%, {max(a)*100:+.3f}%]；逐种子同质量提速通过{overall['per_seed_speed_passes']}/5，冻结整体加强判据：{overall['stronger_five_seed_support']}。",'',
            '该均值描述训练种子的变化，六日期仍为六个场景，不将种子或计时重复当成新增独立场景。质量不合格时不宣称全种子同质量提速。','',
            '## 实际动作使用','', '| 模型种子 | 搜索调用 | 首个决策委托FRAME | 比例 |','|---|---:|---:|---:|']
    for b in behavior:
        v='不适用' if b['delegate_share'] is None else f"{b['delegate_share']*100:.1f}%"
        lines.append(f"|{b['seed']}|{b['calls']}|{b['delegate_first']}|{v}|")
    lines+=['','只统计重复0；不同闭环路径的调用不是同状态配对样本。委托宏动作包含FRAME_GUARD本身，不能把委托后的收益全部归于新增学习策略。','',
            '## 训练与校准诊断','', f"三次新增拟合墙钟合计{train['wall_s']:.1f}秒，均为最终检查点。129训练、22校准记录沿用GRID77，未新增教师模拟。",'',
            '| 新种子 | 校准查询均值 | 校准平均遗憾 | 校准最大遗憾 | 相对固定核查校准质量匹配 |','|---|---:|---:|---:|---|']
    for t in train['results']:
        ok=t['calibration_mean_regret']<=t['guard_mean_regret']+1e-7 and t['calibration_max_regret']<=t['guard_max_regret']+1e-7
        lines.append(f"|{t['seed']}|{t['calibration_queries']:.3f}|{t['calibration_mean_regret']:.6g}|{t['calibration_max_regret']:.6g}|{ok}|")
    lines+=['','费用固定为历史costindex2，cost_level=0、beta=0；原GRID77共同校准质量匹配False保留。新增校准只诊断，不据此选费用、种子或检查点。本轮检验现有增强电网适配的随机性，不评价最优成本敏感元控制。','',
            '## 逐日期与计时','', '| 场景 | 方法 | 完成/步数 | 完整周成本相对FULL | 质量通过 | 查询 | 控制秒均值[最小,最大] | P95秒 |','|---|---|---|---:|---|---:|---:|---:|']
    for w in D['weeks']:
      for m in D['methods']:
        g=next(g for g in R['groups'] if g['date']==w and g['method']==m['method']);r=next(r for r in R['rows'] if r['date']==w and r['method']==m['method']);delta=g['cost_delta_full'];cs='未双方完成' if delta is None else f'{delta*100:+.4f}%'
        lines.append(f"|{w}|{m['method']}|{g['complete']}/{r['steps']}|{cs}|{g['quality_pass']}|{g['queries']}|{g['controller_s']:.3f}[{g['controller_min_s']:.3f},{g['controller_max_s']:.3f}]|{g['decision_p95_s']:.4f}|")
    lines+=['','## 审计和解释','',
            '保存向量、合法因果反馈、逐步成本、预测查询、两次重复轨迹和互斥计时分解均检查；FRAME与旧种子0/1需逐步重现GRID82，不独立AC重求解。所有失败和较差结果保留。',
            'FULL仅引用冻结GRID81的质量/存活/完整周成本，不使用其旧计时计算本轮提速。实际时间比较全部来自本轮同批三类以上模型，DC/safeQP与公共元数据缓存由各臂共享。',
            '会议对照为ICAPS2022作者DoubleDQN内核的任务适配，保留返回、追加64和完整FRAME委托；没有完整复现AWA*/RRT*任务。当前六日期已经暴露，本轮为开发比较。']
    (OUT/'GRID84_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETE:three new fits,72 runs and saved-data audit/report complete; all five seeds retained.\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
