"""Postfit read-only accounting; no new model, fitting, thresholds or labels."""
from common import *
def main():
    verify();assert read(O/'review.json')['passed'];D=read(O/'design.json');scale=D['target']['gain_scale'];fee=D['target']['cost_per_query']
    tr=read(O/'validation_trajectories.json');res=read(O/'validation_results.json');sm=read(O/'state_metrics.json');old=read(R/'outputs/grid107_direct_value/state_metrics.json')
    by={(r['method'],r['sample_id']):r for r in tr};dates=sorted({r['date'] for r in tr});parts={};families={}
    for arm in ['JOINT','FACTOR']:
        parts[arm]={}
        for base in ['BATCH1','FACTOR' if arm=='JOINT' else 'JOINT']:
            perdate={}
            for date in dates:
                perseed=[]
                for seed in range(3):
                    rr=[r for r in tr if r['method']==f'{arm}_seed{seed}' and r['date']==date]
                    gg=[];qq=[];dd=[]
                    for r in rr:
                        b=by[base if base=='BATCH1' else f'{base}_seed{seed}',r['sample_id']]
                        gg.append((r['gain']-b['gain'])/scale);qq.append(-fee*(r['queries']-b['queries'])/scale);dd.append(r['net_utility']-b['net_utility'])
                    assert abs(np.mean(gg)+np.mean(qq)-np.mean(dd))<1e-12
                    perseed.append(dict(seed=seed,gain_contribution=float(np.mean(gg)),query_fee_contribution=float(np.mean(qq)),net_delta=float(np.mean(dd))))
                perdate[date]=dict(per_seed=perseed,**{k:float(np.mean([s[k] for s in perseed])) for k in ['gain_contribution','query_fee_contribution','net_delta']})
            parts[arm][base]=dict(per_date=perdate,macro={k:float(np.mean([d[k] for d in perdate.values()])) for k in ['gain_contribution','query_fee_contribution','net_delta']})
        families[arm]={k:float(np.mean([res[f'{arm}_seed{s}']['all']['macro'][k] for s in range(3)])) for k in res[f'{arm}_seed0']['all']['macro']}
    metric_compare={}
    for arm,names,source in [('JOINT',[f'JOINT_seed{s}' for s in range(3)],sm),('FACTOR',[f'FACTOR_seed{s}' for s in range(3)],sm),('DIRECT',[f'DIRECT{s}' for s in range(3)],old),('DQN',[f'DQN0_{s}' for s in range(3)],old)]:
        metric_compare[arm]={part:{k:[source[n][part][k] for n in names] for k in ['false_stop','unprofitable_continue','continue_value_rmse']} for part in ['train','validation']}
    save(O/'postfit_accounting.json',dict(passed=True,scope='Read-only decomposition of frozen outputs; no causal attribution, extra fit or criterion change.',
        family_means=families,batch1=res['BATCH1']['all']['macro'],net_decomposition=parts,state_metrics_comparison=metric_compare))
    lines=['# 本轮为什么没有进步：只读核对','',
        '两种结构模型都更少继续查询，但漏掉了更多正收益。JOINT三个种子的验证漏停为20/23/26（41个正收益可达状态），原DIRECT为13/14/13；无利继续减少到120—137次，原DIRECT为283—325次。这是少查与漏益之间的不同取舍，不能只按预测RMSE或查询减少判断成功。','',
        '训练集也存在同样问题：JOINT漏掉72—82个正收益状态（共211），原DIRECT漏掉27—38个。因此不能把问题全部归因于新验证场景；当前网络/固定训练目标和预算下，正收益决策本身也没有充分学好。这里没有隔离出究竟由网络宽度、集合压缩、优化过程还是数据稀疏造成，不能凭结果断言某一个原因。','',
        'JOINT相对FACTOR三个日期平均略好，5月反向较大，且配对种子只有1/3同向。5月JOINT种子0/1收益保留率为0%/10.00%，FACTOR对应5.17%/91.22%；JOINT种子2又恢复到91.84%，说明主要正收益的捕获不稳定。3月552正收益难例JOINT三个种子都停止，配对输入没有解决此前的关键漏停。','',
        '## 净效用差异的精确记账','',
        '将每条完整补查序列相对对照的净效用差写为“收益差/原尺度 − 原费用×查询数差/原尺度”，先按日期平均，再按三个种子与四个日期平均。该分解是固定结果的算术记账，不是长期电网因果归因。','']
    for arm in ['JOINT','FACTOR']:
        r=parts[arm]['BATCH1']['macro'];fm=families[arm]
        lines.append(f"- {arm}相对固定一批：收益部分{r['gain_contribution']:.12f}，少付查询费部分{r['query_fee_contribution']:.12f}，合计{r['net_delta']:.12f}；收益保留率{fm['gain_capture']*100:.2f}%，查询比例{fm['query_fraction']*100:.2f}%（分母为完整同站补查）。")
    r=parts['JOINT']['FACTOR']['macro'];lines += [f"- JOINT相对FACTOR：收益部分{r['gain_contribution']:.12f}，查询费部分{r['query_fee_contribution']:.12f}，合计{r['net_delta']:.12f}。",'',
        '本轮不支持“保留对应关系后，小模型已经更会补查”的结论，也不能否定动作与反馈包含可用信息。GRID111检验的是已查普通奖励的关联，本轮检验的是能否预估超过当前最好动作的新增净收益，两者不是同一问题。当前应先把正收益拟合与停止决策的失败查清，再决定是否改训练或扩大；本轮不自动加试验。']
    (O/'结果解释_中文.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(O/'postfit_accounting_source.json',dict(source=str(W/'diagnose.py'),sha256=sha(W/'diagnose.py'),phase='After all six final fits and audit; no frozen training source modified'))
    print(json.dumps({k:parts[k]['BATCH1']['macro'] for k in parts},indent=2))
if __name__=='__main__':main()
