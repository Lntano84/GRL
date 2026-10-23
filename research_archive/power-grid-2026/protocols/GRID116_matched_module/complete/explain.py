"""Postfit read-only payoff decomposition; no changed criteria, fitting or predictions."""
from common116 import *
def main():
    assert read(O/'review.json')['passed'];assert not (O/'payoff_accounting.json').exists();verify()
    v=read(O/'validation_results.json');d=read(O/'decision.json');D=read(O/'design.json');fee=D['unchanged']['cost_per_query'];scale=D['unchanged']['gain_scale']
    result={}
    for other in ['FACTOR','BASE']:
        records=[]
        for s in range(3):
            j=v[f'BASE_JOINT_seed{s}']['all']['macro'];b=v[f'BASE_FACTOR_seed{s}' if other=='FACTOR' else f'DIRECT{s}']['all']['macro']
            gain=(j['mean_gain']-b['mean_gain'])/scale;charge=-fee*(j['mean_queries']-b['mean_queries'])/scale;delta=j['mean_net_utility']-b['mean_net_utility']
            assert abs(gain+charge-delta)<1e-12
            records.append(dict(seed=s,gain_contribution=gain,query_charge_contribution=charge,net_delta=delta))
        result[other]=dict(records=records,family_mean={k:float(np.mean([r[k] for r in records])) for k in ['gain_contribution','query_charge_contribution','net_delta']})
    families={}
    for k,fmt in [('JOINT','BASE_JOINT_seed{}'),('FACTOR','BASE_FACTOR_seed{}'),('BASE','DIRECT{}')]:
        families[k]={metric:float(np.mean([v[fmt.format(s)]['all']['macro'][metric] for s in range(3)])) for metric in v[fmt.format(0)]['all']['macro']}
    jf=result['FACTOR']['family_mean'];delta_pct={k:100*(d['family_mean_utility'][k]/d['family_mean_utility']['BASE']-1) for k in ['JOINT','FACTOR']}
    save(O/'payoff_accounting.json',dict(passed=True,scope='Postfit arithmetic only; same saved trajectories/fees. No new criteria, labels, decisions, fit or causal attribution.',
        joint_minus=result,family_metric_means=families,relative_mean_gain_over_base_percent=delta_pct,source_sha256=sha(W/'explain.py')))
    text=['# GRID116结果怎么理解','',
        f"这版保留原64宽主干后，配对模块JOINT平均净效用为{d['family_mean_utility']['JOINT']:.8f}，比原BASE高{delta_pct['JOINT']:.2f}%；汇总模块FACTOR为{d['family_mean_utility']['FACTOR']:.8f}，高{delta_pct['FACTOR']:.2f}%。两组都是2/3种子超过原BASE，种子1反向，不能只报平均改善。",'',
        '这说明上一轮把原主干缩窄后得到的负结果，不能直接归为“新增信息没有价值”。但本轮不是单因素宽度对照，不能把恢复全部归因于宽度；它仍不能单独证明新增结构信息是平均改善的唯一原因。','',
        '本轮对配对机制的答案更直接：同参数、同初始化、同抽样下，JOINT只有1/3种子胜过FACTOR，按种子平均后的四个日期都低于FACTOR。预设机制判据未通过。当前不支持“保留对应关系比先汇总更好”的主张。','',
        f"两者差距主要来自多花的查询费。JOINT相对FACTOR，平均收益部分贡献{jf['gain_contribution']:+.8f}，查询费用部分{jf['query_charge_contribution']:+.8f}，合计{jf['net_delta']:+.8f}。其中种子1两组获得的奖励增量几乎相同，JOINT却进行了更多查询。该分解只是固定搜索序列的记账，不是整周运行因果分析。",'',
        f"查询比例（相对完整同站补查）的种子/日期宏平均：JOINT {families['JOINT']['query_fraction']*100:.2f}%，FACTOR {families['FACTOR']['query_fraction']*100:.2f}%，原BASE {families['BASE']['query_fraction']*100:.2f}%。收益保留率对应为{families['JOINT']['gain_capture']*100:.2f}%、{families['FACTOR']['gain_capture']*100:.2f}%、{families['BASE']['gain_capture']*100:.2f}%。这描述搜索收益与查询取舍，不代表真实在线提速或整周质量。",'',
        '固定配对扰动在验证正收益状态上仍未改变任何一个种子的继续/停止判断，当前配对通道的关键决策作用依然较弱；一次扰动不能证明完全不用配对信息。第552步两组都是[停、继续、继续]，第108步两组都继续，不能凭修复一个难例就宣布整体机制成立。','',
        '我的判断：保留原主干的这版设置值得作为开发记录保留，FACTOR也可作为后续学习对照；现阶段不能把配对结构写成已证实贡献。先不扩大、不上真实闭环，也不追加网络变体救结果。若继续研究，最具体的入口是原样核对种子1为什么增加无利查询，以及这些误判能否由已有公开输入区分；训练目标或新结构的改变应另定方案。']
    (O/'结果判读_中文.md').write_text('\n'.join(text)+'\n',encoding='utf8');verify();print(json.dumps(dict(relative_gain=delta_pct,joint_minus_factor=jf),indent=2),flush=True)
if __name__=='__main__':main()
