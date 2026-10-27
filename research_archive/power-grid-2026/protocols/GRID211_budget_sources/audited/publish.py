"""Independent contribution/cross-seed audit and source interpretation."""
from pathlib import Path
import collections,json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,read=C.R,C.O,C.read
def main():
    C.verify();assert read(O/'complete.json')['passed'] and not (O/'independent_audit.json').exists();new={(r['method'],r['sample_id']):r for r in read(C.NEW/'oof_trajectories.json')};old={(r['method'],r['sample_id']):r for r in read(C.OLD/'oof_trajectories.json')};rows=read(O/'all7134_contributions.json');task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];budget=read(C.NEW/'expected_budget_comparison.json')['GRID208'];checked=0;repeats=collections.defaultdict(list)
    for r in rows:
        s,sid=r['seed'],r['sample_id'];a=new[f'GATE_seed{s}',sid];b=old[f'GATE_seed{s}',sid];n=new[f'NET_seed{s}',sid];costrow=next(p for p in budget['per_date_seed'] if p['seed']==s and p['date']==r['date']);tn,to=costrow['NEW']['keep_probability'],costrow['OLD']['keep_probability'];w=r['weight'];directraw=w*(a['net_utility']-b['net_utility']);directmatched=w*(tn*(a['net_utility']-n['net_utility'])-to*(b['net_utility']-n['net_utility']));assert abs(directraw-r['actual_NET'])<1e-10 and abs(directmatched-r['matched_NET'])<1e-10
        assert abs(r['matched_gain_NET']+r['matched_cost_NET']-r['matched_NET'])<1e-10
        if r['group']=='COMMON':assert a['gain']==b['gain'] and a['queries']==b['queries']
        if r['group']=='REMOVED':repeats[sid,r['position']].append(r)
        checked+=1
    assert checked==7134;rec=read(O/'removed_state_recurrence.json');assert len(rec)==len(repeats)
    for r in rec:
        rr=repeats[r['sample_id'],r['position']];assert sorted(p['seed'] for p in rr)==r['seeds'];assert len({(p['actual_next_gain'],p['actual_next_queries']) for p in rr})==1
    summary=read(O/'summary.json');s0,s1,s2=summary['per_seed'];assert s0['groups']['ADDED']['pairs']==s1['groups']['ADDED']['pairs']==0;assert s2['groups']['ADDED']['pairs']==12;assert s0['totals']['actual_NET']>0 and s1['totals']['actual_NET']<0 and s1['totals']['matched_NET']>0
    text=f'''# GRID211判读：把可利用的节省与预算配比效应分开

种子1在同预期查询量下的净效用变化为{s1['totals']['matched_NET']:+.9f}，但它没有新增一次追加。共同保留的136次追加，在预算匹配后产生{s1['groups']['COMMON']['matched_NET']:+.9f}的净效用差；取消部分为{s1['groups']['REMOVED']['matched_NET']:+.9f}。因此这条优势来自原有收益的保留配比，并不表示找到了新有利动作。预算比较仍然有效，只是它和实际新增机会不是同一证据。

该种子实际取消42次追加，3个回本配对来自同一个物理调用，实际净效用变化{s1['totals']['actual_NET']:+.9f}。不能把三个相关前缀当三次独立失败，也不能忽略少数收益损失抵消了很多次节省。

种子0提供更直接的小正向：没有新增追加，只取消58次，费用节省略超过收益损失，实际净效用变化{s0['totals']['actual_NET']:+.9f}。种子2的12次新增追加均未回本，实际贡献{s2['groups']['ADDED']['actual_NET']:+.9f}，是该种子实际损失的大头；取消部分实际贡献{s2['groups']['REMOVED']['actual_NET']:+.9f}。同预算时共同批次的保留率又带来更大的负向配比影响，不能将它全部归于新增12次本身。

全部157个取消配对中，146个零收益、5个微小正收益不回本、6个回本。按相同前缀和停止位置核查，7个无利状态在2个种子中重复被取消；没有全3种子重复的取消状态。六个回本损失各在单个种子出现，但仍共享4个物理调用。重复证据有限，不据此声明已经学会稳定取消。

我的取舍：继续保留208，保存210的节省线索；下一步先核查训练数据里，“原策略要查、候选策略认为可取消”这类分歧是否有足够正反收益样本，且与实际停止状态吻合。这个核查不训练、不执行组合策略、不直接以这些留出结果训练路由器。若条件不足，先解决训练支持，再设计学习模块。

全7134条原始与同预算贡献均已独立还原，48个日期预算的成本项严格抵消，所有种子和反例保留。无新预测、拟合、神经网络前向或物理回放；最终8和原FRAME比较不动。
''';(O/'判读与后续边界.md').write_text(text,encoding='utf8',newline='\n');C.save('independent_audit.json',dict(passed=True,all7134_native_saved_utility_differences_and_expected_mixtures_reproduced=True,same_state_seed_recurrence_recomputed=True,all3_seed_common_added_removed_effects_separated=True,new_fits=0,new_predictions=0,new_physics=0));C.archive('audited');dest=R/'论文协作资料/FRAME_预算改善来源_GRID211'
    for n in ['判读与后续边界.md','independent_audit.json']:shutil.copy2(O/n,dest/n)
    msg='GRID211分解：种子1无新增追加，同预算正向来自共同批次保留配比，实际误删3相关回本使净效用负；种子0仅取消58次实际小正；种子2新增12次均不回本，是实际损失大头。157取消含151无利/6回本；7同状态在2种子重复无利取消，无3种子重复。下一步只读核查训练分歧支持，不自动训练/组合。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID211查询变化与预算配比分解\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID211后\n\n1. 保留208及有效取消线索，区分实际节省与同预算共同批次保留配比，不扫深度。\n2. 下一步只读核查拟合数据中208查/210停的分歧数量、正反收益和实际STOP域支持，检查in-sample选择偏差；不训练/执行新路由或组合。\n3. 新训练有明确依据才通知固定范围；不调费用阈值，不扩物理/最终8/论文/原FRAME。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID211查询与预算来源核查\n\n[报告](protocols/'+C.A.name+'/audited/GRID211_report.md)；[判读](protocols/'+C.A.name+'/audited/判读与后续边界.md)；零拟合/预测，配比与真实新增机会分开。\n\n').encode('utf8')+p.read_bytes());print('GRID211_AUDITED',checked)
if __name__=='__main__':main()
