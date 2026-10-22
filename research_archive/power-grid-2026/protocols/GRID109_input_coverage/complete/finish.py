"""Independent saved-data checks and calibrated Chinese report; no fitting."""
from pathlib import Path
import collections, hashlib, json, pickle, shutil
import numpy as np
ROOT=Path('D:/Codex/GRL研究'); OUT=ROOT/'outputs/grid109_input_coverage'; WORK=ROOT/'work/grid109_input_coverage'
REPO=ROOT/'work/GRL-results-integration'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf8')

def main():
    assert not (OUT/'complete.json').exists()
    plan=read(OUT/'plan.json'); D=read(OUT/'diagnosis.json'); details=read(OUT/'positive_flip_details.json'); critical=read(OUT/'critical_coverage.json')
    inputs=read(OUT/'input_manifest.json'); assert all(sha(ROOT/p)==h for p,h in inputs.items())
    C=pickle.loads((ROOT/'outputs/grid102_search_rl/cases.pkl').read_bytes()); N=np.load(ROOT/'outputs/grid102_search_rl/normalization.npz')
    raw=np.concatenate([c['states'] for c in C if c['partition']=='train']); X=np.clip((raw-N['mean'])/N['scale'],-20,20).astype(np.float32).astype(float)
    rows=read(ROOT/'outputs/grid107_direct_value/training_rows.json'); target=np.concatenate([c['gains']-read(ROOT/'outputs/grid102_search_rl/design.json')['cost_per_query']*c['queries'] for c in C if c['partition']=='train'])
    gs=collections.defaultdict(list)
    for i,r in enumerate(rows): gs[r['physical_state_group']].append(i)
    names=read(ROOT/'outputs/grid102_search_rl/design.json')['input']['features']
    group_indices=[[names.index(n) for n in a] for a in plan['groups'].values()]
    checked=0; pairdetails=[]
    for r in details+critical:
        z=np.array([r['raw_features'][n] for n in names]); x=np.clip((z-N['mean'])/N['scale'],-20,20).astype(np.float32).astype(float)
        delta=X-x
        for metric in ['standardized41','semantic_balanced']:
            if metric=='standardized41': distance=np.linalg.norm(delta,axis=1)/np.sqrt(41)
            else: distance=np.sqrt(sum(np.einsum('ij,ij->i',delta[:,ix],delta[:,ix])/len(ix) for ix in group_indices)/3)
            reps=sorted([min(ix,key=lambda i:distance[i]) for ix in gs.values()],key=lambda i:(distance[i],rows[i]['physical_state_group']))
            reported=r['neighbours'][metric]
            for k in [1,5,20]:
                chosen=reps[:k]; a=reported['k'][str(k)]
                assert a['positive_calls']==sum(target[i]>0 for i in chosen)
                assert abs(a['distance']['mean']-np.mean(distance[chosen]))<1e-10
                assert len({rows[i]['physical_state_group'] for i in chosen})==k
                checked+=1
            assert [rows[i]['physical_state_group'] for i in reps[:20]]==[v['physical_state_group'] for v in reported['nearest20']]
        ni=reps[0]; contributions=[]
        for group,ix in zip(plan['groups'],group_indices):
            contributions.append(dict(group=group,standardized_rms=float(np.sqrt(np.mean(delta[ni,ix]**2)))))
        top=sorted(range(41),key=lambda j:-abs(delta[ni,j]))[:8]
        pairdetails.append(dict(sample_id=r.get('sample_id'),context=r.get('context'),position=r.get('position'),
            nearest_semantic_call=rows[ni]['physical_state_group'],query_target=r['normalized_target'],neighbour_normalized_target=float(target[ni]/read(ROOT/'outputs/grid102_search_rl/design.json')['gain_scale']),
            group_differences=contributions,largest_existing_input_differences=[dict(feature=names[j],query_raw=float(z[j]),train_raw=float(raw[ni,j]),standardized_difference=float(delta[ni,j])) for j in top]))
    save(OUT/'nearest_pair_differences.json',pairdetails)
    decomposed=read(OUT/'utility_decomposition.json')['all_calls']; decision108=read(ROOT/'outputs/grid108_loss_budget/decision.json')
    delta=decision108['contrasts']['budget_underHuber'] if 'budget_underHuber' in decision108['contrasts'] else None
    # Reconstruct both endpoints from original trajectory rows, independent of saved decomposition.
    trajectories=read(ROOT/'outputs/grid108_loss_budget/validation_trajectories.json'); totals=[]
    for seed in range(3):
        means=[]
        for b in [7250,29000]:
            rs=[r for r in trajectories if r['method']==f'HUBER_{b}_{seed}']
            dates=sorted({r['date'] for r in rs})
            means.append(float(np.mean([np.mean([r['net_utility'] for r in rs if r['date']==d]) for d in dates])))
        total=sum(r['contribution_to_date_macro'] for r in decomposed if r['seed']==seed)
        assert abs(total-(means[1]-means[0]))<1e-12;totals.append(total)
    for seed in range(3):
        draws=np.load(ROOT/f'outputs/grid108_loss_budget/draws_HUBER_29000_{seed}.npz')['state_draws']
        assert draws.sum()==29000*32 and draws.min()>0
    # Check sealed test paths only for existence; no test contents or labels read.
    base=read(ROOT/'outputs/grid100_data_foundation/design.json')
    test=[r for r in base['scenario_plan'] if r['partition']=='test'];assert len(test)==8
    assert not any((ROOT/base['environment']/'chronics'/r['scenario']).exists() for r in test)
    assert not any((ROOT/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists() for r in test)
    review=dict(passed=True,source_hashes_unchanged=len(inputs),independent_neighbour_set_checks=checked,
        training_array_reproduced=True,sampling_totals_per_seed=928000,utility_decomposition_matches_all_seeds=True,
        exact_collision_scope='All frozen4438 train/validation model inputs; no collisions.',
        sealed_test_paths_absent=8,new_fits=0,new_inference=0,new_physical_calls=0,new_features=0,
        limits='Saved-data audit only. No independent grid simulation or proof that feature enrichment will work.')
    save(OUT/'review.json',review)
    conclusions=dict(training_states_unsampled_explanation_rejected=True,positive_effective_diversity_limited=True,
        exact_input_label_contradiction_found=False,nearest_neighbour_sign_rule_supported=False,
        missing_physical_candidate_encoding_confirmed_by_code=True,feature_insufficiency_causally_proven=False,
        larger_training_budget_recommended=False,new_model_or_feature_approved=False,ready_to_scale=False)
    save(OUT/'decision.json',conclusions)
    bygroup=collections.defaultdict(list)
    for r in details:
        if any(e['event']=='positive_newly_missed' for e in r['events']): bygroup[r['physical_state_group']].append(r)
    meancontrib=collections.defaultdict(float)
    for r in decomposed: meancontrib[r['physical_state_group']]+=r['contribution_to_date_macro']/3
    lines=['# GRID109：第552步修正后，为什么其他地方反而漏查','',
        '**主要问题已经收窄到：训练正收益模式有限，模型对这些模式的迁移不稳定。继续增加更新次数不是当前最有价值的方向。** 现有输入确实没有描述待补查动作的具体拓扑变化，但这还不能证明补上这些信息就会改善。','',
        '## 查了什么','',
        '本轮只使用已有3582个训练状态、856个验证状态、上一轮保存的预测和抽样记录。比较Huber的7250与29000次更新，保留三个种子全部结果；没有拟合、重新推理、物理回放、增加特征或打开最终测试集。',
        '对所有新漏查和修正状态逐项检查，并用全部验证状态作参照。邻居按不同物理搜索调用去重，每次调用只保留最近的一个前缀；同时报告原41维标准化距离，以及预算、已见质量、反馈历史三组等权距离。距离只用于描述覆盖，不是新策略。','',
        '## 1. 新漏查不是十三个独立场景','',
        '三个种子的新增正收益漏查为7、6、5个；跨种子去重后是13个状态，来自8次物理搜索、3个验证日期。被修正的则为2个状态、2次搜索，包括第552步。',
        '新增漏查的完整清单如下。括号内为涉及的不同前缀/可达位置数；即时净收益已扣除查询费，再除以原收益尺度，仅用于搜索决策比较。']
    for g,rs in sorted(bygroup.items()):
        vals=[r['normalized_target'] for r in rs]
        lines.append(f"- {g}，第{rs[0]['step']}步（{len(rs)}个状态），净收益{min(vals):.4f}～{max(vals):.4f}。")
    lines+=['',
        '十月19日第二份时序第1964步的一个前缀，三个种子都从正确继续变成停止；它不是只在某一个种子中偶然出现。八月3日第一份时序第1899步的两次漏查收益也较大，不能只用漏查次数衡量影响。','',
        '## 2. 训练不是没见过正收益，而是独立模式不多','',
        '训练3582个状态中，211个下一批有正净收益，但只来自54次物理搜索、15个日期。按原日期/前缀均衡抽样规则，正收益权重为5.66%。同一次物理状态衍生的多个前缀增加了行数，不能当成同样数量的新运行状态。',
        '29000次更新时，每个种子抽样928000个状态。211个正收益状态全部被采到，跨三个种子的最低抽样次数为43，正收益状态抽样中位数约258～262次。因此，本轮不支持“正例根本没有进入训练”的解释；继续重复相同样本也不会自动增加运行模式。',
        '有限的正收益覆盖是已确认的数据特征，尚不是全部误差的因果解释。多数训练正例的动作能被学对，仍有17、20、30个被长训练模型漏掉；训练拟合和跨日期迁移问题同时存在。','',
        '## 3. 新漏查附近通常是负收益，但不能据此直接停止','',
        '按原41维距离，13个新漏查状态中，8个在最近5次训练搜索的代表邻居里没有正收益；三组等权距离下是7个。它们的最近训练距离中位数分别为0.347和0.341；全部41个验证正收益状态的对应中位数为0.314和0.306。新漏查在这两种描述下更缺少附近的正收益例子，但样本相关、数量也小，不作显著性或分布外断言。',
        '反过来，第552步原距离下最近5次搜索全部为负收益，等权距离下有1个正收益；它自己却是明显正收益，而且延长训练能修正它。因此，“邻居多数负收益就停止”会再次漏掉这类关键例外，不能把邻域统计直接改成触发条件。',
        '全部验证正收益中也有18/41（原距离）或16/41（等权距离）在最近5次训练搜索内没有正收益。这个现象并非新漏查独有，更准确的说法是：现有输入空间里的局部覆盖对收益符号提供的信息有限，模型的边界移动会同时修正和破坏不同位置。','',
        '## 4. 当前41维到底缺少什么信息','',
        '代码核对确认，这41维主要描述候选数量、已见奖励/最大负载率的均值与离散程度、合格比例、历史改善以及剩余补查量。它们保留了“前面查得怎样”和“还剩多少”，没有把以下内容编码给模型：',
        '- 当前电网各线路的负载与站内连接状态；',
        '- 剩余候选具体改变哪些母线连接、与当前最佳动作的结构差异；',
        '- 已查询动作的结构与其预测反馈之间的对应关系。',
        '这使得模型主要根据汇总统计预测“剩余动作是否值得查”，而没有直接看到剩余动作的结构。这个缺口与任务逻辑直接相关，比再堆同类均值、方差或简单加宽网络更值得检查。',
        '不过，4438个训练/验证输入没有完全相同的向量，也没有发现“同一输入对应正负两种标签”的精确矛盾。相似输入、不同收益只能提出表示不足的假设，不能证明原41维在理论上无法学习。第552步能被修正也说明不能笼统说“它学不会”。','',
        '## 5. 还发现一个重要区别：漏查计数不等于策略收益','',
        '沿原教师反馈复算完整补查序列后，三个种子的验证净效用变化为'+ '、'.join(f'{v:+.8f}' for v in totals)+'。第三个种子虽然增加了漏查，整体净效用仍略升，因为它同时改变了其他无益查询与补查路径。',
        f"按同一指标精确分解，十月19日第二份时序call004对三个种子平均净效用变化贡献{meancontrib['2037-10-19_101|call004']:+.8f}，是主要负贡献之一。完整调用级分解保存在utility_decomposition.json，所有正反贡献均保留。",
        '这些只是固定教师路径上的搜索收益分解，不能当成整周电网成本或独立物理干预的因果结果。','',
        '## 我的判断与下一步','',
        '暂时停止增加训练更新次数，也不急着再换损失或扩大网络。当前最值得研究的是让补查决策知道“准备查什么”，而不仅是“前面查得怎么样”。',
        '建议下一步先核对已有数据能否构造一组小而明确的候选结构信息：待查动作相对当前最佳动作的连接差异，以及这些变化涉及的站内/相邻线路压力。再检查它们能否区分上述8次搜索与相似负例。候选定义、来源和分组验证方式应先确定，不能用验证结果反复挑字段；未查询动作的预测结果不能作为输入。',
        '这是下一项建议，不是本轮已经验证的改进。若结构信息在已有数据中无法可靠还原，先说明缺口，再讨论采集，不能直接扩大。最终8个测试场景继续封存。','',
        '## 可复核材料','',
        '- positive_flip_details.json：全部15个正收益动作变化状态、种子、实际步数、两套距离和最近20次训练搜索。',
        '- critical_coverage.json：108、552及RF路径552三个固定上下文。',
        '- validation_coverage.json：全部856个可达验证状态的覆盖参照。',
        '- nearest_pair_differences.json：与最近训练状态的现有特征差异。',
        '- utility_decomposition.json：每种子、每物理调用对原日期宏平均净效用的完整贡献。',
        '- review.json、input_manifest.json：独立距离/分解核对与来源哈希。']
    (OUT/'GRID109_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID109_input_coverage/complete';archive.mkdir(parents=True,exist_ok=False)
    collab=ROOT/'论文协作资料/FRAME_输入覆盖诊断_GRID109_20261022';collab.mkdir(parents=True,exist_ok=False)
    save(OUT/'complete.json',dict(passed=True,read_only_diagnosis=True,review='review.json',report='GRID109_report.md',archive=str(archive),collaboration=str(collab),new_fits=0,new_physical_calls=0,ready_to_scale=False))
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p,archive/p.name)
    for p in WORK.glob('*.py'): shutil.copy2(p,archive/p.name)
    for n in ['GRID109_report.md','diagnosis.json','decision.json','review.json','complete.json']: shutil.copy2(OUT/n,collab/n)
    save(archive/'archive_manifest.json',{p.name:sha(p) for p in archive.iterdir() if p.is_file()})
    note='\n\n## 2026-10-22 — GRID109 saved-input and coverage diagnosis\n\nRead-only: all Huber positive flips deduplicated to13 newly missed states/8 physical calls versus2 recovered/2 calls; all3 seeds and all856 validation controls retained.211 positive train states come from54 calls and15 dates,5.6607% sampling mass, all drawn>=43 times at29000. New misses have no positive among nearest5 distinct train calls in8/13 original41-RMS or7/13 semantic-balanced metric; this also occurs in18/41 or16/41 of all validation positives and in the corrected552 context, so neighbour sign is not a trigger rule/proof of insufficiency. No exact input collisions across4438 states. Code confirms absence of physical line/topology vectors, pending-action topology and action-feedback correspondence from current summary inputs. Hypothesis worth checking: candidate-conditioned structural information; no feature added or selected, no fit/inference/physical rollout, no final-test data accessed. Per-call utility decomposition independently reproduces all3 endpoint deltas; third seed slightly improves despite extra misses. No scaling.\n'
    for n in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write(note)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID109输入与覆盖诊断完成\n\n1. 汇报：新增漏查13状态/8调用；正例已充分抽样但独立运行模式有限，邻域多数负例不能直接作停止条件。\n2. 下一候选方向：先核对已有数据中待查动作/最佳动作结构与相关线路压力是否可可靠还原，固定小型信息方案后再讨论验证；本轮没有添加特征或授权新训练。\n3. 不继续延长预算/换损失/扩网络或新闭环；最终8测试场景封存。\n\n报告：research_archive/power-grid-2026/protocols/GRID109_input_coverage/complete/GRID109_report.md\n',encoding='utf8')
    print(json.dumps(review,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
