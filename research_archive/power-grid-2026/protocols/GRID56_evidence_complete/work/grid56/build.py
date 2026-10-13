"""GRID56: final evidence closure and paper tables from preserved results only."""
import argparse,csv,hashlib,json,math,os
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--out',type=Path)
args=parser.parse_args();ROOT=args.source_root.resolve();OUT=args.out or ROOT/'outputs/grid56'
assert not OUT.exists(),'Do not overwrite previous evidence bundle.'
OUT.mkdir(parents=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda n,x:(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csvsave(n,rows):
    with (OUT/n).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
mean=lambda v:math.fsum(v)/len(v)

# Finish audit closure without rerunning the scientific worker or old audit.
B=ROOT/'outputs/grid55b';V=B/'qualified_view';D=read(V/'design.json');F=read(V/'finished.json');R=read(V/'review.json')
assert read(B/'review.json')['passed'] and read(B/'supplement_audit.json')['passed']
assert R['passed_saved_data'] and len(F['summaries'])==128
frozen=read(B/'code_freeze.json')
for p,h in frozen.items(): assert sha(ROOT/p)==h,p
raw=read(B/'raw_preservation.json')
for p,h in raw.items(): assert sha(ROOT/p)==h,p
receipts=read(B/'local_trajectory_receipts.json')
for a in receipts:
    p=ROOT/a['local_path'];assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256'],p
assert read(B/'windows_power_events.json')==[] and read(B/'power_log_access.json')['readable']
assert read(B/'awake_lease_release.json')['returned']!=0 and not (B/'RUN.lock').exists()
provenance=read(V/'provenance.json');assert len(provenance)==128 and sum(p['new_supplement'] for p in provenance)==8
ix={(s['week'],s['repeat'],s['method']):s for s in F['summaries']}
for p in provenance:
    slot,week,method=p['logical_slot'].split('|');s=ix[(week,int(slot),method)]
    folder=ROOT/p['source_folder'];a=read(folder/'summary.json')
    assert a['repeat']==p['physical_repeat'] and {**a,'repeat':int(slot)}==s
    assert sha(folder/'summary.json')==p['summary_sha256']
    assert str(folder)==D['run_paths'][p['logical_slot']]
comparison_checks=0
for c in R['comparisons']:
    values=[mean([1-ix[(w,t,c['method'])]['controller_s']/ix[(w,t,c['opponent'])]['controller_s'] for t in [0,1]]) for w in D['weeks']]
    assert abs(mean(values)-c['macro_control_reduction'])<1e-14
    sample=np.asarray(values)[np.random.default_rng(20261011).integers(0,8,size=(10000,8))].mean(1)
    assert np.max(np.abs(np.quantile(sample,[.025,.975])-c['family_bootstrap95']))<1e-12
    comparison_checks+=1
gate={}
for s in R['summaries']:
    m=s['method'];q=[(ix[(w,t,m)]['cost']-ix[(w,t,'FULL')]['cost'])/abs(ix[(w,t,'FULL')]['cost']) for w in D['weeks'] for t in [0,1]]
    assert abs(max(q)-s['max_complete_week_cost_regret'])<1e-14
    if m!='FULL':
        values=[mean([1-ix[(w,t,m)]['controller_s']/ix[(w,t,'FULL')]['controller_s'] for t in [0,1]]) for w in D['weeks']]
        gate[m]=all(ix[(w,t,m)]['complete'] for w in D['weeks'] for t in [0,1]) and max(q)<=.005 and mean(values)>=.05 and sum(v>0 for v in values)>=6
        assert gate[m]==s['investment_screen_passed']
assert gate['COMPRESSED_GNN0'] and gate['COMPRESSED_GNN1'] and not gate['LOCAL_GNN0'] and not gate['LOCAL_GNN1']
assert R['version_selection']['engineering_preference']=='COMPRESSED_CANDIDATE_QUALITY_SCREEN'
ARCH=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026'
archived=0
for manifest in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for a in read(manifest).get('files',[]):
        p=manifest.parent/a['path']
        if not p.exists(): p=ARCH/a['path']
        assert sha(p)==a['sha256'],p
        archived+=1
audit={'passed':True,'original_frozen_hashes_checked':len(frozen),'raw_anchor_hashes_checked':len(raw),
       'local_supplement_file_receipts_checked':len(receipts),'archived_file_entries_checked':archived,
       'provenance_slots':128,'all_method_replacement_slots':8,'direct_statistics_and_bootstrap_checks':comparison_checks,
       'independent_gate_formula':True,'new_simulations':0,'new_fits':0,'independent_ac_resolve':False,
       'scope':'Closure over preserved vectors audit, file receipts, source identity, arithmetic, bootstrap and original rule. Not independent physics simulation or training.'}
save('FINAL_AUDIT.json',audit)

sources=[ROOT/'outputs'/st/'review.json' for st in ['grid34','grid37','grid52']]+[V/'review.json',V/'finished.json',V/'provenance.json',B/'code_freeze.json',B/'audit_revision.json']
sources += [ROOT/'outputs'/st/n for st,n in [('grid18','finished.json'),('grid19','finished.json'),('grid21','training_finished.json'),('grid21','design.json'),('grid50','training_finished.json'),('grid50','design.json'),('grid40b','training_finished.json')]]
sources += [ROOT/'work'/st/n for st,n in [('grid50','compressed_residual.py'),('grid50','train.py'),('grid20','components.py'),('grid21','prior.py'),('grid28','policy.py'),('grid27','policy.py'),('grid08','common.py')]]
save('sources.json',[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sources])
table=[];family=[]
for s in R['summaries']:
    m=s['method'];c=next((c for c in R['comparisons'] if c['method']==m and c['opponent']=='FULL'),None)
    costs=[];counts=[]
    for week in D['weeks']:
        local=[ix[(week,t,m)] for t in [0,1]];full=[ix[(week,t,'FULL')] for t in [0,1]]
        delta=mean([(a['cost']-b['cost'])/abs(b['cost']) for a,b in zip(local,full)])
        speed=mean([1-a['controller_s']/b['controller_s'] for a,b in zip(local,full)])
        costs.append(delta)
        family.append({'week':week,'method':m,'cost_delta_full_pct':100*delta,'control_reduction_full_pct':100*speed,
            'mean_control_s':mean([a['controller_s'] for a in local]),'mean_n1_queries':mean([a['n1_forecasts'] for a in local]),
            'both_complete':all(a['complete'] for a in local),'quality_within_0_5_pct':delta<=.005})
    g=[a for a in F['summaries'] if a['method']==m]
    row={'method':m,'complete_runs':s['completed_runs'],'quality_pass_runs':s['quality_qualified_pairs'],
         'macro_control_reduction_full_pct':0 if c is None else 100*c['macro_control_reduction'],
         'pooled_control_reduction_full_pct':0 if c is None else 100*c['pooled_equal_horizon_control_reduction'],
         'query_reduction_full_pct':0 if c is None else 100*c['pooled_equal_horizon_query_reduction'],
         'mean_family_cost_delta_full_pct':100*mean(costs),'worst_family_cost_delta_full_pct':100*max(costs),
         'mean_week_control_s':s['control_s']/16,'mean_week_p95_ms':1000*s['mean_family_p95_decision_s'],
         'mean_n1_forward_rank_ms':1000*math.fsum(a['forward_rank_s'] for a in g)/sum(a['n1_calls'] for a in g),
         'mean_n1_prior_ms':1000*math.fsum(a['prior_s'] for a in g)/sum(a['n1_calls'] for a in g),
         'bootstrap_low_pct':None if c is None else 100*c['family_bootstrap95'][0],
         'bootstrap_high_pct':None if c is None else 100*c['family_bootstrap95'][1]}
    table.append(row)
csvsave('main_table.csv',table);csvsave('per_family.csv',family)
tr=read(ROOT/'outputs/grid21/design.json')['training_weeks'];teacher=[]
for stage in ['grid18','grid19']:
    a=read(ROOT/'outputs'/stage/'finished.json'); selected=[s for s in a['summaries'] if s['week'] in tr]
    teacher.append({'stage':stage,'training_episodes':len(selected),'states':sum(len(s['calls']) for s in selected),
        'training_episode_wall_s':math.fsum(s['wall_s'] for s in selected),'training_control_s':math.fsum(s['controller_s'] for s in selected),
        'training_n1_forecasts':sum(s['n1_forecasts'] for s in selected),'training_all_forecasts':sum(s['public_forecasts'] for s in selected),
        'physical_steps':sum(s['steps'] for s in selected),'early_failed_training_weeks':[s['week'] for s in selected if not s['complete']],
        'whole_invocation_wall_s_including_nontraining':a['wall_s']})
assert sum(a['states'] for a in teacher)==167
train=read(ROOT/'outputs/grid50/training_finished.json');original=read(ROOT/'outputs/grid21/training_finished.json')
costs={'teacher':teacher,'teacher_training_episode_wall_s':math.fsum(a['training_episode_wall_s'] for a in teacher),
       'teacher_training_n1_forecasts':sum(a['training_n1_forecasts'] for a in teacher),
       'teacher_training_all_forecasts':sum(a['training_all_forecasts'] for a in teacher),
       'teacher_states':167,'valid_preference_targets':148,'no_target_states_retained':19,
       'current_candidate_fits':train['fits'],'current_candidate_fit_invocation_wall_s':train['wall_s'],
       'original_gnn_fits':[r for r in original['fits'] if r['kind']=='gnn'],
       'softgnn_fits':read(ROOT/'outputs/grid40b/training_finished.json')['fits'],
       'current_candidate_extra_teacher_queries':train['new_forecasts'],
       'excluded_or_unidentified':['Author PPO pretraining cost/exposure unknown','Trace acquisition/extraction separate','Feature/canonicalization/preprocessing outside recorded fit timer not reconstructed','Earlier abandoned model runs are research cost, not charged as one final fit']}
save('training_teacher_costs.json',costs)

claims=[
 {'claim':'最终候选相对同共有缓存FULL减少完整控制时间','status':'SUPPORTED','evidence':'GRID52确认14.606%；GRID55B同日期工程版15.752%。两种子及八家族展示。'},
 {'claim':'约70%拓扑查询减少等于约70%完整控制提速','status':'REJECTED','evidence':'主表查询70.00%/70.62%，完整控制15.44%/16.07%，计量不同。'},
 {'claim':'最终候选质量达到预设工程尺度','status':'SUPPORTED_SCREEN_ONLY','evidence':'两种子最坏成本+0.08045%，全周完成；0.5%是投资尺度，不是等价证明。'},
 {'claim':'新版更快，因此胜过旧版','status':'REJECTED','evidence':'新版相对旧版平均控制时间多3.50%；采用新版因为原版质量尾部超尺度。'},
 {'claim':'最终候选全面快于PPO及SoftGNN','status':'REJECTED','evidence':'PPO/SoftGNN仍略快；质量、查询和控制时间分别展示。'},
 {'claim':'全部收益来自图消息传递','status':'NOT_ESTABLISHED','evidence':'共有缓存/停止有作用；GRID37静态顺序在开发周已达到近似成绩。'},
 {'claim':'图模型比等信息MLP普遍更好','status':'NOT_ESTABLISHED_FOR_FINAL_MODEL','evidence':'GRID34原版与MLP仅开发证据，当前软压缩版没有同配置MLP闭环。'},
 {'claim':'论文在RTE生产电网中节省实际电费','status':'REJECTED','evidence':'目前为公开Grid2Op/LightSim2Grid轨迹回放，运行成本指标是模拟奖励账目。'},
 {'claim':'本轮训练了新的RL策略','status':'REJECTED','evidence':'监督候选排序使用已有作者PPO先验；最终拟合是CE+.1KL，不是RL。'}]
save('claim_evidence.json',claims)
own=R['own_seed_average'];now=next(a for a in table if a['method']=='COMPRESSED_GNN0')
lines=['# GRID56 最终审计与论文证据包','',
 '补测保存数据审计已完成，本轮完成冻结哈希、原结果锚点、新八条文件收据、128计时槽位来源、21项直接统计/家庭bootstrap和对称选型的收口复核。没有新增仿真、训练或独立AC重求解。','',
 '## 主版本与主结果','',
 '软压缩GNN两个种子为工程主候选；旧版作为质量—速度消融保留。不是因为新版更快，而是它通过原质量尺度、原版未通过。默认历史源码未被改写。','',
 '| 方法 | FULL控制时间减少 | 查询减少 | 平均成本变化/FULL | 最坏成本变化/FULL | 完成 | 质量尺度通过 |',
 '|---|---:|---:|---:|---:|---:|---:|']
for a in table:
    lines.append(f"| {a['method']} | {a['macro_control_reduction_full_pct']:.3f}% | {a['query_reduction_full_pct']:.3f}% | {a['mean_family_cost_delta_full_pct']:+.4f}% | {a['worst_family_cost_delta_full_pct']:+.5f}% | {a['complete_runs']}/16 | {a['quality_pass_runs']}/16 |")
lines+=['',f"新版种子平均减少{100*own['macro_control_reduction']:.3f}%，家庭bootstrap95%[{100*own['family_bootstrap95'][0]:.3f}%,{100*own['family_bootstrap95'][1]:.3f}%]。八日期来自一个电网，种子/重复不是独立网格样本。GRID52为冻结后的确认，GRID55B为已暴露同日期工程比较，不合并宣称两次独立确认。",'',
 '## 训练与教师费用','',f"20个训练周共167个状态，148个有效偏好、19个无目标状态保留。仅训练周的已记录回放墙钟{costs['teacher_training_episode_wall_s']/60:.2f}分钟，N1教师查询{costs['teacher_training_n1_forecasts']:,}、全部公开预测{costs['teacher_training_all_forecasts']:,}。两次教师完整进程时间另列，GRID19包含四个开发周，不能把它们混作训练费用。",'',
 f"最终两种子固定64轮，训练调用墙钟{train['wall_s']:.2f}秒，单个拟合{train['fits'][0]['training_s']:.2f}/{train['fits'][1]['training_s']:.2f}秒，106449可训练参数；额外教师查询0。已有PPO训练与离线输入构造费用未完整识别，不能称89.99秒为从零构建全系统费用。",'',
 '## 当前最小缺口','',
 '先补最终配置的廉价顺序消融，而不是继续训练新网络。GRID37中STATIC_ID在四个旧开发周已获得近似的完整时间和合格质量，它不能因不是近期论文而被省略：这是判断排序学习是否必要的最小消融，不扩主对照菜单。',
 '下一项有限实验应只把既有STATIC_ID原顺序用于最终八日期，保持全部控制/停止/缓存/输入规则，并按完整周成本、存活、查询比较当前GNN。单独批次的墙钟仅描述，不作为公平提速主张；若需公平计时再明确另设配对矩阵。无需新训练。',
 '当前没有最终软压缩版的MLP控制；旧版MLP仅提供开发线索。先做一个静态消融即可，暂不扩成多模型清单。','',
 '## 工程与证据边界','',
 '原GRID55 RAW128含待机异常且保留；修复视图仅对受影响日期全部八方法统一用repeat0/2。首次审计PowerShell5模块加载失败和PowerShell7修复均留档；不存在重复科学求解。共有DC、安全QP和先验缓存提供给适用对照，不能归为GNN独有。',
 'SoftGNN为官方网络/转换器的资源匹配适配，不是原论文完整训练复现；作者PPO训练暴露未知。局部先验减少87.94%不是完整控制收益。现有成本是模拟指标，不是生产收益。',
 '主张定位：学习辅助的有限拓扑搜索与控制集成，在一个公开电网环境展示质量—速度折中。是否足以形成录用贡献仍需最终廉价消融和论文审查，不给录用概率。','',
 '文件：FINAL_AUDIT.json、main_table.csv、per_family.csv、training_teacher_costs.json、claim_evidence.json、method_and_results_draft.md、REPRODUCE.md、quality_speed.pdf/png。']
(OUT/'GRID56_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

draft='''# 论文方法与结果草稿（中文工作稿）

工作标题：面向电网拓扑控制的学习辅助分批搜索与质量—速度折中。

## 方法

系统以作者既有电网控制器为底座，保持连续调度、恢复动作和合法性约束，在需要昂贵拓扑候选仿真时改变候选查询顺序与补查方式。它输出经公开一步仿真筛选的动作，而不是直接把网络预测当作物理可行证明。

输入仅包含当前公开观测、当前基动作和静态拓扑动作库。236个母线节点、14维节点特征、最多372条有向弧和4维边属性；全局状态11维，基动作1605维，909个候选各含540维已知拓扑计划。连通边和潮流/负载率随当前状态更新；未来候选仿真分数只在实际查询之后进入选择。

图编码采用两个64维消息传递层，以当前边属性补充邻居消息并按邻居数归一化。全图均值/最大池化与候选所属变电站两母线的局部表示、候选计划编码、基动作编码和全局状态拼接，共享候选解码头产生残差z_i。作者PPO及合法物理尾部顺序给出覆盖所有候选的先验π_i=-log(1+rank_i)。最终评分s_i=π_i+2 asinh(z_i/2)，原版使用2 tanh(z_i/2)。尺度2沿用原配置，未通过确认集扫参。

训练从20个既有训练周的完整搜索公开反馈提取目标。严格有效候选按exp(-(rho_i-rho_min)/0.01)归一化得到软偏好q；167个状态中148个有目标，其余保留计量但不参与偏好拟合。最小化CE(q,p)+0.1 KL(p||softmax(π))，p为合法候选上的模型softmax。固定64轮、批量8、Adam学习率0.001、权重衰减0.0001、两个模型种子，零初始化校正头；不按开发或测试表现选择检查点。本文没有新RL训练，作者已有PPO仅作为已部署先验。

搜索首先查询前128个合法等价类代表；若已有候选rho低于作者0.9尺度则可停止，否则以64为批补查。在至少256项且连续两批未改善当前有效最优时可启发式停止；无有效候选时继续搜索。停止之前补全当前最优动作所在变电站的候选。逐项可接受性、成本反馈和fp32奖励破并列保持一致。这是经验停止规则，不是全池最优或长期安全证书；所有早失败和成本尾部都报告。

DC和安全QP编译缓存对所有方法公平开放，先验静态元数据缓存对全部先验使用者开放；第一次构造与每次动作字段新鲜度核验计入控制墙钟。物理env.step、导出和初始化分开记账；主速度来自实际controller.propose墙钟而不是查询数量或单个神经网络前向。

## 评价与结果使用规则

GRID52的八日期在评价前冻结，两个模型种子和两次计时重复全部保留，主对照家族为完整拓扑搜索、作者PPO排序和SoftGNN。SoftGNN保留官方GAT及转换器，但数据、动作域和训练预算为资源匹配适配。它不是原论文全部实验的复现。

GRID55在同一八日期加入原版作配置比较，并公平安装先验等价缓存。最后一个日期重复1被操作系统待机干扰，原128条及其RAW统计保留。GRID55B为该日期所有八方法补一次完整重复2，路径、向量、逐步成本和公开反馈与原两重复逐项相同；修复视图用原repeat0/新repeat2，其他日期保持原repeat0/1。该工程结果不称新独立确认。

主表在main_table.csv；每个日期先平均计时重复，再对日期宏平均，两个模型种子分别展示。成本按对应FULL完整周成本归一化，提前失败不作为低成本成功，重复与模型种子不当独立电网样本。0.5%是预先工程投资尺度，不是统计等价检验。

最终候选控制提速约15.75%、拓扑查询减少约70%、最坏完整周成本增幅约0.080%。原版更快但质量尾部约1.7%，因此选择新版作质量合格主候选。它仍略慢于PPO和SoftGNN；与PPO的质量优势、与SoftGNN的查询差异分别报告，不写全面支配。

## 消融与局限

原版与软压缩版是一个评分激活的消融，不等于图结构必要性证明。旧开发集上的MLP差距较小，静态顺序也可达到接近的表现，不能将完整15.75%收益全归因于图消息传递。最终配置的廉价顺序消融需单独补齐；在此之前本文只建立集成框架的工程结果，图模型相对便宜排序的增量仍待核验。

教师回放费用、最终拟合费用和已有PPO成本分开报告。模型仅在一个RTE公开模拟环境和有限日期分布上评价，无生产部署、独立AC重求解、灾难故障覆盖或录用保证。
'''
(OUT/'method_and_results_draft.md').write_text(draft,encoding='utf-8')
(OUT/'REPRODUCE.md').write_text('''# 复核与复现入口

本包不启动回放或训练，以下命令只读已有结果、验证收据和重建论文表格：

```powershell
& work/grid02/.venv/Scripts/python.exe work/grid56/build.py --source-root C:/Users/windows/Documents/Codex/2026-09-24/influence-maximization-im-ccf-b-grl --out outputs/grid56_reproduction
```

输出目录必须不存在。需要本地大轨迹、模型权重及旧档案；Git小型结果归档不是这些资源的替代。

实际控制复现的组件入口为work/grid55/run.py的worker分支、work/grid55/worker.py，停止逻辑work/grid28/policy.py，共有缓存work/grid29/cache_dc.py、work/grid30/cache_safe.py、work/grid54/cache_prior.py；原正式运行器不允许对已完成目录重新运行。下一次有限消融必须建立新协议/输出目录，明确引用同一冻结组件，不能修改旧design.json解除保护。

最终模型：outputs/grid50/models/gnn_seed0/final.pt与gnn_seed1/final.pt。原版：outputs/grid21/models/gnn_seed0/final.pt与gnn_seed1/final.pt。模型hash见outputs/grid55b/code_freeze.json，归档中保留原新模型来源和失败记录。

八日期、动作库、作者依赖、LightSim2Grid后台和单线程要求见outputs/grid55/design.json及运行metadata.json。没有额外下载、安装、训练或隐藏成绩读取步骤。
''',encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.8,4.4),layout='constrained')
labels={'LOCAL_GNN0':'Original GNN s0','LOCAL_GNN1':'Original GNN s1','COMPRESSED_GNN0':'Compressed GNN s0','COMPRESSED_GNN1':'Compressed GNN s1','SOFTGNN0':'SoftGNN s0','SOFTGNN1':'SoftGNN s1','LOCAL_PPO':'Author PPO'}
colors={'COMPRESSED_GNN0':'#126da2','COMPRESSED_GNN1':'#48a9dc','LOCAL_GNN0':'#af8d32','LOCAL_GNN1':'#d5b866','SOFTGNN0':'#577f63','SOFTGNN1':'#8aac92','LOCAL_PPO':'#956a9b'}
for a in table:
    m=a['method']
    if m=='FULL':continue
    axes[0].scatter(a['macro_control_reduction_full_pct'],a['worst_family_cost_delta_full_pct'],color=colors[m],label=labels[m],s=52)
    axes[1].scatter(a['query_reduction_full_pct'],a['macro_control_reduction_full_pct'],color=colors[m],s=52)
axes[0].axhline(.5,color='#777777',ls='--',lw=.8,label='0.5% engineering screen')
axes[0].set_yscale('symlog',linthresh=.2)
axes[0].set(xlabel='Macro controller-time reduction vs FULL (%)',ylabel='Worst completed-week cost change vs FULL (%)',title='Quality and actual controller time')
axes[1].set(xlabel='Pooled topology-query reduction vs FULL (%)',ylabel='Macro controller-time reduction vs FULL (%)',title='Queries are not end-to-end speed')
for ax in axes:ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),fontsize=7.5,loc='outside lower center',ncol=4)
fig.savefig(OUT/'quality_speed.pdf');fig.savefig(OUT/'quality_speed.png',dpi=170);plt.close(fig)
save('outputs_manifest.json',[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()])
(OUT/'RUN_STATE.md').write_text('COMPLETED: offline audit closure, paper evidence, tables, figure and method draft. No new simulation/fit.\n',encoding='utf-8')
print(json.dumps({'passed':True,'audit':audit,'teacher_training_wall_min':costs['teacher_training_episode_wall_s']/60,'paper_bundle':str(OUT)},ensure_ascii=True))
