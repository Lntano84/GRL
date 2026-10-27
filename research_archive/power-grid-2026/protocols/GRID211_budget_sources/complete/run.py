"""Exact decomposition of fixed choices and analytic query-budget mixtures."""
from pathlib import Path
import collections,hashlib,importlib.util,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid211_budget_sources';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID211_budget_sources';NEW=R/'outputs/grid210_amount_depth';OLD=R/'outputs/grid208_public_amount'
sp=importlib.util.spec_from_file_location('grid211_fixed_helpers',R/'work/grid210_amount_depth/run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    C.trained()
def prepare():
    assert not O.exists();O.mkdir(parents=True);C.trained();assert read(NEW/'independent_audit.json')['passed']
    save('design.json',dict(id='GRID211_BUDGET_SOURCES',scope='Fixed210vs208 all7134seed-starttraces/all4681eligible states/all48date-seed budgetrows;no fitting or policyvariant.',decomposition='ActualNETdifference=(newextraG-oldextraG-fee*(newextraQ-oldextraQ))/scale. Matched=(tnew*newextraG-told*oldextraG-fee*(tnew*newextraQ-told*oldextraQ))/scale,where t=minbudget/owncost. Group common/added/removed/neither. Report reward,cost,NET separately;matchedcostsumzero perdate. Perrow/call/date/seed identities conserve.',interpretation='Commonselected extra batches have identical actualgain/query;only thinningprobabilitieschange. Distinguish retainedcommonbenefit at equalexpectedbudget from discoveryofnewprofitableaction. Analyticcomparison remains prespecified,notnewrandompolicy/realhardbudget/time.',recurrence='Removedunprofitable vsremovedprofitable states keyed(sample_id,baseline_stop_position),notinitialprefix alone. Count1/2/3seed recurrence anduniquephysicalcalls;sharedteacherdata notindependent successes. Allnew12unprofitable and6newlostprofit pairs retained.',focus='Explain seed1matchedpositive/actualnegative andseed2matchednegative. Showallseeds,dates,calls,toppositiveandnegative withfullledgers;no selection for futuretraining ortrigger.',limits='No new predictions/fits/NNforward/physics/data/final8/paper/newFRAME/fees/rules. No depthscanorposthocpolicycombinations. Allteacherpath dataexposed exploratory.'))
    files=[W/'run.py',O/'design.json',R/'work/grid210_amount_depth/run.py',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid113_relational_learning/train_metadata.json']
    for d in [NEW,OLD]:files += [d/n for n in ['complete.json','independent_audit.json','gate_predictions.json','oof_trajectories.json','search_metrics.json','family_summary.json','decision.json','changed_choices.json']]
    files += [NEW/'expected_budget_comparison.json',NEW/'all_actual_date_seed_differences.json'];save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID211_PLANNED')
def evaluate():
    verify();assert not (O/'started.json').exists();save('started.json',dict(new_fits=0,new_predictions=0));meta,lookup,cases=C.B.load();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];new={(r['method'],r['sample_id']):r for r in read(NEW/'oof_trajectories.json')};old={(r['method'],r['sample_id']):r for r in read(OLD/'oof_trajectories.json')};pn={(p['seed'],p['sample_id']):p for p in read(NEW/'gate_predictions.json')};po={(p['seed'],p['sample_id']):p for p in read(OLD/'gate_predictions.json')};budget=read(NEW/'expected_budget_comparison.json')['GRID208'];dates=sorted({c['date'] for c in cases});assert len(dates)==16;ledger=[];date_rows=[];repeat=collections.defaultdict(list);all_fields=['actual_gain_NET','actual_cost_NET','actual_NET','matched_gain_NET','matched_cost_NET','matched_NET']
    for row in budget['per_date_seed']:
        s,d=row['seed'],row['date'];cc=[c for c in cases if c['date']==d];N=len(cc);weight=1/(16*N);tn,to=row['NEW']['keep_probability'],row['OLD']['keep_probability'];rr=[]
        for c in cc:
            sid=c['sample_id'];b=new[f'NET_seed{s}',sid];bn=old[f'NET_seed{s}',sid];a=new[f'GATE_seed{s}',sid];o=old[f'GATE_seed{s}',sid]
            for k in ['gain','queries','chosen_pool_id','actions','baseline_stop']:assert b[k]==bn[k]
            gn,go=a['gain']-b['gain'],o['gain']-b['gain'];qn,qo=a['queries']-b['queries'],o['queries']-b['queries'];an,ao=a['append'],o['append'];group='COMMON' if an and ao else 'ADDED' if an else 'REMOVED' if ao else 'NEITHER'
            if group=='COMMON':assert gn==go and qn==qo
            actual_gain=(gn-go)/scale;actual_cost=-fee*(qn-qo)/scale;matched_gain=(tn*gn-to*go)/scale;matched_cost=-fee*(tn*qn-to*qo)/scale;key=s,sid;p=pn.get(key);qnext=p['actual_next_queries'] if p else 0;g=p['actual_next_gain'] if p else 0.;net=p['actual_next_NET'] if p else 0.;position=p['position'] if p else None
            if an or ao:
                assert p and pn[key]['position']==po[key]['position'];assert abs((g-fee*qnext)/scale-net)<1e-10
                if an:assert abs(gn-g)<1e-12 and qn==qnext
                if ao:assert abs(go-g)<1e-12 and qo==qnext
            r=dict(seed=s,sample_id=sid,position=position,date=d,call=c['physical_state_group'],group=group,selected_before=ao,selected_after=an,actual_next_gain=g,actual_next_queries=qnext,actual_next_NET=net,old_keep=to,new_keep=tn,weight=weight,actual_gain_NET=weight*actual_gain,actual_cost_NET=weight*actual_cost,actual_NET=weight*(actual_gain+actual_cost),matched_gain_NET=weight*matched_gain,matched_cost_NET=weight*matched_cost,matched_NET=weight*(matched_gain+matched_cost));ledger.append(r);rr.append(r)
            if group=='REMOVED':repeat[sid,position].append(r)
        total={k:sum(r[k] for r in rr) for k in all_fields};assert abs(total['matched_cost_NET'])<1e-10;assert abs(total['matched_NET']-(row['NEW']['mean_net_utility']-row['OLD']['mean_net_utility'])/16)<1e-10;actual_expected=next(r for r in read(NEW/'all_actual_date_seed_differences.json')['GRID208'] if r['seed']==s and r['date']==d);assert abs(total['actual_NET']-actual_expected['NET_delta']/16)<1e-10;date_rows.append(dict(seed=s,date=d,new_keep=tn,old_keep=to,counts=dict(collections.Counter(r['group'] for r in rr)),**total))
    assert len(ledger)==7134;seed_rows=[];call_rows=[]
    for s in range(3):
        rr=[r for r in ledger if r['seed']==s];groups={g:dict(pairs=sum(r['group']==g for r in rr),**{k:sum(r[k] for r in rr if r['group']==g) for k in all_fields}) for g in ['COMMON','ADDED','REMOVED','NEITHER']};totals={k:sum(r[k] for r in rr) for k in all_fields};expected=next(r for r in budget['per_seed'] if r['seed']==s);rawexpected=next(r for r in read(NEW/'decision.json')['actual_per_seed'] if r['seed']==s);assert abs(totals['matched_NET']-expected['NET_delta'])<1e-10 and abs(totals['actual_NET']-rawexpected['NET_delta'])<1e-10;seed_rows.append(dict(seed=s,groups=groups,totals=totals))
        for call in sorted({r['call'] for r in rr}):
            cr=[r for r in rr if r['call']==call];call_rows.append(dict(seed=s,call=call,date=cr[0]['date'],changed_pairs=sum(r['group'] in ['ADDED','REMOVED'] for r in cr),**{k:sum(r[k] for r in cr) for k in all_fields}))
    repeated=[]
    for key,rr in repeat.items():
        assert len({r['seed'] for r in rr})==len(rr) and len({r['call'] for r in rr})==1;assert len({(r['actual_next_gain'],r['actual_next_queries']) for r in rr})==1;repeated.append(dict(sample_id=key[0],position=key[1],call=rr[0]['call'],date=rr[0]['date'],seeds=sorted(r['seed'] for r in rr),seed_count=len(rr),profitable=rr[0]['actual_next_NET']>0,zero_gain=rr[0]['actual_next_gain']==0,actual_next_NET=rr[0]['actual_next_NET'],queries=rr[0]['actual_next_queries']))
    recurrence={}
    for label,good in [('removed_unprofitable',False),('removed_profitable',True)]:
        recurrence[label]={str(n):dict(unique_states=len(rr),pairs=sum(r['seed_count'] for r in rr),calls=len({r['call'] for r in rr}),dates=len({r['date'] for r in rr})) for n in [1,2,3] for rr in [[r for r in repeated if r['profitable']==good and r['seed_count']==n]]}
    removed=[r for r in ledger if r['group']=='REMOVED'];added=[r for r in ledger if r['group']=='ADDED'];assert len(removed)==157 and sum(r['actual_next_NET']>0 for r in removed)==6 and len(added)==12 and not any(r['actual_next_NET']>0 for r in added);lost=[r for r in removed if r['actual_next_NET']>0];summary=dict(per_seed=seed_rows,removed_state_recurrence=recurrence,removed_profitable_pairs=lost,removed_pairs=157,removed_zero_pairs=sum(r['actual_next_gain']==0 for r in removed),removed_positive_unprofitable_pairs=sum(r['actual_next_gain']>0 and r['actual_next_NET']<=0 for r in removed),added_pairs=12,added_profitable_pairs=0,all48_budget_cost_terms_cancel=True,new_fits=0,new_predictions=0,new_physics=0)
    save('all7134_contributions.json',ledger);save('all48_date_contributions.json',date_rows);save('all_call_contributions.json',call_rows);save('removed_state_recurrence.json',repeated);save('summary.json',summary)
    lines=['# GRID211：种子1的同预算改善和种子2损失来自哪里','','本轮仅拆解固定210与208策略，没有新预测、拟合或策略变体。原始策略与同预期查询量分别计算，所有7134条种子—起点轨迹、48个日期—种子预算配对都保留。','','## 精确分解口径','','原始差值由新增/取消补查的真实增量与已知费用变化构成。预算匹配仅对已经选中的追加按各自保留概率稀释，基线NET完整保留。分为双方共同追加、新版新增、旧版被取消三组，分别列奖励和费用贡献。日期内匹配后总费用差为0，但调用间、组间的费用分配仍会改变。下列“共同追加”收益变化来自保留概率不同，并非找到了新候选。','','|种子|组|配对|原始净效用差|同预算奖励项|同预算费用项|同预算净效用差|','|---|---|---:|---:|---:|---:|---:|']
    for sr in seed_rows:
        for g,z in sr['groups'].items():
            if g!='NEITHER':lines.append(f"|{sr['seed']}|{g}|{z['pairs']}|{z['actual_NET']:+.9f}|{z['matched_gain_NET']:+.9f}|{z['matched_cost_NET']:+.9f}|{z['matched_NET']:+.9f}|")
        z=sr['totals'];lines.append(f"|{sr['seed']}|合计||{z['actual_NET']:+.9f}|{z['matched_gain_NET']:+.9f}|{z['matched_cost_NET']:+.9f}|{z['matched_NET']:+.9f}|")
    lines+=['','## 取消查询的跨种子重复性','', '按“同一初始前缀＋相同停止位置”识别同一状态；不能仅按初始前缀配对，因为不同种子可能在不同批次停止。这些状态仍共享教师调用，不是独立成功场景。','', '|取消类别|重复种子数|不同状态|配对数|物理调用|日期|','|---|---:|---:|---:|---:|---:|']
    for label,groups in recurrence.items():
        for n,z in groups.items():lines.append(f"|{label}|{n}|{z['unique_states']}|{z['pairs']}|{z['calls']}|{z['dates']}|")
    lines+=['',f"157个取消配对中，{summary['removed_zero_pairs']}个零奖励、{summary['removed_positive_unprofitable_pairs']}个微小正奖励未回本、6个回本。新版12个新增追加均未实际回本。不能因取消了很多无益查询就忽略少数有利机会，也不能因有损失就丢弃有效取消的线索。",'', '## 六个新增取消的回本配对','', '|种子/调用|前缀前8位/停止位置|查询|实际局部净收益|原始宏平均净效用贡献|','|---|---|---:|---:|---:|']
    for r in lost:lines.append(f"|{r['seed']}/{r['call']}|{r['sample_id'][:8]}/{r['position']}|{r['actual_next_queries']}|{r['actual_next_NET']:.6f}|{r['actual_NET']:+.9f}|")
    lines+=['','## 各种子主要正反调用','']
    for s in range(3):
        rr=[r for r in call_rows if r['seed']==s];lines.append(f'种子{s}，按原始策略贡献：')
        for label,cr in [('正向',sorted(rr,key=lambda r:-r['actual_NET'])[:3]),('反向',sorted(rr,key=lambda r:r['actual_NET'])[:3])]:
            for r in cr:lines.append(f"- {label} {r['call']}：原始净效用{r['actual_NET']:+.9f}，同预算净效用{r['matched_NET']:+.9f}；改变{r['changed_pairs']}个相关配对。")
        lines.append('')
    lines+=['所有调用、日期和原始/匹配奖励与费用项另存，不只保留以上极值。','', '## 后续研究边界','', '预算匹配的改善可以体现查询利用效率，但共同批次保留率提高不等于学到了新有利动作。后续应优先追踪重复出现的有效取消与少数高代价误删是否能从公开状态区分，先核查拟合支持和样本覆盖；不能将本轮事后收益、日期或编号变成触发条件。','', '继续保留208较均衡候选和210低查询端，不再扫深度、不自动拼接策略或训练路由器。没有新增物理、最终8、论文或原FRAME比较；当前局部奖励与查询代理效用不等于完整周质量和实际时延。']
    (O/'GRID211_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');verify();save('review.json',dict(passed=True,all7134_actual_and_matched_contributions_reconstructed=True,all48_cost_cancellations_and_seed_metrics_reproduce=True,same_state_recurrence_requires_same_stop_position=True,all157removed_12added_6lost_kept=True,new_fits=0,new_predictions=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,new_predictions=0,new_NET_forwards=0,new_physics=0,final8_absent=True));archive('complete');dest=R/'论文协作资料/FRAME_预算改善来源_GRID211';dest.mkdir(exist_ok=False)
    for n in ['GRID211_report.md','summary.json','review.json','complete.json','design.json']:shutil.copy2(O/n,dest/n)
    print('GRID211_COMPLETE',json.dumps(dict(seeds=seed_rows,recurrence=recurrence,lost=lost),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
