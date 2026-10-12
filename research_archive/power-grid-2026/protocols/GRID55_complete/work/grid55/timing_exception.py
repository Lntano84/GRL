"""Keep raw frozen results; document external standby and a pending repair plan."""
import datetime,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid55'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=read(OUT/'design.json');R=read(OUT/'review.json');assert R['passed_saved_data'] and R['runs']==128
raw_files=[OUT/n for n in ['design.json','code_freeze.json','finished.json','review.json','version_selection.json','GRID55_report.md']]
before={p.name:sha(p) for p in raw_files}
events=read(OUT/'windows_power_events.json')
enters=[e for e in events if e['event_id']==506];exits=[e for e in events if e['event_id']==507]
assert enters and exits
dt=lambda s:datetime.datetime.fromisoformat(s)
standby_pairs=[]
for entered in sorted(enters,key=lambda e:e['time_utc']):
    matching=sorted((e for e in exits if dt(e['time_utc'])>dt(entered['time_utc'])),key=lambda e:e['time_utc'])
    if matching:
        left=matching[0]
        standby_pairs.append({'enter':entered,'exit':left,'duration_s':(dt(left['time_utc'])-dt(entered['time_utc'])).total_seconds()})
longest=max(standby_pairs,key=lambda r:r['duration_s']);enter=longest['enter'];exit=longest['exit'];standby_s=longest['duration_s'];assert standby_s>1000
affected=[]
for index,entry in enumerate(D['execution_order']):
    fp=OUT/'runs'/f"r{entry['repeat']}__{entry['week']}__{entry['method']}"
    summary=read(fp/'summary.json')
    ended=datetime.datetime.fromtimestamp((fp/'summary.json').stat().st_mtime,tz=datetime.timezone.utc)
    started=ended-datetime.timedelta(seconds=summary['wall_s'])
    overlap=sum(max(0.,(min(ended,dt(p['exit']['time_utc']))-max(started,dt(p['enter']['time_utc']))).total_seconds()) for p in standby_pairs)
    if overlap>0:
        affected.append({'index':index,'week':entry['week'],'repeat':entry['repeat'],'method':entry['method'],
                         'control_wall_s':summary['controller_s'],'control_cpu_s':summary['controller_cpu_s'],
                         'standby_overlap_s':overlap,'run_start_utc_estimated':started.isoformat(),'run_end_utc':ended.isoformat()})
assert {r['week'] for r in affected}=={'2035-11-12_9'}
assert {r['index'] for r in affected}=={125,127},affected
item=D['execution_order'][125];assert item['method']=='COMPRESSED_GNN1' and item['repeat']==1
folder=OUT/'runs'/f"r1__{item['week']}__{item['method']}"
S=read(folder/'summary.json');C=read(folder/'components.json')
largest=max(C['solver_records'],key=lambda r:r['wall_s'])
assert largest['step']==1524 and largest['wall_s']>1000
assert S['controller_s']-S['controller_cpu_s']>1000
assert all(r['same_path'] for r in R['repeat_checks'])
status={'numeric_saved_data_audit_passed':True,'runs_completed':128,'date_families':8,
        'repeat_paths_exact':len(R['repeat_checks']),'timing_qualification':'UNRESOLVED_EXTERNAL_STANDBY',
        'version_selection_usable':False,'raw_version_selection_preserved':read(OUT/'version_selection.json'),
        'affected_run':{'index':125,'week':item['week'],'repeat':1,'method':item['method'],
                        'control_wall_s':S['controller_s'],'control_cpu_s':S['controller_cpu_s'],
                        'largest_solver_record':largest},
        'all_affected_runs':affected,
        'external_events':{'enter':enter,'exit':exit,'event_interval_s':standby_s,'all_standby_pairs':standby_pairs,
                           'source':'System/Microsoft-Windows-Kernel-Power modern standby events506/507',
                           'source_sha256':sha(OUT/'windows_power_events.json')},
        'new_fits':0,'additional_runs_started':0,
        'interpretation':'Standby overlap and >1000s wall/CPU discrepancy make this run unsuitable for an ordinary algorithm-speed interpretation. No duration subtraction, selective result deletion, corrected score, or repeated completed run performed. Cost/path evidence stays valid; joint final speed/version decision unresolved.'}
(OUT/'review_status.json').write_text(json.dumps(status,indent=2,allow_nan=False),encoding='utf-8')
proposal={'status':'PROPOSED_NOT_EXECUTED_REQUIRES_USER_DECISION',
          'reason':'Frozen128-run/no-repeat budget complete; two runs in one timing repetition overlap external standby and cannot decide final speed fairly.',
          'week':item['week'],'new_repeat_id':2,'implementations':[m['method'] for m in D['methods']],
          'new_runs':8,'expected_wall_minutes':[15,20],'wall_cap_s':2400,'new_fits':0,
          'unchanged':'Same frozen weights,models,controller,prior cache,fresh-field validation,legal inputs,masks,stop,timers,physicalbackend. All8implementations measured together;no new baselines.',
          'data_rule':'Keep original128/raw audit unchanged. New8form a supplemental complete paired repetition. If authorized, publish raw results plus explicitly repaired128-cell timing view using original7date repeat0/1 and affecteddate repeat0/2 for ALL methods,not only replacing the slow seed. No unseen/independent-confirmation claim.',
          'qualification':'Prevent system idle standby for this bounded run without changing the permanent power plan; check power-event log. Compare each supplemental physical path/cost/query trace against its historical counterpart. New failure/nondeterminism/standby makes timing qualification unresolved,not a license to keep repeating.',
          'decision_rule':'Apply exactly the prior frozen symmetric quality/time version-selection rule to qualified paired timing view;retain all model seeds. Do not choose fastest seed,delete failure,subtract sleep duration,or use CPUtime as replacement primary metric.'}
(OUT/'TIMING_REPAIR_PROPOSAL.json').write_text(json.dumps(proposal,indent=2),encoding='utf-8')
lines=['# GRID55 结果资格：成本可读，最终速度选型待修复','',
       '**请先读本页，再读原始GRID55_report.md。原审计核对通过不表示受待机影响的墙钟具有正常算法比较资格。**','',
       '128/128回放全部完成2017步。保存数据核验通过：258176物理步、853352公开预测；64/64计时重复物理路径一致。没有新训练，全部原新/对照种子保留。','',
       '| 方法 | 完整周最大成本增幅（相对FULL） |','|---|---:|']
for method in ['LOCAL_GNN0','LOCAL_GNN1','COMPRESSED_GNN0','COMPRESSED_GNN1','SOFTGNN0','SOFTGNN1','LOCAL_PPO']:
    s=next(r for r in R['summaries'] if r['method']==method)
    lines.append(f"| {method} | {100*s['max_complete_week_cost_regret']:.5f}% |")
lines+=['','这是同日期配对质量证据：原版较快但一个日期超过原0.5%投资尺度；新版最大增幅0.08045%。0.5%不是等价检验或会议要求。','',
        '## 外部待机污染','',
        f"Windows现代待机事件506/507：{enter['time_utc']}至{exit['time_utc']}，约{standby_s/60:.2f}分钟。对应worker125，r1/2035-11-12_9/COMPRESSED_GNN1。控制墙钟{S['controller_s']:.2f}s，控制CPU{S['controller_cpu_s']:.2f}s；第1524步OSQP调用墙钟{largest['wall_s']:.2f}s，记录的编译墙钟{largest['compilation_s']:.2f}s，后端求解{largest['backend_s']:.2f}s。",'',
        '随后两个较短待机区间（约69.82秒、10.49秒）还与该日期的PPO重复1重合。按事件区间与原运行起止时间核对，受影响的是worker125（新版种子1）与127（PPO），共两条，而非只有一条。','',
        '电源事件与运行区间及异常墙钟/CPU差吻合。不能把原报表里新版种子1的巨大负提速解释为模型退步，也不能精确减去待机时长伪造正常控制成绩。物理路径、成本与另一重复一致，质量结论不受这一停顿影响。','',
        '旧版未经该待机污染的宏平均控制时间减少18.453%/18.625%；新版种子0为14.388%。这些值保留可读，但不能只用新版种子0代替两个种子的最终选型。原RAW投资筛查和NO_CLEAR_PREFERRED_VERSION保留，仅作为含异常数据的冻结计算，不作为科研选型结论。','',
        '## 待用户决定的最小计时修复','',
        '仅为该日期补一次完整八臂计时（新的repeat2，约15–20分钟），不覆盖原128条，模型/权重/规则不变。全部八臂共同补测，避免只重复一个慢结果。届时原始表与注明异常修复的配对表并列呈现，质量/路径/查询逐项对账；新记录若仍受外部待机影响，不自动重复或增加预算。','',
        '本批自动跟进明确要求不重复已完成条目、不得修改冻结预算，因此当前没有自动执行这八条。proposal文件为PROPOSED_NOT_EXECUTED_REQUIRES_USER_DECISION。']
(OUT/'GRID55_review_status.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (OUT/'RUN_STATE.md').open('a',encoding='utf-8') as f:
    f.write('TIMING QUALIFICATION UNRESOLVED:external standby overlappedworkers125and127;raw128andcost/pathaudit retained. Supplemental8only proposed,not started;human decision needed.\n')
assert all(sha(p)==before[p.name] for p in raw_files)
print(json.dumps({'saved_data_audit':True,'runs':128,'repeat_paths':len(R['repeat_checks']),
                  'external_standby_minutes':standby_s/60,'affected_indices':[r['index'] for r in affected],'timing_version_selection':'UNRESOLVED',
                  'additional_runs_started':0,'raw_hashes_preserved':True}))
