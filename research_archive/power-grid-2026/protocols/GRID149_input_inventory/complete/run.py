"""Input/target inventory and saved base-command audit, with no environment or fit."""
from pathlib import Path
import collections,gzip,hashlib,importlib.util,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid149_input_inventory'
spec=importlib.util.spec_from_file_location('grid149_prior',R/'work/grid148_state_context/run.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
G=P.G;B=P.B;REPO=P.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID149_input_inventory'
read=P.read;sha=P.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive():
 d=A/'complete';d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
 assert not O.exists();O.mkdir();P.verify();assert read(P.O/'complete.json')['passed']
 folders=['grid101_gain_learning','grid102_search_rl','grid107_direct_value','grid113_relational_learning','grid133_nonlinear_gate','grid140_global_compression','grid142_batch_representation','grid144_class_mass','grid146_single_entry','grid148_state_context']
 paths=[R/f'outputs/{p}/design.json' for p in folders]
 paths+=[R/'work/grid101_gain_learning/features.py',R/'work/grid102_search_rl/observations.py',R/'work/grid112_relational_plan/schema.py',R/'work/grid110_structure_audit/extract.py',R/'work/grid28/run.py',P.O/'decoded_schema.json',P.O/'input_coverage.json',R/'outputs/grid147_generalization_diagnosis/saved_state_inventory.json',Path(__file__)]
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
 inv=read(R/'outputs/grid147_generalization_diagnosis/saved_state_inventory.json')['records'];sc=read(P.O/'decoded_schema.json');bounds=np.r_[0,np.cumsum(sc['action_shapes'])]
 assert bounds[-1]==1605;fields=sc['action_fields'];records=[];hashes={};byfield=collections.defaultdict(list)
 for r in inv:
  p=R/r['observation_file'];hashes[p.relative_to(R).as_posix()]=sha(p)
  with np.load(p) as z:
   v=z['base_action'];assert v.shape==(1605,)
   h=hashlib.sha256(v.tobytes()).hexdigest();commands={}
   for name,a,b in zip(fields,bounds[:-1],bounds[1:]):
    x=v[a:b];neutral=-1. if name=='_curtail' else 0.;active=x!=neutral
    commands[name]=dict(active_count=int(active.sum()),l1=float(abs(x[active]-neutral).sum()),minimum=float(x.min()) if len(x) else None,maximum=float(x.max()) if len(x) else None)
    byfield[name].append(commands[name]['active_count'])
  records.append(dict(call_id=r['call_id'],date=r['date'],partition=r['partition'],base_hash=h,commands=commands))
 for p,h in hashes.items():assert sha(R/p)==h
 save('state_sources.json',hashes);save('base_command_records.json',records)
 base=dict(calls=305,distinct_vectors=len({r['base_hash'] for r in records}),fields={k:dict(calls_with_explicit_command=sum(v>0 for v in vv),max_active_count=max(vv)) for k,vv in byfield.items()},
  neutral_contract='SerializedActionIDF2023:zero line/topology/redispatch/storage/alert commands;curtail=-1 is unset. Raw active commands,not realized power/flow. Verify defaults against serializer before physical interpretation.')
 save('base_command_summary.json',base)
 ds={p:read(R/f'outputs/{p}/design.json') for p in folders}
 assert len(ds['grid101_gain_learning']['feature_names'])==37 and len(ds['grid102_search_rl']['input']['features'])==41
 assert ds['grid140_global_compression']['features']==ds['grid144_class_mass']['features']==ds['grid146_single_entry']['features']
 inventory=[
 dict(stages='GRID101',target='Whole same-station immediate reward-gain vsquery fee',input='37 paid-prefix numeric summaries',raw_state='rho_limit/current baseline only indirectly',base_command='No explicit1605 command vector',pairing='Station subsets only,not full action-feedback tensors',source='work/grid101_gain_learning/features.py'),
 dict(stages='GRID102/103/107/120/123',target='Local next-batch net gain or sequential query utility',input='37 summaries +4 locked-station/batch/budget channels=41',raw_state='No full generator/load/topology vectors',base_command='No explicit continuous command vector;effects already in paid candidate feedback',pairing='No per-candidate connection tensor in base network',source='work/grid102_search_rl/observations.py'),
 dict(stages='GRID110/111',target='Diagnostic structure/paid reward association for local net gain',input='Public combined command partitions +observed feedback +current local pressure',raw_state='rho/line_status/topo recovered',base_command='Used for combined action reconstruction,not a complete continuous feature vector',pairing='Explicit correspondence inspected,no fit at110/111',source='work/grid110_structure_audit/extract.py'),
 dict(stages='GRID113/116/119',target='Local next-batch net gain',input='41 summary +68 per-action partition +8 feedback fields;paired vsfactorized controls',raw_state='Topology via intended local partitions;not full physical state',base_command='Topology effects included during combined-command construction;continuous base commands not explicitly encoded',pairing='Yes;already trained with adverse/mixed results under old target',source='work/grid112_relational_plan/schema.py'),
 dict(stages='GRID133',target='Local no-improvement/action-preservation gate',input='Original first37 numeric fields',raw_state='Not full state',base_command='Not explicit',pairing='No',source='outputs/grid133_nonlinear_gate/design.json'),
 dict(stages='GRID140/144',target='Earlier global STOP +original completion preserves originalFRAME action and reduces queries',input='18 summaries +18missing;class weights changed in144',raw_state='No explicit per-line physical context',base_command='Not explicit',pairing='No action-feedback tensors',source='outputs/grid140_global_compression/design.json'),
 dict(stages='GRID142',target='Same global action-preserving compression',input='Original18 +6batch reward/rho summaries +24missing',raw_state='Not full physical state',base_command='Not explicit',pairing='Numeric batch aggregation only;failed original screen',source='outputs/grid142_batch_representation/design.json'),
 dict(stages='GRID146',target='Same global target but only one learning exit at128',input='Original18 +18missing,153fit rows',raw_state='Not full state',base_command='Not explicit',pairing='No;failed original screen',source='outputs/grid146_single_entry/design.json'),
 dict(stages='GRID148',target='Current global label,descriptive-only geometry',input='12 physical/graph/visible-feedback summaries compared with18/combined30',raw_state='Onlyrho/line_status/topo',base_command='Not encoded',pairing='Summary geometry only,no trained model;no stable held-out improvement',source='outputs/grid148_state_context/design.json')]
 save('input_target_matrix.json',inventory)
 # Exact native API still available for a lower-capacity baseline,under a NEW task.
 save('untried_control.json',dict(candidate='One inherited logistic classifier on original18 +18missing global task',
  rationale='Past logistic experiments used local net-gain/trajectory targets. Current global action-preserving comparison hasRF140/142/144/146,but no matched regularized linear baseline.',
  inherited_parameters=ds['grid101_gain_learning']['logistic'],
  unchanged='GRID140386fit/198cal/142dev,12/4/4dates,original hierarchical date/call/prefix weights,original labels,search and strict empirical cutoff procedure',
  intervention='Classifier family and necessary fit-only standardization;not an equal-capacity architecture ablation or a new feature experiment.',
  no_fit_in_this_stage=True,no_claim_it_will_help=True))
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h
 save('review.json',dict(passed=True,contracts=9,base_vectors=305,saved_vectors_unchanged=True,fit_count=0,physical_calls=0,
  old_targets_not_merged=True,old_relational_failures_not_repeated=True,explicit_base_vs_reconstruction_distinguished=True))
 lines=['# GRID149：历史输入、目标和剩余对照核查','',
 '结论：不能把过去“学习失败”视为同一问题上的重复证据。不同阶段有轨迹等价、局部净收益和全局动作保持三类目标；结构配对已经在旧净收益目标上试过，当前全局目标仍缺一个同输入的正规低容量对照。','',
 '| 阶段 | 目标 | 输入 | 基础动作如何使用 |','|---|---|---|---|']
 for z in inventory:lines.append(f"| {z['stages']} | {z['target']} | {z['input']} | {z['base_command']} |")
 lines += ['',
 '旧关系模型并非完全没用基础动作：重建站内连接意图时已合并公共候选与基础动作。准确的缺口是连续调度/限发/储能等基础命令没有作为完整数值上下文输入，不能把这说成以前模型完全没有动作结构。已支付候选反馈也已经包含共同基础动作的运行后果，缺少显式命令不等于完全没有相应信息。','',
 f"305个既有基础动作向量有{base['distinct_vectors']}种精确字节取值。按序列化字段解码的活动命令计数另存；它们只是控制命令，未求解执行后功率或潮流，未据这些分布证明它们能预测标签。",'',
 '## 下一项可控实验','',
 '当前18项全局动作保持任务只试过森林及输入/权重/单入口变体，尚没有同输入、同日期、同目标的正则化逻辑回归。GRID101等旧逻辑回归预测的是另一个标签，不能当作这个对照已经做过。','',
 '因此下一步优先补一个低容量基线，直接检验森林拟合偏乐观的问题；不再给现有模型追加12项物理摘要，不重做旧关系网络。参数继承GRID101的C=1、lbfgs、最多2000次、种子20261022、无类别平衡，仅训练一次。标准化必须只用拟合行，原18项/缺失标记、原386分层权重、日期和动作标签保持。','',
 '这改变分类器及其必要预处理，不称同参数量消融。仍按原校准规则和完整搜索/固定256强对照评价，模型与阈值在开发验证评分前封存，失败后不补训。当前阶段没有执行拟合；下一阶段训练前通知。','',
 '所有历史坏结果和各自目标继续保留，基础动作是否值得新增输入仍是未验证问题，不用它为下一训练编造依据。最终8测试未生成。']
 (O/'GRID149_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
 save('complete.json',dict(passed=True,new_fits=0,new_physical=0,new_policy=0,final8_absent=True));archive()
 c=R/'论文协作资料/FRAME_输入目标对齐_GRID149';c.mkdir(exist_ok=False)
 for n in ['GRID149_report.md','input_target_matrix.json','base_command_summary.json','untried_control.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID149核对9组历史输入/目标，区分旧轨迹等价/局部净收益与当前全局动作保持；旧关系表示已训练，不重复。基础动作305向量活动字段仅做命令盘点，不声称可预测。当前全局18项任务缺少同输入逻辑回归对照；计划继承GRID101 C1/lbfgs/2000/种子，无类别平衡，一次新拟合，原日期/权重/标签/搜索校准保持，评分前封存。此阶段零训练/物理，训练前通知。报告research_archive/power-grid-2026/protocols/GRID149_input_inventory/complete/GRID149_report.md'
 entry='\n\n## 2026-10-25 — GRID149输入与目标对齐完成\n\n'+msg+'\n'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
  if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID149\n\n1. 补当前全局动作保持任务的同18项逻辑回归基线，继承旧固定参数，一次拟合；原分层权重和全部对照保持，训练前通知。\n2. 模型/阈值评分前推送；主筛查仍原动作一致+严格超过CAP256及逐日期查询约束，不按结果放宽。\n3. 不追加物理12项或重做旧关系网络，不把基础命令分布视为预测证据；最终8封存，无新物理/扩大。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID149输入与目标对齐\r\n\r\n[报告](protocols/GRID149_input_inventory/complete/GRID149_report.md)；不同学习目标分开，下一步补同任务低容量基线。\r\n\r\n'.encode('utf8')+p.read_bytes())
 print('GRID149_COMPLETE',json.dumps(base,ensure_ascii=False))

if __name__=='__main__':main()
