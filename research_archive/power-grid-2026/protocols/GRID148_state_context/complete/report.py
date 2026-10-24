"""Scope checks, balanced interpretation and publication; no training."""
import json,gzip,shutil
from pathlib import Path
import run as D
O,R,REPO=D.O,D.R,D.REPO

def main():
 D.verify();assert D.read(O/'state.json')['stage']=='compared_pending_report'
 ex=D.read(O/'extraction_review.json');cv=D.read(O/'comparison_review.json');assert ex['passed'] and cv['passed']
 sums=D.read(O/'neighbor_summary.json');dates=D.read(O/'neighbor_per_date.json');cases=D.read(O/'case_context_comparisons.json')
 rows=D.G.rowfile(D.B.O/'fit_rows.json.gz')+D.G.rowfile(D.B.O/'calibration_rows.json.gz')+D.B.eligible_data('validation');parts=['fit']*386+['calibration']*198+['validation']*142
 with gzip.open(O/'public_state_descriptors.jsonl.gz','rt',encoding='utf8') as f:features=list(map(json.loads,f))
 assert len(rows)==len(features)==726
 for s in sums:
  dd=[d for d in dates if d['view']==s['view'] and d['partition']==s['partition']]
  for key in ['continue_rows','continue_full5','continue_full5_zero_continue','stop_full5','stop_full5_has_continue']:assert sum(d[key] for d in dd)==s[key]
 pressure=[]
 for part in ['fit','calibration','validation']:
  for label in [True,False]:
   ii=[i for i,r in enumerate(rows) if parts[i]==part and bool(r['target']['stop_label'])==label]
   pressure.append(dict(partition=part,stop_label=label,rows=len(ii),zero_incident_overload=sum(features[i]['features']['incumbent_overload_endpoint_share']==0 for i in ii)))
 D.save('both_label_pressure_counts.json',pressure)
 history=[R/f'outputs/{folder}/{name}' for folder,name in [('grid99_feature_learning','GRID99_report.md'),('grid110_structure_audit','GRID110_report.md'),('grid113_relational_learning','GRID113_report.md'),('grid116_matched_module','GRID116_report.md'),('grid123_event_auxiliary','GRID123_report.md')]]
 assert all(p.exists() for p in history)
 D.save('historical_report_sources.json',{p.relative_to(R).as_posix():D.sha(p) for p in history})
 D.save('input_coverage.json',dict(saved_observation_fields=D.read(O/'decoded_schema.json')['observation_fields'],
  fields_used_by_this_diagnostic=['rho','line_status','topo_vect'],
  public_other_inputs=['candidate order','alias-to-representative mapping','candidate substation ownership','visible outcomes','static endpoint/topology-slot mapping'],
  saved_but_not_encoded_here=['generator/load per-device values','dispatch/storage/curtailment fields','cooldowns/maintenance','full base_action1605 commands'],
  limitations='Only this prespecified12-summary geometry was tested. No conclusion that full state or all structural representations are uninformative;previous relational training used a different net-gain target.'))
 D.save('decision.json',dict(new_training_supported_by_this_geometry=False,new_fit=False,new_policy=False,ready_to_scale=False,
  conclusion='Public state/geometry is recoverable and distinguishes the selected adverse pair,but this12-feature summary worsens held-out neighborhood support overall. Do not train30-input forest or larger network automatically.',
  next='Keep originalFRAME and controls. If continuing learning,first inventory current-observation/base-action/paid-feedback information actually omitted by each historical input contract and predefine a new-target test;do not repeat old net-gain structural training or claim the current diagnostics prove impossibility.',
  old_labels_models_cutoffs_unchanged=True,final8_absent=True))
 error=next(c for c in cases if c['call_id']=='2037-03-02_100|call009');err=error['contexts'][0];pos=error['contexts'][1]
 assert err['state_features']['incumbent_hops_to_maxrho_line']==1 and pos['state_features']['incumbent_hops_to_maxrho_line']==0
 get=lambda p,v:next(s for s in sums if s['partition']==p and s['view']==v)
 lines=['# GRID148：当前电网状态对原动作保持标签的帮助','',
  '结论：状态和候选空间能够可靠对应，物理上下文能区分部分原摘要相似的案例；但本轮12项汇总描述没有带来稳定的几何区分改善。当前不支持直接训练“原18项＋新12项”，也不据此扩大网络。没有新训练、策略执行或物理仿真。','',
  '## 先接上已有工作','',
  'GRID98/99已经给局部补查添加结构/压力摘要，目标是取消补查后的轨迹等价，未形成稳定改善。GRID110还原了观测和动作空间；GRID111检查动作连接与已知反馈关联；GRID113/116/123进一步做过关系表示、保留骨干模块或辅助事件训练，旧净收益目标下未建立稳定增量。本轮复用这些工程和失败结论，不把“发现可解码状态”当新贡献。','',
  '当前问题则是：在原FRAME全局搜索的可停检查点，提前停止并按原条件补查，能否保持原最终动作且严格少查。旧净收益、轨迹等价、当前动作保持是不同标签；不能直接把某个旧目标的正例当当前目标的正例，也不能因旧模型失败就宣布任何状态输入都无效。','',
  '## 字段和时序怎样核验','',
  '复用GRID03序列化的观测/动作空间，按安装版本Grid2Op1.12.5原生反序列化。原生字段顺序与安装源码AST、GRID110记录一致；305次状态的观测4460维和基础动作1605维全部往返一致，状态哈希与原搜索记录相同，并沿用GRID147已核对的before-step来源。118站、186线、540拓扑位置与旧元数据逐项对应，909个公共动作所属站和命令哈希全部一致。','',
  '[Grid2Op官方观测说明](https://grid2op.readthedocs.io/en/dev-switches/user/observation.html)将rho定义为电流/热限，line_status表示线路通断，topo_vect给出设备所在母线。本轮只取这些已存在的当前字段，以及公开候选、别名和已查询反馈；没有调用环境构造、reset、step、simulate或forecast。','',
  '图按当前母线连接构造：每站的1/2号母线分开，在线线路按真实端点母线连接；同站不同母线没有自动连边。站到高负载线路的距离取其当前母线到线路端点的最少边数。它描述当前连接关系，不是电气灵敏度，也不能预测执行候选后的潮流。','',
  '## 固定检查了哪些信息','',
  '12项描述在读取结果前固定：当前过载线数和过载量；当前最佳站所接在线线路最大rho、全网过载端点占比；最佳站到最高负载线/全部过载线的当前母线图距离；已查可接纳动作中涉及过载端点的比例及最好奖励差；公开下一批涉及这些端点的比例和平均图距离；最佳站相对前缀、前一批是否改变。','',
  '候选覆盖按原别名关系核对，一个代表查询不重复计费；当前最佳站仍按原规范代表所属站确定。下一批只看公开命令/编号，不取其未查询结果。没有可达路径时保留缺失标记，不填一个猜测距离。母线1/2整体交换不改变描述；未来标签作为额外参数会被拒绝。','',
  '12项并非12个全新机制：其中局部压力等已有历史工作，当前新检查在于它们对动作保持目标的作用。状态的其它字段和基础动作命令尚未进入这套描述，不能把本轮结果扩大成“电网状态没有用”。','',
  '## 关键案例能区分，全部记录没有稳定改善','',
  '主误停2037-03-02_100|call009与原18项下最近的正确停止拟合状态，站内候选/未查数量等原结构摘要相同；新的物理关系不同。','',
  '| 原先相似的一对 | 最佳站接线最大rho | 是否直接接当前过载线 | 到最高负载线端点图距离 |','|---|---:|---|---:|',
  f"| 误停状态 | {err['state_features']['incumbent_incident_max_rho']:.6f} | 否 | 1 |",
  f"| 最近正确停止拟合状态 | {pos['state_features']['incumbent_incident_max_rho']:.6f} | 是 | 0 |",'',
  '该误停在原18项下的5个最近拟合搜索都为停止标签；新状态描述或合并描述下，有2个继续标签进入邻域。这是对这一对输入的可观测区分，不是新模型已经识别成功。全部预定案例、近邻及不利变化一并保留。','',
  '比较范围是全部726个原检查点：386拟合、198校准、142开发验证。固定三种视图：原18项、状态12项、合并30项，分别补缺失标记。只用386拟合行计算中位数/IQR，按相同查询深度和初始分支找跨日期最近5次不同拟合搜索，没有拟合近邻分类器或选择阈值。','',
  '| 分区/视图 | 继续状态中有5个近邻 | 5近邻均无继续标签 | 停止状态中有5个近邻 | 其中混入继续标签 |','|---|---:|---:|---:|---:|']
 for p in ['fit','calibration','validation']:
  for v in ['ORIGINAL18','STATE12','COMBINED30']:
   s=get(p,v);lines.append(f"| {p}/{v} | {s['continue_full5']} | {s['continue_full5_zero_continue']} | {s['stop_full5']} | {s['stop_full5_has_continue']} |")
 lines += ['',
  '开发验证的22个继续状态都有5个可比近邻。原描述有7个邻域没有任何继续标签，新状态和合并描述均增至11个；校准从11/45增至14/45或13/45。拟合几何看似改善，未在这两组暴露记录上保持。停止状态邻域混入继续标签也增加，不是只提高继续识别而付出少量代价的单向结果。','',
  '较少邻域空白不等于分类性能，更不等于闭环质量；上述反向变化也不证明所有非线性模型都学不会。但本次预定诊断没有给“直接加12列训练”的正面资格，不按有利单例采用新描述或重新加权距离。','',
  '训练/校准/开发分别有18/15/1行不足5个匹配搜索，三视图相同，全部保留并明确排除在“完整5邻域”计数之外。每日期结果另存；同调用前缀和同日期副本不是独立场景。','',
  '## 局部没有过载不能直接作为停止条件','',
  '开发验证22个必须继续的前缀中，21个当前最佳站没有接在线过载线路，来自16次搜索、4个日期；120个可停止前缀中也有106个如此。它在两类里都很普遍，不能据“本站不拥堵”直接决定停止或继续。训练和校准的全部两侧计数也保存。','',
  '这与GRID110在旧净收益标签下的反例方向相容，但此处是重新核对当前动作保持标签，不是复用旧标签得出相同结论。当前站与拥塞线路距离近，也不是未来候选必然有收益的因果证据。','',
  '## 当前判断与后续边界','',
  '保留原FRAME分批机制、RF140和固定预算对照，不采用本轮描述作为新模型输入，不训练30维森林，不扩大网络。物理上下文能解释个案，尚不能提供稳定的整体可分性证据。','',
  '若继续学习，下一步应先盘点各历史输入契约实际包含了什么：当前观测、共同基础动作、动作作用位置和已揭示后果如何对应，哪些信息尚未进入当前18项模型，哪些已经在旧任务试过。然后再针对当前标签提出可验证方案，避免重复做过的结构学习。当前数据已保存，不需要为字段恢复再采集。','',
  '本轮只用了观测中的rho、line_status、topo_vect；机组/负荷分设备值、调度/储能/限发、冷却/检修，以及基础动作命令均未编码。本轮否定的是这套固定汇总描述的直接训练依据，没有证明完整状态或其它表示无效。任何后续训练仍需另行冻结并提前通知，最终8测试不动。','',
  '## 核验','',
  f"{ex['serialized_vector_roundtrips']}次完整向量往返、909个动作命令、915项原生字段切片、{ex['graph_station_distance_checks']}项图距离的独立边松弛核对通过。726个隐藏后缀隔离、母线标签交换和未来参数拒绝检查通过；2178条三视图邻域记录及{cv['checks']['independent_scalar_distances']}项独立标量距离核对通过。",'',
  '原模型、分数、日期、排序、预算、动作标签和所有旧结果保持。没有独立AC重求解、候选预测、新物理步骤、模型拟合、学习推理、策略回放或论文图表修改。']
 (O/'GRID148_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
 D.save('review.json',dict(passed=True,extraction=ex,comparison=cv,all_date_counts_reconciled=True,
  historical_targets_distinguished=True,saved_but_unused_state_fields_disclosed=True,
  old_sources_unchanged=True,new_fits=0,new_physical_calls=0,new_policy_replays=0,final8_absent=True))
 D.save('complete.json',dict(passed=True,new_fits=0,new_physical=0,new_policy=0,training_supported=False,ready_to_scale=False,final8_absent=True))
 D.save('state.json',dict(stage='complete'));D.verify();D.archive('complete')
 c=R/'论文协作资料/FRAME_电网状态核查_GRID148';c.mkdir(exist_ok=False)
 names=['GRID148_report.md','design.json','decoded_schema.json','input_coverage.json','descriptor_profile.json','neighbor_summary.json','neighbor_per_date.json',
  'case_context_comparisons.json','both_label_pressure_counts.json','review.json','decision.json','complete.json']
 for n in names:shutil.copy2(O/n,c/n)
 msg='GRID148复用GRID03/98/110空间与历史失败结论，核查当前动作保持标签的12项物理/母线图/已知反馈描述。305状态610往返、909命令、71980图距离及726后缀/母线交换/未来参数拒绝通过；无环境/预测/训练。当前误停个案原邻域0继续→新2继续，但全部开发22继续空白邻域7→11、校准11→13，停止侧也更混，不支持直接30项训练。当前站无在线过载出现于开发21/22继续与106/120停止，不能作停止充分条件。所有视图/日期/低支持行/反例保留。仅用rho/status/topo，不冒称完整物理输入无效；下一步先盘点历史输入契约与基础动作/状态/反馈对应，不重复旧净收益模型，不自动训练/扩大/最终8。报告research_archive/power-grid-2026/protocols/GRID148_state_context/complete/GRID148_report.md'
 entry='\n\n## 2026-10-24 — GRID148当前电网状态核查完成\n\n'+msg+'\n'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
  if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — GRID148后

1. 保留原FRAME、RF140和全部简单/失败对照；不直接把12项物理/图汇总接成30项新模型，不按个案选特征或调距离。
2. 若继续，先盘点历史学习输入契约：当前观测、基础动作、公开候选意图和已揭示反馈的对应关系；区分GRID99轨迹等价、GRID110–123净收益与当前动作保持目标，避免重做旧实验。
3. 当前仅核查rho/line_status/topo_vect相关12项，不声称完整状态无效。字段可恢复但整体邻域没有改善；未形成稳定依据前不训练或扩大。
4. 任何新训练另行冻结并提前通知；保持原标签/预算/日期/种子/对照和不利结果。无新物理/最终8/RL/论文图表。
''',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID148电网状态与当前标签核查\r\n\r\n[报告](protocols/GRID148_state_context/complete/GRID148_report.md)；空间可恢复，12项汇总没有稳定几何改善，不启动新训练。\r\n\r\n'.encode('utf8')+p.read_bytes())
 count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),old_sources_verified=True,zero_new_fits=True))
 d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json')
 (d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
 print('GRID148_COMPLETE',count)

if __name__=='__main__':main()
