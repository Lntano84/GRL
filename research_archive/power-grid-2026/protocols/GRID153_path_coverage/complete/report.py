import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];runs=[x for x in D.read(O/'per_run.json') if x['repeat']==0];calls=[x for x in D.read(O/'call_coverage.json') if x['repeat']==0];s=D.read(O/'coverage_summary.json')
 errors=[]
 for folder,name,method in [('grid150_linear_control','all_new_model_failures.json','LR150_CALIBRATED'),('grid146_single_entry','all_single_entry_failures.json','RF146_ENTRY128_CALIBRATED')]:
  p=R/'outputs'/folder/name
  for e in D.read(p):
   if e['method']==method and e['partition']=='validation':
    hits=[x for x in calls if e['call_id'] in x['teacher_exact_hits']];errors.append(dict(method=method,teacher_call=e['call_id'],matched_actual_frame_sources=[x['source'] for x in hits],matched=bool(hits),source_sha256=D.sha(p)))
 assert len(errors)==2 and all(x['matched'] for x in errors);D.save('known_error_coverage.json',errors)
 lines=['# GRID153：实际FRAME与FULL教师路径覆盖','',f"四个已暴露开发场景的原FRAME共有{s['frame_calls']}次搜索，其中{s['exact_teacher_call_matches']}次（90%）与已有教师搜索具有逐字节相同的观测、基础动作和公开候选上下文。不能把近期教师开发集上的学习失败主要归因于路径来源不同。",'',
 '| 日期 | FRAME/教师步数 | FRAME/教师搜索次数 | 搜索状态精确重合 | 首次动作分歧步 |','|---|---:|---:|---:|---:|']
 for x in runs:lines.append(f"| {x['date']} | {x['frame_steps']}/{x['teacher_steps']} | {x['frame_calls']}/{x['teacher_calls']} | {x['exact_teacher_call_matches']}/{x['frame_calls']} | {x['first_difference']['action_hash'] if x['first_difference']['action_hash'] else '无'} |")
 lines+=['','两份计时重复的状态、动作、成本轨迹和搜索记录完全一致；这里只算四条场景轨迹，不算八条新场景。五月双方均在1390步结束，其余三日期均2017步；未完成场景保留，不把它当完整周质量通过。',
 '',f"共同时间范围内{s['common_horizon_steps']}个控制步有{s['same_before_steps']}个输入观测精确相同。八月第一次分歧后还会回到相同状态，因此首次分歧不代表后续所有状态都不同。观测不重合只说明已保存样本的精确覆盖，不等于统计分布外。",'',
 '27个重合搜索的全部已查询反馈逐字段匹配教师结果，候选顺序、别名、掩码、阈值一致，教师反馈重执行原FRAME的选择也一致。未重合的三个搜索是三月1751/1754步与八月1908步；不借用其它状态的候选结果补齐。',
 '', 'GRID146主阈值误停的三月call009、GRID150主阈值误停的八月call001，均在实际FRAME轨迹中找到完全相同上下文。这两个反例属于实际会遇到的状态，不能用部署分布偏移把它们排除。这里没有运行模型产生新的停止决策。',
 '', '检查了44646个逐步向量哈希、90份调用向量与四对重复，所有冻结源保持不变；无环境初始化、仿真、训练、推理、采集或新标签，最终8场景未生成。',
 '', '下一步仅检查已保存FRAME反馈的标签可识别性：提前停止后的同站补查若需要当时未查过的候选，标为未知；不把未知标成继续，也不从相似状态借标签。先确认现有资料还能回答多少问题，再决定是否需要新采集。']
 (O/'GRID153_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed=True,exact_search_coverage=27/30,new_fits=0,new_physics=0,new_labels=0));D.archive('complete')
 c=R/'论文协作资料/FRAME_实际路径覆盖_GRID153';c.mkdir(exist_ok=False)
 for n in ['GRID153_report.md','coverage_summary.json','per_run.json','known_error_coverage.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID153既有实际FRAME路径核查：四开发场景30搜索中27（90%）观测/基础动作/公开上下文精确重合教师，全部共同反馈一致。GRID146/150主误停都在实际轨迹出现，不能主要归因于教师路径来源。44646向量/90调用/四对重复通过。五月1390步未完成保留；无采集/拟合/推理/标签/物理/最终8。下一步只读检查已付费反馈能否识别原停止标签，未知保留。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID153实际路径覆盖\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID153后\n\n1. 只读核查全部30个既有实际FRAME搜索的提前停止标签可识别性；保留原标签、全部前缀，缺失反事实反馈标未知，不假装继续。\n2. 不以路径偏移解释掉GRID146/150实际出现的误停，不训练未过筛26项，不扫描组合权重。\n3. 无新仿真/采集/最终8/论文/自动扩大；扩大前通知。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID153实际路径覆盖\r\n\r\n[报告](protocols/GRID153_path_coverage/complete/GRID153_report.md)；实际搜索27/30精确重合，近期误停不能主要归因于路径来源。\r\n\r\n'.encode('utf8')+p.read_bytes())
 count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=6));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID153_REPORT_COMPLETE')
if __name__=='__main__':main()
