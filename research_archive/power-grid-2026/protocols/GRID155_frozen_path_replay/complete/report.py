import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];s=D.read(O/'summary.json');rep=D.read(O/'replays.json')
 lines=['# GRID155：冻结模型在已有FRAME状态上的表现','',
 '本轮没有重训或更改阈值。对四个已暴露日期的30个实际FRAME搜索状态，按原18项输入和固定规则重放RF140、LR150、等权组合151及CAP256。所需补查结果未观测时保留未知，不隐藏回退。','',
 '| 方法 | 原动作一致/不同/未知 | 提前停止次数 | 查询总数 | 四日期查询减少宏平均 |','|---|---:|---:|---:|---:|']
 for x in s:lines.append(f"| {x['method']} | {x['action_matches']}/{x['action_differences']}/{x['unknown']} | {x['stops']} | {x['queries']} | {x['date_macro_reduction']:.2%} |")
 lines += ['', 'RF140保住30/30次原动作，但只提前停了两次，总共少查128个候选；宏平均6.60%中，五月只有一次搜索，从256少到192贡献很大。总查询减少只有1.12%，不能把宏平均数字单独解释成普遍效率优势。CAP256总共少查1061个、宏平均7.23%，RF在三月、八月、十月都查询更多。',
 '', 'LR150有一处需要未查询补查结果，因此观测回放中是29次一致、1次未知，不是30次通过。该未知对应此前在精确教师状态已确认的八月call001误停，独立教师审计已知最终动作会变；这里只保留“观测未知”与“教师已知反例”两种证据来源的区别。',
 '', '等权组合只触发一次，减少63个候选。三种学习策略都未通过预定保存状态筛查，不采用、不扩大，不再扫描阈值或组合权重。CAP256在这30个状态没有动作变化，但此前原训练243次搜索存在10次动作变化，不能据此替换原FRAME或宣布它普遍安全。',
 '', '此前已观测事后机会有34.10%的日期宏平均；冻结模型没有充分识别这些机会。路径来源差异并不能解释掉近期误停，新增8项反馈字段的预定筛查也未通过。当前应检验模型是否学到跨日期稳定关系，继续堆特征或扩大网络还缺依据。',
 '', '78前缀隐藏后缀扰动、156次原模型/导出分数复算、70个精确上下文旧分数复核、150个完整保存搜索策略回放通过；原模型/输入/阈值源未改。没有新训练、物理回放或数据扩展。保存状态上的查询数不含闭环轨迹改变后的影响，未测完整控制时间，不推导整周质量或成本。',
 '', '下一步检查原RF140配置的按日期留一稳定性：只用原12个拟合日期，每次拿掉一个日期训练、在被拿掉日期评估；不换标签、特征、种子、超参数或阈值，不评估最终8或再使用开发日期选方案。它是当前表示的学习诊断，不是新增候选策略竞争。训练前单独告知。']
 # Counts are read from verified receipts, so prose cannot silently overstate old-score coverage.
 review=D.read(O/'review.json');lines[-2]=lines[-2].replace('70个精确上下文旧分数',str(review['checks']['old_exact_context_scores'])+'个精确上下文旧分数')
 (O/'GRID155_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed_audit=True,all_learning_screens_passed=False,new_fits=0,new_physics=0,adopted=False,ready_to_scale=False));D.archive('complete')
 c=R/'论文协作资料/FRAME_冻结模型实际状态_GRID155';c.mkdir(exist_ok=False)
 for n in ['GRID155_report.md','summary.json','decision.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID155原冻结模型实际FRAME保存状态回放：RF140动作30/30、仅2次停/128查询节省、宏6.60%且总1.12%；CAP256动作30/30、1061节省、宏7.23%，不推导普适安全。LR29一致1观测未知（精确教师已知该误停）；等权30/30宏0.42%。无重训/阈值扫描，三学习均未过原筛查，无物理/最终8/部署/扩大。下一步拟查固定RF原配置按原12拟合日期留一稳定性，训练前通知，不选新模型。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID155冻结模型实际状态回放\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID155后\n\n1. 训练前告知后，固定RF140原配置，用原12拟合日期逐一整日留出诊断；只看拟合日期交叉验证，不修改标签/18输入/权重规则/种子/参数/原阈值。\n2. 不新增部署候选或根据留一结果择模型/阈值；报告继续类AP、AUC、完整搜索动作和查询。\n3. 无新物理/采集/最终8/论文/自动扩大；扩大前通知并等用户决定。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID155冻结模型实际状态回放\r\n\r\n[报告](protocols/GRID155_frozen_path_replay/complete/GRID155_report.md)；原模型未充分识别机会，三学习方案未过筛。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=5));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID155_REPORT_COMPLETE')
if __name__=='__main__':main()
