import json,shutil
import run as D
def main():
 D.verify_trained();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'] and D.read(O/'diagnosis_receipt.json')['passed'];s=[x for x in D.read(O/'summary.json') if x['partition']=='validation'];cases=D.read(O/'primary_counterexample_comparison.json')
 lines=['# GRID158：只在192判断是否解决了问题','',
 '没有解决。仅在192训练的原配置随机森林，开发62次搜索中61次保持原动作、1次不同，查询减少宏平均5.19%，未通过原门槛。不能为了它更好看继续换检查点或降低质量要求。','',
 '| 方法 | 动作差异/62 | 查询总数 | 日期查询减少宏平均 |','|---|---:|---:|---:|']
 for x in s:lines.append(f"| {x['method']} | {x['action_differences']} | {x['queries']} | {x['date_macro_query_reduction']:.2%} |")
 lines+=['', '原RF140全部实际退出本来就在192；只限制其决策位置，无论保留原阈值还是重新按原规则校准，结果均与原模型一致。新模型与原模型的差别来自重新拟合192样本，不能说改变决策位置本身产生收益。原配置/标签/18项/12-4-4日期均保持，153拟合样本仅10个继续标签；开发集已暴露，仍只是开发回归。',
 '', '新模型继续类AUC：拟合0.868、校准0.464、开发0.741。仅拿拟合表现解释效果会夸大其稳定性。模型、阈值和原模型位置对照均在本次开发评分前封存并推送；未根据开发误停重新定阈值。',
 '', '## 三次主误停揭示的不同情况','',
 '三次主误停都与实际FRAME轨迹的保存状态完全一致：三月128过早在站31停止，后续搜索转到站16后补查得到动作355；八月192时奖励仍在明显改善，却提前停止，后续全局还找到更优动作271；十月192时上一批无改进，下一批仅有约0.000359的一步奖励改进，却把最好动作从站4移到站11，随后同站补查得到444。',
 '', '十月案例说明，继续搜索的作用可能不只是直接提高当前奖励，还可能发现更值得补查的变电站。仅根据当前最好奖励或上一批改进大小判断，可能漏掉这种间接作用。这里只是已保存反馈支持的机制解释，尚未证明加入站级信息就能学好，也不把单次差异解释成完整周成本的原因。',
 '', '因此停止“只换检查点”的尝试。下一步先在原拟合数据里核对这类“全局发现变电站、局部补查获得最终动作”的情况占多少、是否跨日期出现，再盘点停止前确实可见的站级覆盖信息；不拿未来动作结果作在线输入，不立即加网络。',
 '', '唯一一次拟合完成，原模型未重训，274项公开输入复算、1220组独立策略影子核对、2440次完整保存搜索回放通过。没有新物理、数据、最终8、完整控制时间测量、模型采用或扩大。全部较差结果与三个反例保留。']
 (O/'GRID158_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('complete.json',dict(passed_audit=True,development_screen_passed=False,fits=1,adopted=False,ready_to_scale=False));D.archive('complete')
 c=R/'论文协作资料/FRAME_单192对照_GRID158';c.mkdir(exist_ok=False)
 for n in ['GRID158_report.md','decision.json','cutoff.json','stage192_ranking.json','primary_counterexample_comparison.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg='GRID158唯一192固定RF拟合：dev61/62动作一致、查询宏少5.19%，未过原门槛；原RF192位置控制等于原RF4.14%，不再扫描检查点。新误停十月call005：192停滞、256微小改进跨站4→11，随后补查得到444，不能仅以直接奖励判断发现站价值。三主误停均实际路径精确重合。下一步只读统计全局发现站→局部最终动作的反馈路径，不加网络/改标签/扩数据/最终8。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID158单192对照未过筛\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID158后\n\n1. 不再扫描检查点、阈值或网络。零拟合统计原12拟合日期中全局发现站→局部最终动作的反馈路径，按调用和日期而非重复前缀单独计数。\n2. 盘点停止前已知站级覆盖信息；未来结果只能作标签/审计，不能用来造在线特征或把描述当效果证明。\n3. 旧模型仍不采用，不放宽动作/质量要求。无新物理/数据/最终8/论文，扩大需用户决定。\n',encoding='utf8',newline='\n')
 p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID158单192对照\r\n\r\n[报告](protocols/GRID158_single_192/complete/GRID158_report.md)；61/62保动作，未过筛；停止仅换检查点的尝试。\r\n\r\n'.encode('utf8')+p.read_bytes());count=0
 for phase in D.A.iterdir():
  for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=count,collaboration_files_checked=7));d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID158_REPORT_COMPLETE')
if __name__=='__main__':main()
