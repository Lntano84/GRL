import json,shutil
import run as D
def main():
 D.verify();O,R,REPO=D.O,D.R,D.REPO;assert D.read(O/'review.json')['passed'];f=D.read(O/'family_summary.json');p=D.read(O/'paired_summary.json');ev=D.read(O/'paired_event_summary.json');stop=sum(r['net_contribution'] for r in ev if r['category'].startswith('LESS_QUERY'));cont=sum(r['net_contribution'] for r in ev if r['category'].startswith('MORE_QUERY'));D.save('direction_decomposition.json',dict(earlier_stop_contribution=stop,extra_continue_contribution=cont,sum=stop+cont,not_new_independent_evidence=True))
 lines=['# GRID166：把已有改善放在一起，哪一部分有效','',
 '一个固定三模型多数票，使平均净效用相对原FACTOR略升0.026%。种子0/1分别提高约0.53%/0.52%，种子2下降约0.92%；种子1四个日期都提高。这个幅度很小，仍可作为局部重复改善的线索，不当作显著或独立确认。没有训练、调权重、扫阈值或选择其它组合。','',
 '三票分别来自原FACTOR、费用路由FACTOR和小COST42。它只是普通多数决策，规则是在前一轮已暴露案例诊断后提出的；因此全部读数都是开发探索。','',
 '| 策略 | 平均净效用 | 平均查询数 | 收益保留率 |','|---|---:|---:|---:|']
 for k in ['FACTOR','MAJORITY','ROUTED','COST42','BASE']:lines.append(f"| {k} | {f[k]['mean_net_utility']:.8f} | {f[k]['mean_queries']:.3f} | {f[k]['gain_capture']:.2%} |")
 lines+=['', '按物理调用均衡的平均效用略低于原FACTOR，不能只挑总体微增宣称全面改善。真正有用的是下一项方向分解。','',
 f'把原FACTOR的继续改为更早停止，净贡献合计{stop:+.8f}；把原本停止改为继续，净贡献合计{cont:+.8f}。前一部分包含86对少查、同收益同动作的记录，涉及32个物理状态组、四日期；后一部分有50对增加查询却没赚回费用。它们几乎相互抵消。',
 '', '因此后续不再增加投票模型或寻找权重。检查一个更符合压缩目的的单向版本：以原FACTOR为基础，只在另两模型都主张停止时提前停；原FACTOR已主张停止时保持停止。查询数在一次固定顺序补查中不会超过原FACTOR，但可能漏掉收益，仍按原净效用核对。',
 '', '这不是新的学习架构。该单向规则由当前方向分解启发，已知一部分保存路径上的预期效果；下一项是复算这一结构性解释，不冒充新数据上的预测成功。所有种子、日期、正反记录都保留，最终测试仍不动。',
 '', '本轮复核13314个已保存状态的投票、全部训练/验证起点和1770个验证配对。新增拟合/模型前向/物理回放均为0。多模型在线推理开销尚未计入旧查询费用，不据此声称真实控制提速。']
 (O/'GRID166_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');D.save('decision.json',dict(small_positive_changes_retained=True,relative_mean_change=p['relative_change_vs_factor'],positive_seeds=2,seed1_all4dates_positive=True,next='Fixed one-sided early-stop-only correction,no reopen of originalFACTORstop.',no_ensemble_grid=True,adopted=False));D.save('complete.json',dict(passed_audit=True,new_fits=0,new_forwards=0,new_physics=0));D.archive('complete')
 c=R/'论文协作资料/FRAME_多数票方向分解_GRID166';c.mkdir(exist_ok=False)
 for n in ['GRID166_report.md','family_summary.json','paired_summary.json','direction_decomposition.json','paired_event_summary.json','review.json','complete.json']:shutil.copy2(O/n,c/n)
 msg=f'GRID166唯一固定多数票零拟合：相对FACTOR平均+0.026%，2/3种子正，种子1四日期均正，调用均衡略负全部披露。方向分解：早停净贡献{stop:+.8f}，新增继续{cont:+.8f}几乎抵消；86对同收益动作少查/32物理组，50对无利多查。下一步一个固定单向“只提前停、不打开原STOP”结构复算，承认来自已暴露分解/非新确认，不继续组合搜索。'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  path=REPO/'docs'/n;old=path.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID166多数票方向分解\n\n'+msg+'\n'
  if n=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
  else:old+=entry
  path.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID166方向分解\n\n1. 仅一个单向提前停规则复算：baseFACTOR继续且两个帮助模型都STOP则提前停；baseSTOP保持。它来自已有早停/继续分解，已知部分预期效果，明确开发解释而非新确认。零拟合、无阈值/权重变体。\n2. 核每次固定补查查询不增加，原NET效用/收益/动作/全日期种子/调用均衡，保留漏收益反例。小幅正向重复值得追，不设全种子同赢门槛。\n3. 无新物理/数据/最终8/论文；多模型推理费用未测，不自动扩大或采用。\n',encoding='utf8',newline='\n')
 path=REPO/'research_archive/power-grid-2026/README.md';path.write_bytes('# 最新：GRID166多数票方向分解\r\n\r\n[报告](protocols/GRID166_fixed_vote/complete/GRID166_report.md)；早停改善被额外继续抵消，检查单向压缩。\r\n\r\n'.encode('utf8')+path.read_bytes());n=0
 for phase in D.A.iterdir():
  for name,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/name)==h;n+=1
 for file in c.iterdir():assert D.sha(file)==D.sha(O/file.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_checked=n,collaboration_files_checked=7));q=D.A/'verification';q.mkdir();shutil.copy2(O/'publication_verification.json',q/'publication_verification.json');(q/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(q/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n');print('GRID166_REPORT_COMPLETE')
if __name__=='__main__':main()
