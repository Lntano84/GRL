"""Write the full Chinese V2 and preserve V1 and user edits."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review';DEST=OUT/'manuscript_v2'
PY='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
NODE='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
SKILL=Path('C:/Users/windows/.codex/plugins/cache/openai-primary-runtime/documents/26.1015.12040/skills/documents')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def reduction(v):return f'减少{v*100:.2f}%' if v>=0 else f'增加{-v*100:.2f}%'
def main():
 if (DEST/'render_complete.json').exists():print('Existing V2 render retained');return
 DEST.mkdir(exist_ok=True);R=read(OUT/'review.json');D=read(OUT/'decision.json');G=R['groups'];C=D['comparisons']
 source=ROOT/'outputs/frame_cn_merged_20261017_v1/FRAME_ICAPS2027_中文合稿_V1.md';text=source.read_text(encoding='utf-8')
 target='候选按代表编号升序排列。';assert target in text
 text=text.replace(target,'默认候选序列沿用作者动作库的教师使用频次降序排列。动作编号对应这一存储次序，运行时筛选与等价合并保留原有相对顺序。作者库包含909个动作，前128个占库内记录频次的44.38%；这一频次先验用于确定访问顺序。',1)
 method='''### 2.3.5 搜索性质与停止依据

一次搜索中，全局扩展仅访问后续候选，同站补查排除已经查询的代表，因此每个代表最多评估一次，总查询数不超过候选数。每次查询后仅在新候选可用且排序键更优时更新当前动作，由此可得：返回动作始终是已查询可用集合中的最佳动作，按既定数值精度计算的最佳奖励不会随集合扩展而下降。

两个分支具有不同的搜索重点。初始前缀尚未达到负载率阈值时，优先寻找能够达到该阈值的动作，再进行必要的同站补查；初始已经达标时，通过后续奖励改善与停滞反馈确定核查深度。前者采用满意解停止，后者检查进一步改善的机会。前缀、批次和停滞数采用经验设置，补充实验逐一检验其变化。

'''
 text=text.replace('## 2.4 参数设置与训练需求',method+'## 2.4 参数设置与训练需求',1)
 needle='该对照沿用129条训练记录和22条校准记录，每个种子训练30000回合，比较5个训练种子，计算费用系数固定为0，采用最终检查点。'
 if needle not in text:
  # Preserve exact V1 typography while matching only the relevant sentence.
  import re
  match=re.search(r'该对照沿用\s*129\s*条训练记录和\s*22\s*条校准记录.*?采用最终检查点。',text)
  assert match,'META settings sentence missing';needle=match.group()
 text=text.replace(needle,'该对照使用25维状态，包含已查询比例、当前最佳奖励、连续停滞情况以及当前电网负载、发电与拓扑信息。三个动作分别为返回当前动作、继续查询64个候选，以及在初始阶段调用完整FRAME搜索。训练使用129条记录、22条校准记录，各种子训练30000回合并使用末端检查点；训练和校准记录按已有划分保留，未使用新增场景选型。原比较中的成本0来自三个预定成本设置的校准选择，校准质量匹配条件未通过。补充比较固定采用非零成本0.0025，报告全部五个种子。',1)
 utility='''
META采用质量与查询费用组成的效用。记当前最佳可用候选奖励为$R$，质量为$q=\min(1,\max(0,R/2))$；尚无可用候选时取零。记累计查询次数为$Q$、候选数为$M$，查询比例为$z=Q/M$。给定成本水平$\lambda$，令$\beta=\log(1+\lambda)$，效用为$U=q-[\exp(\beta z)-1]$。训练反馈取相邻决策效用之差，使用0.99的折扣因子。成本为零时去掉显式查询费用项；非零设置惩罚额外查询。该适配包含FRAME委托动作，用于比较学习控制与固定反馈流程在共同接口下的表现。
'''
 anchor='本文将其记为META。';assert anchor in text;text=text.replace(anchor,anchor+utility,1)
 # Make the original cohort explicit wherever a new cohort could otherwise be confused with it.
 text=text.replace('评价包含两组时序场景。','原有评价包含两组时序场景。',1)
 setup='''
补充评价另固定生成主种子20261017，在相同六个日历日期生成新的随机时序。所有方法、参数和比较规则在读取控制结果前固定；六个场景全部保留。候选顺序检查、参数单因素检查及META成本比较使用已评价场景，独立种子评价使用新增场景，三类结果分别报告。

0.5%的完整周成本容差用于预设实验质量筛查；同时报告实际成本变化、运行时长和完成数。FULL为共同搜索参照。新增场景采用三次轮换计时，逐日期先汇总重复，再计算日期宏平均；描述性区间按日期簇重采样，时间重复与模型种子均不作为新的场景。
'''
 text=text.replace('## 3.2 主要结果',setup+'\n## 3.2 主要结果',1)
 added=['## 3.6 候选顺序与参数检查','', '为检验结果是否依赖当前候选排列，本文保留默认教师频次顺序，并比较三组预先固定的随机顺序及一种按当前线路负载确定的变电站顺序。所有方法保留相同候选编号与平局规则，仅改变候选访问次序。']
 for c in C:
  if c['phase']=='order':
   own=[g for g in G if g['phase']=='order' and g['method']==c['method']];q=c['macro_query_reduction'];qs='无可比比例' if q is None else f'{q*100:.2f}%'
   mode=c['method'].split('__')[1];label='教师频次' if mode=='STATIC_ID' else '线路负载' if mode=='LOCAL_STRESS' else f'随机（种子{mode.split("_")[1]}）'
   added.append(f"{label}顺序下，FRAME在{sum(g['quality_pass'] for g in own)}/6个场景达到质量标准；相对同顺序FULL，匹配活跃时域的查询宏平均减少{qs}。")
 added+=['','参数检查在四个预定旧场景上单独改变前缀长度、批次大小、最低核查量、连续停滞批数和负载率分支阈值，其余参数保持默认。该检查用于报告方法对设置的依赖程度，默认参数保持不变。']
 for key,label in [('p','前缀长度'),('h','批次大小'),('qmin','最低核查量'),('patience','连续停滞批数'),('safe','分支负载率阈值')]:
  parts=[]
  for value in read(OUT/'design.json')['parameters']['single_axis'][key]:
   name=f'FRAME_{key}_{str(value).replace(".","p")}';own=[g for g in G if g['phase']=='parameters' and g['method']==name]
   parts.append(f"{value}时质量达标{sum(g['quality_pass'] for g in own)}/4，查询{sum(g['queries'] for g in own)}次")
  added.append(label+'分别设置为'+'；'.join(parts)+'。')
 added+=['','## 3.7 非零成本的元推理对照','', '成本0.0025的比较复用原有种子0和1，仅新增种子2、3、4各一次训练，网络、训练记录及训练回合数不变。六个已评价场景上同时运行FRAME、零成本五种子与非零成本五种子，所有配置均报告。']
 for cost in [0,1]:
  own=[g for g in G if g['phase']=='meta' and g['method'].startswith(f'META_COST{cost}_')]
  cc=[c for c in C if c['phase']=='meta' and c['baseline'].startswith(f'META_COST{cost}_')]
  qualities=[sum(g['quality_pass'] for g in own if g['method']==f'META_COST{cost}_SEED{i}') for i in range(5)];avg=statistics_mean(c['macro_controller_reduction'] for c in cc)
  added.append(f"成本{'0' if cost==0 else '0.0025'}的五个种子分别在{'、'.join(str(v) for v in qualities)}个场景达到质量标准。FRAME相对这五个种子的完整控制时间宏平均{reduction(avg)}。该批只有一次计时重复，时间差用于描述；质量不匹配的配置同时报告其计算与质量差异。")
 added+=['','## 3.8 独立生成种子评价与等预算闭环比较','', '新增评价使用主种子20261017生成六个随机时序，仍采用同一公开电网及既定月份，方法参数不变。FULL、FRAME、等追加预算全局扩展和固定512均运行三次，所有场景保留。']
 for method in ['FULL','FRAME_GUARD','GLOBAL_MATCH','FIXED_512']:
  own=[g for g in G if g['phase']=='sealed' and g['method']==method];cost=[g['cost_delta_full'] for g in own if g['cost_delta_full'] is not None];suffix='' if not cost else f"，最大可比完整周成本变化为{max(cost)*100:+.3f}%"
  added.append(f"{method}在{sum(g['quality_pass'] for g in own)}/6个场景达到质量标准，{sum(g['complete'] for g in own)}/6个场景完成整周{suffix}。")
 for baseline in ['FULL','GLOBAL_MATCH','FIXED_512']:
  c=next(c for c in C if c['phase']=='sealed' and c['method']=='FRAME_GUARD' and c['baseline']==baseline);q=c['macro_query_reduction'];qs='不适用' if q is None else f'{q*100:.2f}%'
  qualifications='两者在全部场景均达到质量标准。' if c['both_quality_all_dates'] else '部分场景存在质量差异，时间和查询差同时作为描述结果报告。'
  qualifications+='两者运行时域一致。' if c['same_horizon_all_dates'] else '部分场景的终止时刻不同，查询比例仅使用可比时域。'
  added.append(f"相对{baseline}，FRAME的完整控制时间宏平均{reduction(c['macro_controller_reduction'])}，匹配活跃时域的查询减少{qs}。"+qualifications)
 added+=['','等预算全局扩展根据当前最佳动作的变电站归属和未查询候选数确定追加查询量，再将这些查询用于全局序列后续候选。相同状态下的追加预算一致；闭环动作不同后，各自后续访问状态和整周查询总量可以不同。奖励遗憾在FULL访问状态上另行计算，闭环效果使用各策略实际轨迹评价。','', '## 3.9 搜索步骤延迟与计算波动','', '新增三重复实验同时报告全部控制步骤和触发拓扑搜索步骤的P50、P95与P99。完整控制时间按日期报告均值、范围及标准差。逐次查询减少主要压缩预测仿真，连续优化与其他控制操作仍占用计算，因此查询降幅与完整控制时间降幅分别解释。原十三日期对固定512的全部步骤P95变化0.42%，属于另一组比较。','']
 text=text.replace('# 4 结论','\n'.join(added)+'\n# 4 结论',1)
 sealed=next(c for c in C if c['phase']=='sealed' and c['method']=='FRAME_GUARD' and c['baseline']=='FULL');count=D['sealed_frame_quality_dates']
 closing=f"\n独立生成种子的补充评价中，FRAME在{count}/6个场景达到预设质量标准，相对FULL的完整控制时间宏平均{reduction(sealed['macro_controller_reduction'])}。候选顺序、参数和成本目标的完整比较进一步给出了框架适用条件，结构补查的作用结合等预算奖励与闭环结果解释。\n"
 text=text.replace('# 5 相关工作',closing+'\n# 5 相关工作',1)
 text=text.replace('本文评价基于同一公开电网的两组时序场景','本文评价基于同一公开电网的既有时序及两组不同生成种子的时序场景',1)
 text=text.replace('（Hassouna 等，2026；ECML PKDD 2025）','（Hassouna 等，2026）')
 text=text.replace('6个生成场景在首次评价后继续用于对照与机制分析，后续比较属于同场景分析。','首组6个生成场景在首次评价后继续用于对照与机制分析，后续比较属于同场景分析；第二组生成种子在本轮方法固定后确定，检验相同电网与日历条件下的随机时序变化。',1)
 (DEST/'全文待排版.md').write_text(text,encoding='utf-8')
 marker=DEST/'artifact_marker.json'
 if not marker.exists():
  subprocess.run([NODE,str(SKILL/'container_tools/mark_artifact_operation_started.mjs'),'--operation-kind','create','--expected-output-count','1','--output-format','docx'],cwd=ROOT,check=True)
  marker.write_text(json.dumps(dict(started_once=True)),encoding='utf-8')
 subprocess.run([PY,'-X','utf8',str(Path(__file__).with_name('build_docx.py'))],cwd=ROOT,check=True)
 env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'tmp');env['PYTHONIOENCODING']='utf-8';env['PATH']=str(ROOT/'work/grid_cn_draft/render_dependency/extracted/program')+';C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin;'+env['PATH']
 subprocess.run([PY,'-X','utf8',str(SKILL/'render_docx.py'),str(DEST/'FRAME_ICAPS2027_中文合稿_V2.docx'),'--output_dir',str(DEST/'qa'),'--emit_pdf','--dpi','105'],cwd=ROOT,env=env,check=True)
 receipt=read(DEST/'合稿来源与修改说明.json');receipt.update(new_validation_experiments_added=True,old_results_changed=False,additional_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'decision.json',OUT/'review.json',OUT/'design.json']},visual_review_pending=True)
 (DEST/'合稿来源与修改说明.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
 (DEST/'render_complete.json').write_text(json.dumps(dict(rendered=True,visual_check_pending=True,pages=len(list((DEST/'qa').glob('page-*.png')))),ensure_ascii=False,indent=2),encoding='utf-8');print('V2 rendered; all pages require visual inspection before delivery')
def statistics_mean(xs):
 values=list(xs);return sum(values)/len(values)
if __name__=='__main__':main()
