from pathlib import Path
import re,json,hashlib,shutil,collections,html,subprocess
R=Path('D:/Codex/GRL研究');P=R/'work/GRL-results-integration';W=R/'work/weekly_review_20261030';O=R/'outputs/research_review_20261030';A=P/'research_archive/power-grid-2026/reviews/2026-10-30_weekly';D=R/'论文协作资料/本周实验总结_20261026-1030'
assert not A.exists();assert not O.exists() or not any(O.iterdir());assert not D.exists() or not any(D.iterdir());O.mkdir(exist_ok=True);D.mkdir(exist_ok=True);log=(P/'docs/EXPERIMENT_LOG.md').read_text(encoding='utf8');notes=collections.defaultdict(list)
for block in re.split(r'(?m)^## ',log)[1:]:
 title,_,body=block.partition('\n');date=re.search(r'2026-\d\d-\d\d',title);ids={int(n) for n in re.findall(r'GRID(\d+)',title,re.I)}
 for m in re.finditer(r'GRID(\d+)\s*(?:至|–|—|-)\s*(?:GRID)?(\d+)',title,re.I):ids.update(range(int(m[1]),int(m[2])+1))
 if date:
  for i in ids:
   if i<=284:notes[i].append(dict(date=date[0],title=title,body=body.strip()))
curated={}
for line in (W/'week_purposes.txt').read_text(encoding='utf-8-sig').splitlines():
 if line.strip():i,purpose,op,result=line.split('|',3);curated[int(i)]=dict(purpose=purpose,operation=op,override=result)
assert set(curated)==set(range(184,285))
report_candidates=collections.defaultdict(list);folders=collections.defaultdict(list)
for p in (R/'outputs').iterdir():
 m=re.match(r'grid(\d+)(?:_|$)',p.name,re.I)
 if m and p.is_dir():folders[int(m[1])].append(p)
for p in (R/'outputs').rglob('*.md'):
 m=re.search(r'(?i)grid(\d+)',str(p.relative_to(R/'outputs')))
 if m and ('report' in p.name.lower() or 'review_zh' in p.name.lower()):report_candidates[int(m[1])].append(p)
archive_names=collections.defaultdict(list)
for p in (P/'research_archive/power-grid-2026').rglob('*.md'):archive_names[p.name.lower()].append(p)
rows=[];mapped=0;missing_report_archive=[]
def firstpara(text):
 parts=[s.strip() for s in re.split(r'\n\s*\n',text) if s.strip() and not s.strip().startswith('#')]
 return re.sub(r'\s+',' ',parts[0]) if parts else ''
def clean(s):return s.replace('|','／').replace('\n',' ').strip()
def link_report(row):
 if row.get('git_report'):return f"[原报告](https://github.com/Lntano84/GRL/blob/main/{row['git_report']})"
 return '[日志](https://github.com/Lntano84/GRL/blob/main/docs/EXPERIMENT_LOG.md)'
for i in range(285):
 pp=sorted(report_candidates[i],key=lambda p:('corrected' in p.name.lower(),bool(re.fullmatch(fr'GRID0*{i}_report\.md',p.name,re.I)),not any(s in p.name.lower() for s in ['initial','original']),str(p)),reverse=True);report=pp[0] if pp else None;nn=notes[i];weeknotes=[n for n in nn if '2026-10-26'<=n['date']<='2026-10-30'];fallback=None
 if not report and i==43:fallback=R/'outputs/grid43/RUN_STATE.md'
 title=''
 if report:
  title=next((x.lstrip('# ').strip() for x in report.read_text(encoding='utf-8-sig').splitlines() if x.startswith('#')),'');title=re.sub(r'^GRID\d+\s*[:：—-]?\s*','',title)
 elif nn:title=re.sub(r'^.*?—\s*','',nn[-1]['title']);title=re.sub(r'^GRID\d+\s*','',title)
 elif folders[i]:title='历史阶段：'+folders[i][0].name
 else:title='尚未定位到独立阶段材料'
 purpose=curated[i]['purpose'] if i in curated else title
 result=firstpara(nn[-1]['body']) if nn else firstpara(report.read_text(encoding='utf-8-sig')) if report else fallback.read_text(encoding='utf8').strip() if fallback else '未定位到独立报告/日志摘要，不据此推断完成或效果。'
 if i in curated and curated[i]['override']:result=curated[i]['override']
 row=dict(grid=i,purpose=purpose,operation=curated[i]['operation'] if i in curated else '历史记录，详见来源',result=result,first_recorded_date=min((n['date'] for n in nn),default=None),week_dates=sorted({n['date'] for n in weeknotes}),in_week=i in curated,source_report=str(report) if report else None,report_variants=[str(p) for p in pp],source_folders=[str(p) for p in folders[i]],log_entries=nn,git_report=None,report_sha256=None,standalone_report=bool(report))
 if report:
  s=hashlib.sha256(report.read_bytes()).hexdigest();matches=[p for p in archive_names[report.name.lower()] if hashlib.sha256(p.read_bytes()).hexdigest()==s];row['report_sha256']=s
  if matches:row['git_report']=matches[-1].relative_to(P).as_posix();mapped+=1
  else:missing_report_archive.append(dict(grid=i,path=str(report),sha256=s))
 rows.append(row)
assert all(v['grid']==11 for v in missing_report_archive),missing_report_archive
for v in missing_report_archive:
 row=rows[v['grid']];row['git_report']=(A/'historical_report_supplements'/f"GRID{v['grid']:03d}"/Path(v['path']).name).relative_to(P).as_posix();mapped+=1
write=lambda n,t:(O/n).write_text(t,encoding='utf8')
write('experiment_inventory.json',json.dumps(dict(scope='GRID000-284编号索引；编号阶段非独立训练次数',date='2026-10-30',weekly_scope='2026-10-26至2026-10-30，GRID184-284共101编号阶段，184含早先运行的本周审计/发布',rows=rows),ensure_ascii=False,indent=2)+'\n')
week=rows[184:285];table=['# 本周逐项实验清单：101个编号阶段','','统计窗口：2026-10-26（周一）至2026-10-30（周五、当前截止）。GRID184为早先回放在本周完成审计发布；合并日志条目按明确编号展开，GRID225/226的零评分失败与227有效修订分别列明。编号数包含设计资格、诊断、公共导出、标量校准，**不等于101次独立训练或101个独立实验样本**。','','|阶段|本周记录日|目的|实际操作|结果与限制|来源|','|---|---|---|---|---|---|']
for r in week:
 date='/'.join(x[5:] for x in r['week_dates']) or '10-28（组合阶段记录）';table.append(f"|GRID{r['grid']}|{date}|{clean(r['purpose'])}|{clean(r['operation'])}|{clean(r['result'])}|{link_report(r)}|")
write('本周逐项实验清单_101阶段.md','\n'.join(table)+'\n')
alllines=['# 全部GRID实验目的与结果索引：GRID000—284','','这是285个编号位置的材料索引，不把资格、故障诊断和重复计时算作独立训练。按报告与研究日志整理；各阶段费用、控制器和数据集有历史差别，不能横向拼成一种算法的成绩。未定位到独立报告的行明确标注，使用已有日志/目录，不补造结果。','','|编号|目的/问题|结果摘要|证据状态|来源|','|---|---|---|---|---|']
for r in rows:alllines.append(f"|GRID{r['grid']:03d}|{clean(r['purpose'])}|{clean(r['result'])}|{'主报告已逐字节归档' if r['git_report'] else '无独立主报告，见日志/原始目录'}|{link_report(r)}|")
write('全部实验索引_GRID000-284.md','\n'.join(alllines)+'\n')
raw=['# 本周原始研究日志摘录','','保留源条目的原文，用于追查摘要；组合阶段在逐项清单中拆开，以下不重复复制同一条。来源：integration/docs/EXPERIMENT_LOG.md。']
seen=set()
for r in week:
 for n in r['log_entries']:
  if '2026-10-26'<=n['date']<='2026-10-30' and n['title'] not in seen:seen.add(n['title']);raw+=['','## '+n['title'],'',n['body']]
write('本周原始记录摘录.md','\n'.join(raw)+'\n')
# Read published endpoints without fitting or new policy evaluation.
endpoints=[]
for i,folder,arm,countkey in [(208,'grid208_public_amount','GATE','append_positive_NET'),(230,'grid230_training_budget','BUDGET','retained_profitable'),(250,'grid250_context_residual','CONTEXT','profitable_retained'),(258,'grid258_reference_correction','REFSOURCE','profitable_retained'),(261,'grid261_bounded_basis','BOUNDED','profitable_retained'),(265,'grid265_guarded_bias','SHIFT','profitable_retained'),(268,'grid268_conditional_forest','NET_FOREST','profitable_retained'),(271,'grid271_union_guard','UNION_FOREST','profitable_retained'),(274,'grid274_stop_context','CONTEXT_FOREST','profitable_retained'),(277,'grid277_selector_free','SELECTOR_FREE','profitable_retained'),(280,'grid280_invocation_conditioned','ENTRY_CONDITIONED','profitable_retained'),(283,'grid283_fixed_partition_leaf','FIXED_LEAF','profitable_retained')]:
 z=R/'outputs'/folder;f=json.loads((z/'family_summary.json').read_text())[arm];c=json.loads((z/'gate_counts.json').read_text());endpoints.append(dict(grid=i,arm=arm,mean_NET=f['mean_net_utility'],mean_queries=f['mean_queries'],profitable_retained=c[countkey],action_match=f['action_match_rate'],gain_capture=f['gain_capture']))
write('published_endpoint_table.json',json.dumps(endpoints,ensure_ascii=False,indent=2)+'\n')
summary='''# 本周GRL学习模块实验总结

**窗口：2026-10-26—2026-10-30；状态：GRID284分析完成，科研推进停止，自动跟进保持暂停。**

## 先看判断

本周确实找到了一些能保留的小改善。最清楚的是 **GRID261**：在当前已暴露的教师路径开发数据上，保留共同GATE的55/55回本机会，全部7134条种子—起点路径的局部增益与最终候选ID相同，三个种子的查询都减少，日期宏平均少0.1401%，原始计数省122次。这是“小而在三个种子上同向”的内部学习改善，值得留作质量锚。

之后为进一步省查询而增加分数偏置或附加森林，节省幅度更大，却普遍丢失回本机会。不能因为均值NET微升、某个种子正向、或查询更少就替换261；也没有把这些线索全部否定，局部有利区域和所有反例都保留了。

本周研究的重心是**学习补查模块**：冻结基础NET，研究原搜索停止后是否补一批、哪些补查可以取消、如何保留已有收益。大量阶段属于监督学习、浅森林、线性修正、数据来源核查和经验校准；并不是连续重新训练新的强化学习控制器。

## 这一周围绕什么问题推进

|阶段组|要回答的问题|做了什么|形成的认识|
|---|---|---|---|
|184—188|原学习模块哪里真省查询，直接换目标是否更好|闭环审计、同动作对齐、动作不变目标对照|原FACTOR有同动作省查询点；纯动作变化目标虽更省，却丢高价值机会，收益幅度仍要保留|
|189—204|能否只在原模型停止后补救机会|单批追加、覆盖/数值对照、概率与条件收益两头|追加确能跨种子找回局部收益，但常买到零收益批次；两头浅树改善了内部质量/查询折中|
|205—214|模型是否缺少已有公开信息和训练支持|原41摘要直连、深度对照、内层样本生成|208形成重要父方法；单纯加深收益树无稳定优势，分歧域正例稀少|
|215—245|如何取消无效补查又不误停|取消森林、原判断偏置、完整L2、保护步长、祖先和先验一致性核查|独立取消分类器容易覆盖掉原有好判断；经验正例保护有用，但不是外层保证，内部cal4也不是全链独立|
|246—263|表示、决策上下文和STOP标签来源是否匹配|去重复列、上下文/qgap、18参考NN、同来源重建、tanh基函数|来源修订增加回本支持；有界输入261保住所有既有回本且小幅省查询，是本周最清楚的质量保留结果|
|264—284|能否在261基础上继续扩大节省|偏置、浅森林、并集保护、上下文、入口条件训练、固定分区叶频|更大节省多伴随质量损失；固定分区比条件重分区找回两条相关回本，但几乎回到277，尚无新的稳定净改善|

完整逐项记录在[本周101阶段清单](本周逐项实验清单_101阶段.md)，历史全部编号在[GRID000—284索引](全部实验索引_GRID000-284.md)。**101是编号阶段数，不是独立训练次数**；资格、只读诊断、数据导出、失败修订均分别标明。

## 关键版本对照

下面均为原16个已暴露日期、全部3种子的历史教师路径开发比较。查询是候选查询，NET是固定费用下的局部代理效用；“保留/55”以208共同GATE的55条局部回本记录为参照。前缀和种子配对相关，不是55个独立场景。输入、样本来源或保护集合有变化的版本不是单因素消融；跨版本表用于定位，不替代各报告的匹配对照。

|版本|用途|回本保留/55|平均候选查询|局部代理NET|
|---|---|---:|---:|---:|
'''
labels={208:'共同GATE父方法',230:'较早成本/质量折中候选',250:'低查询研究端点',258:'新STOP来源，高回本端点',261:'当前质量保留锚',265:'向下偏置取舍',268:'附加NET森林取舍',271:'固定并集保护',274:'上下文副本',277:'去selector位森林',280:'条件入口训练森林',283:'固定旧分区的叶频模型'}
for e in endpoints:summary+=f"|GRID{e['grid']}|{labels[e['grid']]}|{e['profitable_retained']}|{e['mean_queries']:.6f}|{e['mean_NET']:.9f}|\n"
summary+='''
### 值得保留的改善与限制

- **190：小补救能重复出现。** 相对基础NET，动作一致率88.984%→90.335%，三个种子都提高；但查询12.884→15.007，质量增加不是等预算优势。
- **201—208：两头学习有实际内部增量。** 浅树条件收益、事件概率和公开摘要直连逐步改善内部折中。208相对206查询13.5917→13.4405、NET .095993729→.096092524；真实查询下降并不意味着每个种子的NET都提高，原报告保留了种子0反向情况。
- **258：纠正STOP来源找到更多机会。** 回本29→48/55，恢复21、另失2；查询三个种子都增加约1.60%，NET变化一好两差，不能称稳定胜出。
- **261：当前最干净的小改善。** 保留全部原增益和候选，三个种子查询均少；77条零增益取消涉及42调用、14日期，合并71状态，仅6跨两个种子、无跨三个种子重复，不能把77当独立样本或说每个状态都重复改善。
- **268及后续森林：省得更多也失得更多。** 268对261查询宏少1.4419%，却回本55→44/55，种子2大额损失主导；种子1的保增益小省查询仍保留，但有四日期预定KEEP，不能拼成功种子部署。
- **283—284：把树分区和叶频效应分开了。** 固定277分区后，对280恢复两条Feb16同调用相关回本、无新失，但多106次查询，NET均值微降；对277仅三条非回本决定变化、原始净省1，全部增益候选相同。284查清删除331个负例没有改变叶正例权重，主要是分母减小；分数与保护截点同时提高，因而决策大多不变。

## 真正的闭环证据要另外看

本周完成GRID184的既有64条真实回放审计和发布。所有方法在3日期完整，在5月日期都止于1390步；FULL参照自身未完，不能判整周质量或用早停冒充提速。可判日期中均2/3通过，3月是共同成本失败。

原FACTOR相对未学习FRAME在相同时域控制时间宏少3.55%、候选查询少9.61%，有23个同观测同基础动作、最终动作相同而查询更少的配对调用；共识组合对同种子FACTOR控制时间反而多1.62%。这些证据支持保留原学习模块的有效点，但**本周后续离线版本尚未重新证明整周成本或实际控制提速**。

## 工程、数值和来源问题有没有混入效果

1. GRID225/226数值资格未过，外层评分均为0，失败完整保留；227只修原不合格单池，其余合格模型复用。不能把225/226当有效性能结果。
2. GRID235原候选/实际拟合计数已更正。原outer4隔离与内部cal4祖先是两回事，不能把条件层留组诊断称全链OOF或盲测。
3. GRID255共18条固定参考NN轨迹，12完整不重训，6条续训补8285更新。原600秒不足，经用户另授权120秒恢复窗，实际累计604.703秒/上限678.156秒；不冒充原窗内完成。
4. 18参考NN+258的22模型/标量构成原40操作探针；后续新假设另列，至283这条分支累计190。**190不是全周所有训练的总数**，也不隐藏6条续训实例和时间。
5. 500→543保护集合、16相关副本、删除331条样本、去selector位及有界表示分别改变了约束暴露、经验风险或函数类，均已披露，不能称同样本同容量的纯消融。

## 材料与上传核对

全部新材料保存在D:/Codex/GRL研究，本总结、逐项清单、历史索引、原始日志摘录、端点表和SHA256清单会一并归档到现有Lntano84/GRL仓库。

本次只做文件级发布核对：本周468份平面SHA清单覆盖9,818个唯一文件，约1.90 GB，本地哈希零不匹配、缺失文件0；发现并补传333份被忽略的既有日志（220,998字节，提交2719645e）。这是补齐材料，不是科研重跑。已定位主报告逐份核对：原有同字节副本不重复复制；新定位的GRID11旧报告补充归档后，完整索引提供原报告链接。

此处确认Git记录、清单、本地材料及远端提交一致；没有在这次整理中重新下载所有远端LFS大对象。原始大轨迹、数据集和运行环境继续按项目规则保留在D盘，不把依赖环境或整个工作目录无差别上传。

## 现在停止，留给审阅后的决定

GRID284结束后，不启动GRID285或新训练，不扩日期/电网，不新做FRAME比较，不读最终8，不修改论文；自动跟进保持PAUSED。待你审阅后再决定是否以261为质量锚继续小幅省查询、是否先补独立验证、或先收敛代码和实验叙述。本报告不替你启动其中任何路线。
'''
write('本周实验总结_20261026-1030.md',summary)
audit=json.loads((W/'publication_audit.json').read_text(encoding='utf8'));audit.update(all_numbered_positions=285,week_numbered_stages=101,canonical_reports_located=sum(r['standalone_report'] for r in rows),all_located_reports_byte_identical_archive=mapped,legacy_reports_supplemented=missing_report_archive,missing_report_archive=[],full_index_without_standalone_report=[r['grid'] for r in rows if not r['standalone_report']]);write('publication_audit.json',json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
readme='''# 本周实验材料阅读入口

研究已经停在GRID284；自动跟进保持暂停，没有安排新训练。

1. [先看本周总结](本周实验总结_20261026-1030.md)：研究问题、关键版本、有效改善与反例、工程边界和当前判断。
2. [本周逐项101阶段](本周逐项实验清单_101阶段.md)：每个阶段的目的、实际操作和结果，失败与资格单列。
3. [全部GRID000—284索引](全部实验索引_GRID000-284.md)：全部编号位置及已定位证据；无独立报告的明确标记。
4. [本周原始日志摘录](本周原始记录摘录.md)：回查原始表述；汇总不覆盖历史记录。
5. [浏览器总览](本周实验总览.html)：可搜索的完整清单。
6. publication_audit.json、published_endpoint_table.json、experiment_inventory.json和文件清单_SHA256.json：来源、数值和材料核对收据。

大部分近期结果来自已暴露教师路径开发数据。局部收益、查询费用代理和实际整周成本/时延分开判读，最终8场景未打开。
'''
write('README.md',readme)
# Standalone lightweight reader; no external service or libraries.
weekrows=''.join('<tr><td>GRID'+str(r['grid'])+'</td><td>'+html.escape(r['purpose'])+'</td><td>'+html.escape(r['operation'])+'</td><td>'+html.escape(r['result'])+'</td><td>'+(f'<a href="https://github.com/Lntano84/GRL/blob/main/{r["git_report"]}">报告</a>' if r['git_report'] else '日志/资格记录')+'</td></tr>' for r in week)
allrows=''.join('<tr><td>GRID'+str(r['grid']).zfill(3)+'</td><td>'+html.escape(r['purpose'])+'</td><td>'+html.escape(r['result'])+'</td><td>'+(f'<a href="https://github.com/Lntano84/GRL/blob/main/{r["git_report"]}">报告</a>' if r['git_report'] else '无独立主报告，见JSON索引/日志')+'</td></tr>' for r in rows)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GRL本周实验总览</title><style>body{font:16px/1.75 system-ui,sans-serif;background:#f5f7fa;color:#172b3a;margin:0}main{max-width:1450px;margin:auto;padding:30px}h1{font-size:30px}a{color:#165aa8}nav,section{background:white;padding:24px;border-radius:12px;margin:18px 0}.status{border-left:5px solid #28785e;padding:16px;background:#ecf8f2}input{font:inherit;padding:12px;width:min(92%,650px);border:1px solid #8fa5b5;border-radius:7px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{vertical-align:top;padding:12px;border-bottom:1px solid #dbe4ec}th{background:#edf3f8;text-align:left}td:first-child{white-space:nowrap}.scroll{overflow:auto}summary{cursor:pointer;font-weight:700;font-size:20px}.muted{color:#54697b}@media(max-width:700px){main{padding:12px}section{padding:12px}table{min-width:850px}}</style><main><h1>GRL学习模块：本周实验总览</h1><p class="muted">2026-10-26—10-30｜101个编号阶段，包含资格、训练、校准与只读诊断</p><p class="status"><b>已停止科研推进，自动跟进暂停。</b>质量保留锚GRID261：55/55回本、全部增益候选相同，三个种子查询均少，宏平均节省0.1401%。这仍是暴露开发数据上的局部结果，不是整周或实际时延结论。</p><nav><a href="本周实验总结_20261026-1030.md">本周总结</a> · <a href="本周逐项实验清单_101阶段.md">逐项Markdown</a> · <a href="全部实验索引_GRID000-284.md">完整历史索引</a> · <a href="publication_audit.json">上传核对收据</a><p>已补传333份原有日志；9,818份本周归档文件本地SHA核对通过。没有重训、重评分或扩展实验。</p></nav><section><h2>本周101阶段</h2><input id="q" placeholder="搜索编号、目标、模型或结果，例如：261、数值、STOP"><div class="scroll"><table id="week"><thead><tr><th>阶段</th><th>目的</th><th>实际操作</th><th>结果/限制</th><th>证据</th></tr></thead><tbody>'''+weekrows+'''</tbody></table></div></section><section><details><summary>全部285个编号位置（GRID000—284）</summary><p>编号不等于独立训练次数；无独立报告的行明确标记，原始文件路径保留在JSON索引。</p><div class="scroll"><table id="all"><thead><tr><th>编号</th><th>目的</th><th>结果摘要</th><th>证据</th></tr></thead><tbody>'''+allrows+'''</tbody></table></div></details></section></main><script>document.getElementById('q').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))})</script></html>'''
write('本周实验总览.html',page)
# Stage-specific delivery receipts and resumable handoff remain outside old immutable archives.
hand='''# GRID284后：用户要求停止并审阅本周材料

2026-10-30用户明确要求“这轮分析完先停下，整理所有实验及目的和结果，该上传都上传”。GRID284已完成并发布1b3552aa，0新拟合/策略/路由，科研推进停止，grl保持PAUSED；不启动285或新训练，不自动恢复。用户审阅后须明确指示再继续。

最新总结和全实验索引位于outputs/research_review_20261030，阅读README.md和本周实验总结_20261026-1030.md。GRID184-284共101编号阶段，285个历史编号位置索引；编号≠独立训练或样本。报告/日志/模型收据的原文件不覆盖。文件级审计468份清单/9818唯一文件哈希和Git覆盖通过，补传333既有日志220998字节，2719645e；不是科研重跑。所有已定位主报告有同字节归档副本。没有本次全量重新下载LFS大对象，原始大数据和依赖继续D盘留存。

本周保留261质量锚：GATE55/55、7134路径增益/候选同、三seed query少宏.1401%raw122；后续森林查询下降多有质量损失。283固定277分区回本45/55，对280恢复两相关前缀但raw+106/NET微降，对277仅seed2三非正决定raw省1。284查6715叶正例权重差最大1.14e-13，331负例删减只改分母，1136叶频升且11截点不降。全部反例/KEEP四日期/祖先开发局限保留。

实际A=GRID284_leaf_support，完整归档不重复。周报归档reviews/2026-10-30_weekly。用户决定下一步前只允许交付修正/文件核对，不新增科研。最终8/物理电网论文FRAME不动，无安装；原费用标签控制器排序别名锁站128/64停滞2/.9保持。Githelper串行等exit_code，白名单，D协作/hash/远端核验；不恢复其他旧自动化，不归档聊天。
'''
(R/'outputs/grid284_leaf_support/CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8');shutil.copy2(R/'outputs/grid284_leaf_support/CONTINUOUS_HANDOFF.md',O/'CONTINUOUS_HANDOFF.md')
for n in ['GRID284_report.md','complete.json','independent_audit.json']:shutil.copy2(R/'outputs/grid284_leaf_support'/n,O/('GRID284_'+n if n!='GRID284_report.md' else n))
write('文件清单_SHA256.json',json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()},ensure_ascii=False,indent=2)+'\n')
A.mkdir(parents=True)
for p in O.iterdir():
 if p.is_file():shutil.copy2(p,A/p.name);shutil.copy2(p,D/p.name)
(A/'archive_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in A.iterdir()},indent=2)+'\n',encoding='utf8')
# Preserve scripts and input audit evidence as part of the user-requested review material.
H=A/'historical_report_supplements'
for v in missing_report_archive:
 d=H/f"GRID{v['grid']:03d}";d.mkdir(parents=True);shutil.copy2(v['path'],d/Path(v['path']).name);(d/'archive_manifest.json').write_text(json.dumps({Path(v['path']).name:v['sha256']},indent=2)+'\n',encoding='utf8')
S=A/'review_sources';S.mkdir()
for p in W.iterdir():
 if p.is_file():shutil.copy2(p,S/p.name)
(S/'archive_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.iterdir()},indent=2)+'\n',encoding='utf8')
entry=f'2026-10-30：按用户要求完成GRID284后停止；整理GRID184-284共101编号阶段、GRID000-284共285位置索引，{mapped}份已定位主报告逐字节归档匹配。本周468清单/9818唯一文件本地SHA全部通过，补传333既有日志220998字节（2719645e），非科研重跑。周报、逐项目的结果、全索引、原始记录摘录、端点数据及校验收据已归档reviews/2026-10-30_weekly与D协作。grl保持PAUSED，无285/新训练；等待用户审阅决定。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=P/'docs'/n;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-30 — 本周全实验整理与发布\n\n'+entry+'\n'
 if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+e+b
 else:t+=e
 p.write_text(t,encoding='utf8')
(P/'docs/NEXT_STEPS.md').write_text('# Next steps — 停止研究，等待用户审阅周报\n\n'+entry+'\n\n1. 阅读research_archive/power-grid-2026/reviews/2026-10-30_weekly/README.md及本周总结。\n2. 如用户仅要求解释或修正材料，处理文档和核对，不新增研究。\n3. 明确获得继续指示后才决定新方案；不自动启动285、训练、最终8、新物理电网论文FRAME。\n',encoding='utf8')
p=P/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：本周实验总结已整理，研究暂停\n\n[阅读本周材料](reviews/2026-10-30_weekly/README.md)；'+entry+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(prepared=True,weekly_stages=len(week),all_numbered_positions=len(rows),canonical_reports=sum(r['standalone_report'] for r in rows),archived_report_matches=mapped,missing_standalone_reports=[r['grid'] for r in rows if not r['standalone_report']],output=str(O),archive=str(A),collaboration=str(D)),ensure_ascii=False))
