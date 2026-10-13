import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(import.meta.dirname,'../..');
const out=path.join(root,'outputs/grid_data_review_20261013');
const data=JSON.parse(await fs.readFile(path.join(out,'data.json'),'utf8'));
const wb=Workbook.create();
const names=['正式比较','正式逐次','频次顺序消融','新先验逐次','计时与路径','口径与来源'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const pct='0.00%;-0.00%;0.00%';
function base(s,title,note,last='N',rows=150){
 s.showGridLines=false;
 s.getRange(`A1:${last}${rows}`).format.font={name:'Arial',size:10,color:'#202938'};
 s.getRange(`A1:${last}${rows}`).format.verticalAlignment='center';
 s.getRange(`A1:${last}${rows}`).format.rowHeight=23;
 s.getRange(`A1:${last}${rows}`).format.columnWidth=17;
 s.getRange('A2').values=[[title]];s.getRange('A2').format.font={bold:true,size:15};
 s.getRange('A3').values=[[note]];s.getRange('A3').format.font={italic:true,color:'#556173'};
}
function table(s,row,headers,values,widths){
 s.getRangeByIndexes(row-1,0,1,headers.length).values=[headers];
 s.getRangeByIndexes(row,0,values.length,headers.length).values=values;
 const h=s.getRangeByIndexes(row-1,0,1,headers.length);
 h.format={fill:'#24374E',font:{bold:true,color:'#FFFFFF',name:'Arial',size:10},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:44};
 for(let i=0;i<headers.length;i++)s.getRangeByIndexes(row-1,i,values.length+1,1).format.columnWidth=widths[i]??17;
 for(let i=0;i<values.length;i++)if(i%2===1)s.getRangeByIndexes(row+i,0,1,headers.length).format.fill='#F3F6FA';
}
const formal=sheets['正式逐次'];
base(formal,'八日期正式比较：128个计时槽位','逻辑重复1中的受影响日期统一使用实际重复2；原始重复未删除。','N',135);
const formalHeaders=['逻辑重复','实际重复','日期','方法ID','完成(1/0)','步数','累计模拟成本','控制时间(s)','N1查询数','成本变化/FULL','控制减少/FULL','无提前失败(1/0)','质量通过(1/0)','N1搜索时间(s)'];
table(formal,5,formalHeaders,data.formal.map(r=>[r.logical_repeat,r.physical_repeat,r.week,r.method,r.complete,r.steps,r.cost,r.control_s,r.n1_queries,null,null,r.no_earlier_failure,r.quality_pass,r.n1_source_s]),[12,12,22,27,12,12,21,18,16,18,18,18,18,19]);
const fidx=new Map(data.formal.map((r,i)=>[`${r.logical_repeat}|${r.week}|${r.method}`,i+6]));
for(let i=0;i<data.formal.length;i++){
 const r=data.formal[i], row=i+6, ref=fidx.get(`${r.logical_repeat}|${r.week}|FULL`);
 formal.getRange(`J${row}:K${row}`).formulas=[[`=(G${row}-G${ref})/ABS(G${ref})`,`=1-H${row}/H${ref}`]];
}
formal.getRange('G6:H133').setNumberFormat('#,##0.00');formal.getRange('I6:I133').setNumberFormat('#,##0');
formal.getRange('J6:K133').setNumberFormat(pct);formal.getRange('N6:N133').setNumberFormat('0.00');
formal.freezePanes.freezeRows(5);formal.freezePanes.freezeColumns(4);

const summary=sheets['正式比较'];
base(summary,'原GNN、质量补强版与主对照','同一八日期、每方法两次计时；宏平均按日期等权，查询比例为汇总口径。','J',25);
const aliases={'FULL':'FULL完整搜索','SOFTGNN0':'SoftGNN seed 0','SOFTGNN1':'SoftGNN seed 1','LOCAL_PPO':'作者PPO辅助搜索','COMPRESSED_GNN0':'质量补强GNN seed 0','COMPRESSED_GNN1':'质量补强GNN seed 1','LOCAL_GNN0':'原GNN seed 0','LOCAL_GNN1':'原GNN seed 1'};
const mainHeaders=['方法','完成/16','质量通过/16','控制减少/FULL\n日期宏平均','N1查询减少/FULL\n汇总比例','平均每周控制(s)','平均成本变化/FULL','最坏成本变化/FULL','决策p95(ms)\n日期平均','方法ID'];
table(summary,5,mainHeaders,data.formal_summary.map(s=>[aliases[s.method],null,null,null,null,null,null,null,s.mean_family_p95_decision_s*1000,s.method]),[30,14,17,21,24,22,22,22,24,27]);
const fullRows=data.formal.map((r,i)=>r.method==='FULL'?i+6:null).filter(x=>x!==null);
const fullQ=fullRows.map(r=>`'正式逐次'!I${r}`).join(',');
for(let i=0;i<data.formal_summary.length;i++){
 const row=i+6, refs=data.formal.map((r,j)=>r.method===data.formal_summary[i].method?j+6:null).filter(x=>x!==null);
 const list=c=>refs.map(r=>`'正式逐次'!${c}${r}`).join(',');
 summary.getRange(`B${row}:H${row}`).formulas=[[`=SUM(${list('E')})`,`=SUM(${list('M')})`,`=AVERAGE(${list('K')})`,`=1-SUM(${list('I')})/SUM(${fullQ})`,`=AVERAGE(${list('H')})`,`=AVERAGE(${list('J')})`,`=MAX(${list('J')})`]];
}
summary.getRange('D6:E13').setNumberFormat(pct);summary.getRange('G6:H13').setNumberFormat(pct);summary.getRange('F6:F13').setNumberFormat('0.00');summary.getRange('I6:I13').setNumberFormat('0.00');
summary.getRange('H6:H13').setNumberFormat('0.00000%;-0.00000%;0.00000%');
summary.getRange('A16').values=[['质量：完成整周、无提前失败、成本不高于FULL的0.5%。该门槛是工程筛查尺度。']];
summary.getRange('A18').values=[['原GNN更快但成本尾部较差。质量补强版守住尺度，PPO和SoftGNN仍略快。']];
summary.getRange('A20').values=[['p95表示绝大多数单步决策的延迟，不等于整周控制总时间。']];

const st=sheets['频次顺序消融'];
base(st,'强频次顺序消融：八个已暴露日期','不同批次不比较控制速度；这里只对照质量、存活和查询。','H',20);
table(st,5,['日期','STATIC完成','STATIC成本变化/FULL','STATIC查询','GNN平均查询','GNN查询减少/STATIC','GNN成本变化/FULL','GNN成本增益/FULL'],data.static.map(r=>[r.week,r.complete==='True'?1:0,+r.cost_delta_full,+r.static_queries,+r.gnn_seed_average_queries,null,null,+r.gnn_cost_gain_of_full]),[22,18,26,18,21,26,26,25]);
for(let i=0;i<data.static.length;i++){
 const row=i+6;st.getRange(`F${row}:G${row}`).formulas=[[`=IF(D${row}=0,"n.a.",1-E${row}/D${row})`,`=C${row}-H${row}`]];
}
st.getRange('C6:C13').setNumberFormat(pct);st.getRange('D6:D13').setNumberFormat('#,##0');st.getRange('E6:E13').setNumberFormat('#,##0.0');st.getRange('F6:H13').setNumberFormat(pct);
st.getRange('F6:F13').format.horizontalAlignment='right';
st.getRange('A16').values=[['STATIC保留作者教师频次顺序；有历史知识，不是随机顺序。5月双方零查询，比例不定义。']];

const p=sheets['新先验逐次'];
base(p,'频次先验GNN试点：16条同批回放','四个旧开发日期、单次计时；六月全部在1901步共同失败，完整周成本变化不定义。','Q',27);
const pheaders=['日期','方法ID','完成(1/0)','步数','累计模拟成本','控制时间(s)','N1查询','N1搜索时间(s)','先验(s)','图构造(s)','前向排序(s)','成本变化/FULL','控制减少/FULL','查询减少/STATIC','控制减少/STATIC','搜索差/STATIC(s)','非搜索差/STATIC(s)'];
table(p,5,pheaders,data.pilot.map(r=>[r.week,r.method,+r.complete,r.steps,r.cost,r.controller_s,r.n1_forecasts,r.n1_source_s,r.prior_s,r.feature_s,r.forward_rank_s,null,null,null,null,null,null]),[22,27,14,14,23,20,17,21,17,18,20,21,21,25,25,25,27]);
const pidx=new Map(data.pilot.map((r,i)=>[`${r.week}|${r.method}`,i+6]));
for(let i=0;i<data.pilot.length;i++){
 const r=data.pilot[i],row=i+6,f=pidx.get(`${r.week}|FULL`),s=pidx.get(`${r.week}|STATIC_ID`);
 p.getRange(`L${row}:Q${row}`).formulas=[[`=IF(AND(C${row}=1,C${f}=1),(E${row}-E${f})/ABS(E${f}),"n.a.")`,`=1-F${row}/F${f}`,`=1-G${row}/G${s}`,`=1-F${row}/F${s}`,`=H${row}-H${s}`,`=(F${row}-H${row})-(F${s}-H${s})`]];
}
p.getRange('E6:F21').setNumberFormat('#,##0.00');p.getRange('G6:G21').setNumberFormat('#,##0');p.getRange('H6:K21').setNumberFormat('0.000');p.getRange('L6:O21').setNumberFormat(pct);p.getRange('P6:Q21').setNumberFormat('0.000');
p.getRange('L6:L21').format.horizontalAlignment='right';p.getRange('L6:L21').setNumberFormat('0.00000%;-0.00000%;0.00000%');
p.getRange('A24').values=[['停止先验修订：平均查询收益不足5%，每种子仅2/4日期正向；成本尺度与无提前失败均通过。']];
p.freezePanes.freezeRows(5);p.freezePanes.freezeColumns(2);

const t=sheets['计时与路径'];base(t,'同批差异分解：GNN相对STATIC','控制差=搜索差+搜索外差；正数表示GNN多耗时。分解是账目恒等式，不是因果识别。','J',18);
table(t,5,['日期','模型种子','查询减少/STATIC','控制减少/STATIC','控制差(s)','N1搜索差(s)','搜索外差(s)','图先验前向(s)','动作不同步数','后状态不同数'],data.diagnostics.map(r=>[r.week,+r.seed,+r.query_reduction_static,+r.control_reduction_static,+r.control_delta_s,+r.n1_source_delta_s,+r.outside_n1_delta_s,+r.graph_prior_forward_s,+r.action_hash_differences,+r.observation_hash_differences]),[22,15,25,25,21,23,23,25,23,23]);
t.getRange('C6:D13').setNumberFormat(pct);t.getRange('E6:H13').setNumberFormat('0.000');

const notes=sheets['口径与来源'];base(notes,'数据口径、方法和来源','原始文件与哈希位于同一数据包；所有旧版本、较差结果和共同失败保留。','C',30);
const noteRows=[
 ['控制时间','每步选择动作的实际墙钟累计','包含排序、图构造、模型推理、公开预测仿真和连续优化；不含环境物理推进、初始化及保存。'],
 ['模拟成本','累计奖励中的成本账目','数值越低越好；不是生产电费。成本变化=(方法成本-FULL成本)/|FULL成本|。'],
 ['N1查询','拓扑候选的公开预测调用数','不等于物理运行步数，也不等于全部控制器预测数。所有方法可使用共同DC/safeQP缓存。'],
 ['宏平均','先对每个日期计算比例再等权平均','15.75%是宏平均控制提速。先加总时间再求比例约13.92%；不能互换。'],
 ['两种子','两份固定训练随机种子','不按结果选种子。种子与计时重复不是独立日期样本。'],
 ['原GNN','LOCAL_GNN0/1','PPO先验与tanh残差；更快但完整周成本尾部超0.5%尺度。'],
 ['质量补强GNN','COMPRESSED_GNN0/1','PPO先验与软压缩残差；不是最近的频次先验模型。'],
 ['新先验GNN','FREQUENCY_GNN0/1','改用作者教师频次顺序作先验；只在四个旧开发日期有限试点，不能与八日期结果横比均值。'],
 ['STATIC','作者教师频次顺序','同一公开动作库有历史经验；所有方法共享库，作者训练日期未知。'],
 ['SoftGNN','资源匹配的适配版','使用作者架构与转换器，未声称完整复现论文全部训练协议。'],
 ['训练性质','当前GNN是监督候选排序','不是本轮训练新的RL策略。已有作者PPO是控制器组成。'],
 ['八日期正式来源','GRID55B qualified_view','基于GRID55原128条；受待机影响的一日期对所有八方法统一换计时槽位，来源JSON与原数据保留。'],
 ['八日期STATIC来源','GRID57 per_family.csv','单次额外回放，不能跨批声称STATIC控制提速。查询和完整周成本可离线对照。'],
 ['四日期试点来源','GRID59 finished/review.json','新16条同批、一次计时；两个日期与早期训练家庭重合，不是封存泛化确认。'],
 ['计时路径来源','GRID59_closeout paired_diagnostics.csv','保存hash和反馈验算；没有独立AC重求解，未在本导出中重新拟合。'],
 ];
table(notes,5,['项目','定义或方法','解释与边界'],noteRows,[25,43,112]);notes.getRange('A6:C20').format.rowHeight=33;
wb.recalculate();
const validation=[];
for(let i=0;i<data.formal_summary.length;i++){
 const s=data.formal_summary[i],v=summary.getRange(`B${i+6}:I${i+6}`).values[0];
 const expected=[s.completed_runs,s.quality_qualified_pairs,s.macro_time_reduction_full??0,1-s.queries/data.formal_summary[0].queries,s.control_s/16,
  null,s.max_complete_week_cost_regret,s.mean_family_p95_decision_s*1000];
 for(let j=0;j<expected.length;j++)if(expected[j]!==null&&Math.abs(v[j]-expected[j])>1e-8)throw new Error(`Summary mismatch ${s.method}:${j} ${v[j]} ${expected[j]}`);
 validation.push({method:s.method,values:v});
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});
const testRaw=data.formal.findIndex(r=>r.method==='COMPRESSED_GNN0')+6;
const testSummary=data.formal_summary.findIndex(r=>r.method==='COMPRESSED_GNN0')+6;
const originalTime=formal.getRange(`H${testRaw}`).values[0][0];
const originalMean=summary.getRange(`F${testSummary}`).values[0][0];
formal.getRange(`H${testRaw}`).values=[[originalTime+10]];wb.recalculate();
if(Math.abs(summary.getRange(`F${testSummary}`).values[0][0]-(originalMean+10/16))>1e-8)throw new Error('Summary did not recalculate');
formal.getRange(`H${testRaw}`).values=[[originalTime]];wb.recalculate();
if(Math.abs(summary.getRange(`F${testSummary}`).values[0][0]-originalMean)>1e-8)throw new Error('Restore failed');
await fs.writeFile(path.join(out,'formula_check.ndjson'),errors.ndjson);
if(/"value":"#/.test(errors.ndjson))throw new Error('Formula error');
const regions={'正式比较':'A1:I20','正式逐次':'A1:N12','频次顺序消融':'A1:H17','新先验逐次':'A1:Q10','计时与路径':'A1:J14','口径与来源':'A1:C21'};
for(const n of names){
 if(process.argv.includes('--recalculation-check'))continue;
 if(process.argv.includes('--format-revision')&&!['正式比较','频次顺序消融','新先验逐次'].includes(n))continue;
 const blob=await wb.render({sheetName:n,range:regions[n],scale:1,format:'png'});
 await fs.writeFile(path.join(out,`${n}_preview.png`),new Uint8Array(await blob.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'GRID_数据与对比.xlsx'));
await fs.writeFile(path.join(out,'workbook_validation.json'),JSON.stringify({passed:true,sheets:6,formal_runs:128,pilot_runs:16,verified_summary_rows:validation,input_change_recalculated_and_restored:true,no_new_simulations:true,no_new_fits:true,native_excel_execution:false},null,2));
console.log(JSON.stringify({saved:true,sheets:6,formal_runs:128,pilot_runs:16}));
