import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const ROOT='D:/Codex/GRL研究';
const OUT=path.join(ROOT,'outputs/frame_experiment_inventory_20261019');
const data=JSON.parse(await fs.readFile(path.join(OUT,'experiment_tables.json'),'utf8'));
const wb=Workbook.create();
const ss={};
for(const n of ['贡献总表','质量速度汇总','逐日期数据','逐轮记录','原报告指标','对照与口径'])ss[n]=wb.worksheets.add(n);
const col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
const NAVY='#263B53', INK='#263342', MUTED='#677788';
const catColor={'A':'#E6EEF7','B':'#E9E5F2','C':'#E4F1E9','D':'#E7EDF6','E':'#F3EDE3','F':'#E5EEF0','G':'#ECEAF2','H':'#F5E6E4'};
function writeSheet(name,title,note,headers,rows,widths,tableName,{fontSize=11,height=50}={}){
  const sh=ss[name];sh.showGridLines=false;
  const last=col(headers.length-1),end=6+rows.length;
  sh.getRange('A2').values=[[title]];
  sh.getRange('A2').format.font={name:'Noto Sans SC',size:20,bold:true,color:INK};
  sh.getRange('A2').format.rowHeightPx=40;
  sh.getRange('A4').values=[[note]];
  sh.getRange('A4').format.font={name:'Noto Sans SC',size:10,color:MUTED};
  sh.getRange('A4').format.rowHeightPx=30;
  sh.getRange('A6:'+last+end).values=[headers,...rows];
  const area=sh.getRange('A6:'+last+end);
  area.format.font={name:'Noto Sans SC',size:fontSize,color:INK};
  area.format.verticalAlignment='center';area.format.wrapText=true;
  area.format.rowHeightPx=height;
  const table=sh.tables.add('A6:'+last+end,true,tableName);
  table.showFilterButton=true;table.style='TableStyleLight9';
  const head=sh.getRange('A6:'+last+'6');
  head.format.fill=NAVY;head.format.font={name:'Noto Sans SC',size:11,bold:true,color:'#FFFFFF'};
  head.format.horizontalAlignment='center';head.format.rowHeightPx=44;
  widths.forEach((w,i)=>sh.getRange(col(i)+'6:'+col(i)+end).format.columnWidthPx=w);
  sh.freezePanes.freezeRows(6);sh.freezePanes.freezeColumns(2);
  if(rows.length){
    rows.forEach((row,i)=>{
      let lines=1;
      row.forEach((v,j)=>{
        if(typeof v!=='string')return;
        const hard=v.split('\n');
        const n=hard.reduce((a,s)=>a+Math.max(1,Math.ceil([...s].reduce((n,c)=>n+(c.charCodeAt(0)>255?1.8:1),0)*fontSize*.64/(widths[j]-16))),0);
        lines=Math.max(lines,n);
      });
      sh.getRange('A'+(i+7)+':'+last+(i+7)).format.rowHeightPx=Math.max(height,Math.min(300,lines*(fontSize*1.5+3)+18));
    });
  }
  return sh;
}

// Main view contains the purpose and conclusion once; numerical source work is on the right.
const mainRows=data.main.map(r=>[
  r[0],r[1]+'\n'+r[2]+'\n'+r[4],r[3],r[5].replace(/。$/,''),r[6].replace(/。$/,''),r[7].replace(/。$/,''),
  [...new Set((r[8].match(/GRID\d{2}[BC]?/gi)||[r[1]]).map(x=>x.toUpperCase()))].join('、')+(r[8].includes('frame_cn_v4')?'；中文V4':'')
]);
writeSheet('贡献总表','FRAME实验总表：比较、结果与贡献',
 '截至2026-10-19。38项贡献证据；另保留96轮记录。质量、速度数值见下一页，出处见逐轮/逐日期/原报告指标。',
 ['类别','实验与场景','和谁比较','实验目的：检验FRAME什么特点','结果','证据说明','来源编号'],
 mainRows,[140,260,225,305,570,340,145],'ContributionEvidence',{fontSize:12,height:100});
data.main.forEach((r,i)=>ss['贡献总表'].getRange('A'+(i+7)).format.fill=catColor[r[0][0]]||'#EEF1F4');
ss['贡献总表'].tabColor='#263B53';

const dateHeaders=['批次','日期/场景','方法','计时重复数','运行步数','质量达标(1/0)','完整周(1/0)','累计运行成本','完整周成本变化/FULL','查询数(日期计一次)',
 '完整控制(s)','预测仿真(s)','连续优化(s)','搜索其他(s)','其余控制(s)','全步骤P95(s)','全步骤P99(s)','搜索步P95(s)','搜索步P99(s)',
 '重复最小控制(s)','重复最大控制(s)','重复标准差(s)','同批速度参照','查询减少比','控制时间减少比','全步骤P95减少比','搜索步P95减少比','重复CV','数据来源'];
const dateRows=data.dates.map(d=>[d.batch,d.date,d.method,d.repeats,d.steps,d.quality,d.complete,d.cost,d.cost_delta,d.queries,
 d.controller,d.forecast,d.optimizer,d.search_other,d.other,d.all_p95,d.all_p99,d.search_p95,d.search_p99,d.min,d.max,d.sd,d.baseline||'无同批参照',
 null,null,null,null,d.cv,d.source]);
writeSheet('逐日期数据','逐日期质量、速度与重复波动',
 '质量/成本相对冻结FULL。搜索P95/P99没有搜索时留空。成本变化只在可比完整周填值；单次计时的标准差/CV留空。',
 dateHeaders,dateRows,[190,190,285,90,100,110,110,160,150,150,150,150,150,150,150,155,155,155,155,160,160,160,250,145,160,165,165,130,600],
 'PerDateEvidence',{height:48});
const ds=ss['逐日期数据'];
for(let i=0;i<data.dates.length;i++){
  const d=data.dates[i],r=i+7;
  const baseIndex=data.dates.findIndex(z=>z.batch===d.batch&&z.date===d.date&&z.method===d.baseline);
  if(baseIndex>=0){
    const b=baseIndex+7;
    ds.getRange('X'+r+':AA'+r).formulas=[[
      '=IF(AND(J'+b+'>0,E'+r+'=E'+b+'),1-J'+r+'/J'+b+',"")',
      '=IF(AND(K'+b+'>0,E'+r+'=E'+b+'),1-K'+r+'/K'+b+',"")',
      '=IF(AND(ISNUMBER(P'+b+'),P'+b+'>0,E'+r+'=E'+b+'),1-P'+r+'/P'+b+',"")',
      '=IF(AND(ISNUMBER(R'+r+'),ISNUMBER(R'+b+'),R'+b+'>0,E'+r+'=E'+b+'),1-R'+r+'/R'+b+',"")'
    ]];
  }
}
const de=6+dateRows.length;
ds.getRange('D7:G'+de).setNumberFormat('0');
ds.getRange('H7:H'+de).setNumberFormat('#,##0.00');
ds.getRange('I7:I'+de).setNumberFormat('0.000%');
ds.getRange('J7:J'+de).setNumberFormat('#,##0');
ds.getRange('K7:V'+de).setNumberFormat('0.000');
ds.getRange('X7:AB'+de).setNumberFormat('0.00%');
ds.getRange('F7:F'+de).conditionalFormats.add('cellIs',{operator:'equal',formula:0,format:{fill:'#F9E5E2',font:{color:'#9A2925'}}});
ds.getRange('I7:I'+de).conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0.005,format:{fill:'#F9E5E2',font:{color:'#9A2925'}}});

// Group rows have contiguous date records. Aggregations are visible Excel formulas.
const groupRank=b=>b.includes('G2主评价')?0:b==='GRID81'?1:b==='GRID88'?2:b==='GRID86'?3:b==='GRID84'?4:b.includes('META费用')?5:b.includes('换序')?6:b.includes('参数')?7:8;
const methodRank=m=>m==='FULL'?0:m==='FRAME_GUARD'?1:m==='GLOBAL_MATCH'?2:m==='FIXED_256'?3:m==='FIXED_512'?4:5;
const groups=[...data.groups].sort((a,b)=>groupRank(a.batch)-groupRank(b.batch)||a.batch.localeCompare(b.batch)||methodRank(a.method)-methodRank(b.method)||a.method.localeCompare(b.method));
const summaryHeaders=['批次','方法','场景数','质量达标','整周完成','步数最小–最大','平均完整周成本变化','最大完整周成本变化','查询合计','查询宏平均减少','控制秒/日期',
 '控制时间宏平均减少','搜索秒/日期','预测秒/日期','优化秒/日期','搜索其他秒/日期','其余控制秒/日期','全步骤P95(ms)','全步骤P99(ms)','搜索步P95(ms)','搜索步P99(ms)',
 '全步骤P95宏平均减少','搜索P95宏平均减少','日期内CV范围','计时重复数','同批速度参照','质量及时间解释','来源'];
const summaryRows=groups.map(g=>[g.batch,g.method,...Array(22).fill(null),null,g.baseline||'无同批参照',
 g.batch==='GRID87'?'仅固定256新计时，不与旧FULL/GUARD混算速度':'质量/成本以冻结FULL为参照；速度方向为本行方法相对同批参照',
 g.batch.split(' ')[0]+' / per_date.csv']);
writeSheet('质量速度汇总','质量与速度：多指标汇总',
 '60个方法×批次。日期等权、先平均重复。比例为“本行方法相对同批参照减少”；质量不足时保留时间但不称同质量加速。',
 summaryHeaders,summaryRows,[190,285,80,100,100,150,165,165,135,165,155,180,155,155,155,180,180,170,170,170,170,190,190,190,105,260,435,210],
 'QualitySpeedSummary',{height:72});
const sm=ss['质量速度汇总'];
const range=(c,a,b)=>"'逐日期数据'!"+c+a+':'+c+b;
const avg=(c,a,b,mul=1)=>'=IF(COUNT('+range(c,a,b)+')=0,"未测",AVERAGE('+range(c,a,b)+')'+(mul===1?'':'*'+mul)+')';
for(let i=0;i<groups.length;i++){
  const g=groups[i],ids=data.dates.map((d,j)=>d.batch===g.batch&&d.method===g.method?j:-1).filter(j=>j>=0);
  if(ids.at(-1)-ids[0]+1!==ids.length)throw Error('noncontiguous group');
  const a=ids[0]+7,b=ids.at(-1)+7,r=i+7,R=c=>range(c,a,b);
  const formulas=[
   '=ROWS('+R('A')+')',
   '=SUM('+R('F')+')&"/"&C'+r,
   '=SUM('+R('G')+')&"/"&C'+r,
   '=MIN('+R('E')+')&"–"&MAX('+R('E')+')',
   avg('I',a,b),
   '=IF(COUNT('+R('I')+')=0,"未测",MAX('+R('I')+'))',
   '=SUM('+R('J')+')',avg('X',a,b),avg('K',a,b),avg('Y',a,b),
   '=IF(OR(COUNT('+R('L')+')=0,COUNT('+R('N')+')=0),"未测",AVERAGE('+R('L')+')+AVERAGE('+R('N')+'))',
   avg('L',a,b),avg('M',a,b),avg('N',a,b),avg('O',a,b),
   avg('P',a,b,1000),avg('Q',a,b,1000),avg('R',a,b,1000),avg('S',a,b,1000),
   avg('Z',a,b),avg('AA',a,b),
   '=IF(COUNT('+R('AB')+')=0,"单次，未估计",TEXT(MIN('+R('AB')+'),"0.00%")&"～"&TEXT(MAX('+R('AB')+'),"0.00%"))',
   '=MIN('+R('D')+')&"～"&MAX('+R('D')+')'
  ];
  sm.getRange('C'+r+':Y'+r).formulas=[formulas];
}
const se=6+groups.length;
sm.getRange('G7:H'+se).setNumberFormat('0.000%');
for(const c of ['J','L','V','W'])sm.getRange(c+'7:'+c+se).setNumberFormat('0.00%');
sm.getRange('I7:I'+se).setNumberFormat('#,##0');
for(const [l,r] of [['K','K'],['M','U']])sm.getRange(l+'7:'+r+se).setNumberFormat('0.000');
sm.getRange('H7:H'+se).conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0.005,format:{fill:'#F9E5E2',font:{color:'#9A2925'}}});
sm.tabColor='#405B78';

const hcols=['编号','类别','实验名称与版本','比较对象','实验目的：检验什么','结果摘要','与当前论文的关系','来源'];
writeSheet('逐轮记录','全部96轮：实验、探索与资格记录',
 'GRID00–89及六个B/C补充轮。原版、修复版、部分完成与未启动余项分开保留；拟合、离线重放、闭环、工程审计不混算。',
 hcols,data.history.map(r=>hcols.map(c=>r[c]||'')),[95,155,285,285,350,650,300,650],'StageHistory',{height:90});
writeSheet('原报告指标','各报告原表指标',
 '4710个原表单元逐项保留。原列名、计数单位与源版本不改；同一轮不同报告或两次计时不是新的独立实验。',
 ['编号','报告文件','原表与章节','原行上下文','原指标/列名','原值','指标类别','来源路径'],
 data.raw_metrics,[100,300,420,500,370,450,150,700],'ReportedMetrics',{height:48});
const dict=[...data.dictionary,['来源','中文V4',ROOT+'/outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md']];
writeSheet('对照与口径','读表所需的方法、场景与统计口径',
 '先分清版本、场景和统计单位，再比较质量与速度。原表指标保留历史定义；新的标准化汇总在本表定义下计算。',
 ['类别','名称','含义'],dict,[150,310,1060],'Definitions',{height:58});

wb.recalculate();
const checks=[];
for(const [name,rg] of [['质量速度汇总','A6:L11'],['贡献总表','A6:G8'],['逐日期数据','A6:L8']]){
 const res=await wb.inspect({kind:'table',range:name+'!'+rg,include:'values,formulas',tableMaxRows:6,tableMaxCols:12,tableMaxCellChars:110,maxChars:3500});
 checks.push({name,result:res.ndjson});console.log(res.ndjson);
}
const err=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'formula error scan',maxChars:3000});
checks.push({errors:err.ndjson});console.log(err.ndjson);
await fs.mkdir(path.join(OUT,'qa'),{recursive:true});
for(const [name,range,file] of [
 ['贡献总表','A6:E9','01_贡献总表'],
 ['贡献总表','E6:G9','02_贡献结果'],
 ['质量速度汇总','A6:L11','03_质量与控制时间'],
 ['质量速度汇总','M6:AB11','04_分解与延迟'],
 ['逐日期数据','A6:K10','05_逐日期'],
 ['逐轮记录','A6:G10','06_逐轮记录'],
 ['原报告指标','A6:G12','07_原表指标'],
 ['对照与口径','A6:C12','08_定义']
]){
 if(process.env.RENDER_CHANGED_ONLY && !['01_贡献总表','02_贡献结果','07_原表指标'].includes(file))continue;
 const preview=await wb.render({sheetName:name,range,scale:1.2,format:'png'});
 await fs.writeFile(path.join(OUT,'qa',file+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(OUT,'qa/inspection.json'),JSON.stringify(checks,null,2));
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(OUT,'FRAME_全部实验分类与质量速度指标_20261019.xlsx'));
await fs.writeFile(path.join(OUT,'workbook_build.json'),JSON.stringify({sheets:Object.keys(ss),stages:data.history.length,mainRows:data.main.length,dateRows:data.dates.length,groups:groups.length,rawMetricRows:data.raw_metrics.length,recalculated:true},null,2));
console.log('EXPORTED');
