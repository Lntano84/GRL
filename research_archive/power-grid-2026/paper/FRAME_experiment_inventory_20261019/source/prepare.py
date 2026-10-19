"""Read existing experiment evidence; no benchmark, fitting or physics calls."""
from pathlib import Path
from collections import defaultdict
from statistics import mean,stdev
import csv,hashlib,json,re
ROOT=Path('D:/Codex/GRL研究'); WORK=ROOT/'work/frame_experiment_inventory_20261019'
OUT=ROOT/'outputs/frame_experiment_inventory_20261019'
sources={}
def register(p):
    p=Path(p); assert p.exists(),p
    sources[str(p).replace('\\','/')]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return str(p).replace('\\','/')
def readcsv(p):
    register(p)
    return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def number(v):
    if v in [None,'','nan','None']:return None
    return float(v)
def truth(v):return str(v).lower()=='true' or v==1
inventory=json.loads((OUT/'source_inventory.json').read_text(encoding='utf-8'))
imap={r['id']:r for r in inventory}
history=list(csv.DictReader((WORK/'curated.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
assert len(history)==96 and len({r['编号'] for r in history})==96
assert set(r['编号'] for r in history)==set(imap)
for r in history:
    info=imap[r['编号']]
    paths=[e['path'] for e in info['reports']]
    # Root state-only entries use the actual compact result/qualification receipt as well.
    if r['编号'] in ['GRID27','GRID43','GRID49','GRID54','GRID58','GRID64','GRID66','GRID69','GRID80']:
        paths += [e['path'] for k,e in info['structured_files'].items() if k!='design.json']
    if r['编号']=='GRID42':
        paths=[str(Path(info['folder'])/'engineering_failure.json'),str(Path(info['folder'])/'design.json')]
    if r['编号'] in ['GRID08','GRID09','GRID10']:
        paths.insert(0,str(Path(info['folder'])/(r['编号']+'_review_zh.md')))
    r['来源']= '\n'.join(register(p) for p in paths[:4])

# Preserve all root-report tables as a searchable long table, including pilots and RAW versions.
raw=[]
for info in inventory:
    for report in info['reports']:
        if 'report' not in report['name'].lower() and 'review_zh' not in report['name'].lower():continue
        src=register(report['path']); lines=report['text'].splitlines(); heading=''; tnum=0;i=0
        while i<len(lines):
            line=lines[i]
            if line.startswith('#'):heading=line.lstrip('# ').strip()
            if line.strip().startswith('|') and i+1<len(lines) and re.match(r'^\s*\|?[\s:|\-]+\|',lines[i+1]):
                headers=[x.strip() for x in line.strip().strip('|').split('|')]
                tnum+=1;i+=2
                while i<len(lines) and lines[i].strip().startswith('|'):
                    vals=[x.strip().replace('**','') for x in lines[i].strip().strip('|').split('|')]
                    if len(vals)!=len(headers):
                        raise ValueError(f'table width {src}:{i+1}')
                    context='；'.join(f'{headers[j]}={vals[j]}' for j in range(min(2,len(vals))))
                    for j in range(1,len(vals)):
                        metric=headers[j]
                        dim=('质量/成本' if re.search('质量|成本|费用|存活|完整周|完成|失败|rho|ρ|保留|遗憾|选择|步数|reward',metric,re.I)
                             else '速度/计算' if re.search('时间|秒|耗时|查询|预测|调用|提速|减少|P95|P99|控制',metric,re.I) else '其他/配置')
                        raw.append([info['id'],report['name'],f'{tnum}. {heading}',context,metric,vals[j],dim,src])
                    i+=1
                continue
            i+=1

main=[]
def add(cat,id,name,vs,sample,purpose,result,judgment,paths=None):
    r=next(r for r in history if r['编号']==id)
    main.append([cat,id,name,vs,sample,purpose,result,judgment,
                 '\n'.join(register(ROOT/p) for p in paths) if paths else r['来源'].split('\n')[0]])
add('A 整体有效性','GRID89','最终FRAME的主评价','FULL','G2六场景；三次轮换计时',
    '检验少查询能否保持闭环效果并减少完整控制计算。',
    '质量6/6，双方完成5/6且终止步数相同；5个可比完整周成本均未增加。查询宏平均少60.728%，完整控制时间少14.301%；日期簇描述区间9.955%～18.739%。',
    '当前最直接的主结果；5%同质量筛查通过。重复不当独立场景。')
add('A 整体有效性','GRID81','固定核查版的第一组生成场景','FULL','G1六场景；两次反向计时',
    '检验最终固定搜索在生成时序中的运行质量与计算收益。',
    '质量6/6，双方完整4/6，最坏完整周成本+0.0866%；查询少65.10%，完整控制时间少14.98%。',
    '与G2分别报告，不能合成12个不同日历日期。')
add('A 整体有效性','GRID70','最终修复版的旧日期回归','原FRAME与FULL','D组13日期；单次新回放',
    '检验增加必要核查、限制无依据补查后能否修复质量。',
    '质量13/13，完成11/13同FULL；原版完成10/13。活跃匹配日期查询少61.999%，最坏成本+0.372%。',
    '当前算法实现锚点；本批不建立新同批重复加速结论。')
add('B 搜索深度','GRID85','安全前缀之后为何仍需核查','前缀动作与核查后动作','D组101次调用；G1为30次',
    '检验前缀达到0.9阈值是否意味着继续搜索没有价值。',
    '前缀低于0.9后仍改变动作：D组12/39，G1为2/12；原版与修复版7处首次分歧均来自该分支。',
    '支持保留核查机会；一步改善不等于整周成本必降。')
add('B 搜索深度','GRID86','低预算是否已经足够','FIXED_256','G1六场景；两次反向计时',
    '检验反馈深度比简单低预算是否更划算。',
    '双方质量6/6、完成4/6；FRAME查询更多（相对固定256的宏平均节省−45.590%），完整控制慢2.021%。',
    '重要边界：这一场景组低预算确实更便宜，完整保留。')
add('B 搜索深度','GRID86','与较大固定预算比较','FIXED_512','G1六场景；两次反向计时',
    '检验动态深度能否避免统一大预算的多余计算。',
    '双方质量6/6、完成4/6；FRAME查询少40.893%，完整控制时间少7.099%；两次均同向。',
    '通过本轮对512的计算优势判据；不抹去对256的反向结果。')
add('B 搜索深度','GRID87','固定256跨旧日期的质量','冻结FULL及FRAME_GUARD','D组13日期；固定256单次',
    '检验G1里便宜的固定256能否覆盖更难旧场景。',
    '固定256质量9/13，FRAME13/13；均完整11/13；三月固定版成本+4.4242%，FRAME+0.372%。',
    '说明统一低预算有质量缺口；未同批运行FULL，不比较旧时间。')
add('B 搜索深度','GRID88','提高固定预算能否兼顾质量与效率','FIXED_512','D组13日期；两次反向计时',
    '检验简单增加查询是否替代反馈分配。',
    'FRAME/固定512质量13/13与11/13，均完整11/13；FRAME查询少33.573%，控制少3.883%；全部步P95仅少约0.42%。',
    '质量与计算组合较好；没有通过双方同质量且时间至少5%的判据。')
add('B 搜索深度','GRID89','最终独立种子下的固定预算比较','FIXED_512','G2六场景；三次轮换计时',
    '检验新随机时序下反馈搜索与固定大预算的质量—计算表现。',
    '质量6/6对5/6，均完整5/6；FRAME查询少32.737%、控制少5.480%；固定版最坏成本+0.9463%。',
    '对手质量未全通过，时间为描述性比较，不称双方同质量加速。')
add('C 补查位置','GRID61','原框架取消局部补查','NO_LOCAL、FIRST_SAFE_BEST','四开发日期×两重复',
    '检验补查是否帮助弥补全局截断后的质量缺口。',
    '按回放计质量：FRAME8/8、NO_LOCAL6/8、首安全4/8；最坏成本+0.372%/+4.810%/+3.952%。',
    '旧版机制支持；追加次数不同，单独不能排除多查询解释。')
add('C 补查位置','GRID62','补查增量很小的剩余日期','NO_LOCAL与FRAME','三个当时剩余日期；两重复',
    '检验旧版补查收益是否跨日期稳定。',
    '两者均质量3/3，最坏完整周成本均约+0.014%；FRAME较FULL控制少6.086%，NO_LOCAL少9.384%。',
    '不利边界：该组NO_LOCAL已经接近，补查增量有限。')
add('C 补查位置','GRID63','原版等预算闭环','GLOBAL_MATCH','旧七日期；两重复',
    '区分结构选点与仅增加查询数量。',
    'FRAME和GLOBAL均质量7/7，查询节省均74.092%；对FULL控制少14.016%/13.721%，结构增量不实质。',
    '保留早期否定结果；算法为提前返回修复前版本。')
add('C 补查位置','GRID85','最终版同状态等预算，旧日期','NO_LOCAL与GLOBAL_MATCH','FULL的100个有可用候选状态',
    '在相同状态、相同追加次数下检验选点作用。',
    '同FULL动作数65/71/84（NO_LOCAL/GLOBAL/FRAME）；补查排序键14胜0负，实质奖励优势13处。',
    '同状态一步机制证据，不当作100个独立场景。')
add('C 补查位置','GRID85','最终版同状态等预算，G1','GLOBAL_MATCH','28个有可用候选状态；18处触发',
    '检查结构选点是否在第一组生成场景持续占优。',
    '同站3胜1负；反向案例两边各追加198查询，全局奖励增0.012966，同站仅0.000606。日期宏平均遗憾FRAME0.000871，全局0.000089。',
    '必须保留：胜的次数更多不代表平均改善更大。',['outputs/grid85_mechanism/GRID85_report.md','outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md'])
add('C 补查位置','GRID89','最终版同状态等预算，G2','GLOBAL_MATCH','新FULL访问的可比状态',
    '在新随机时序里复查同站位置的收益。',
    '同站奖励更好4处、全局更好0处；按有可比状态日期平均后，遗憾FRAME0.001153、全局0.001823。',
    '支持当前组的选点作用，与G1反例共同解释适用条件。',['outputs/grid89_review/GRID89_report.md','outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md'])
add('C 补查位置','GRID89','最终版等预算闭环，G2','GLOBAL_MATCH','六场景；三次轮换计时',
    '检验接近查询规模下的补查位置能否影响长期效果。',
    '质量6/6对5/6，均完整5/6；FRAME查询少0.277%、控制少0.785%；GLOBAL最坏完整周成本+1.1494%。',
    '主要证据是质量差异，不能把0.785%写成明显速度优势。')
add('D 会议方法比较','GRID65','早期停止限定META适配','META_DQN两个种子','旧七日期；两重复',
    '检验规则搜索与会议来源停止策略的竞争力。',
    'FRAME质量7/7，两个META6/7、5/7；接口还没有共同上下文和完整核查动作。',
    '适配历史；主要以补齐权限后的比较为准。')
add('D 会议方法比较','GRID77','共同信息与动作权限','两个META种子×三种费用','129训练、22校准记录',
    '保证对手可以访问相同上下文、返回/续查/委托FRAME。',
    '6次拟合完成；共同校准质量匹配False，按预设备选固定费用0。没有新教师查询。',
    '配置与公平性说明，不是质量胜出实验。')
add('D 会议方法比较','GRID79','门控版同批会议比较','META_CONTEXT0/1及FULL','旧七日期；两重复',
    '核验早期门控版在完整控制时间上的表现。',
    '门控质量7/7、META均6/7；门控对FULL控制少12.056%，对META少7.241%/2.976%。',
    '门控版，不混成最终FRAME；对手质量不足需单列。')
add('D 会议方法比较','GRID82','最终FRAME两种子同批比较','META_CONTEXT0/1，费用0','G1六场景；两重复',
    '最终固定规则与同权限学习控制的直接比较。',
    '双方质量均6/6；FRAME对种子0/1控制时间少6.435%/0.017%，后一两重复方向不同。',
    '不选有利种子；后续五种子更完整。')
add('D 会议方法比较','GRID84','五种子稳定性','META_CONTEXT0–4，费用0','G1六场景；两重复',
    '检验小幅平均优势是否依赖某个训练种子。',
    '全部质量6/6，均完整4/6；FRAME平均时间少2.228%，中位1.325%，范围−0.296%～7.550%；强提速通过1/5。',
    '支持小幅平均计算优势；整体加强判据False。')
add('D 会议方法比较','GRID83','META实际委托行为','META首决策与FRAME子程序','既有调用，不新增回放',
    '明确学习适配到底做了哪些独立决策。',
    '五种子首决策委托FRAME比例后续为12.5%、90.0%、76.7%、43.3%、56.7%；适配包含FRAME。',
    '不能将其描述为纯独立原SOTA任务完整复现。',['outputs/grid84/GRID84_report.md','outputs/grid83_alignment/ICAPS2022对标与贡献重写方案.md'])
add('D 会议方法比较','GRID89','非零计算成本复查','META费用0和0.0025，各五种子','G1六场景；单次新计时',
    '检验零成本训练目标是否影响比较公平性。',
    '费用0五种子均质量6/6；0.0025的种子0/4质量5/6，其余6/6。FRAME平均50.865秒/日期，非零META各50.505～54.675秒。',
    '完整逐种子列入质量速度表；单次计时不建立稳定同质量优势。')
add('E 顺序与参数','GRID89','候选顺序来源核实','动作库编号与教师频次','909原始动作的元数据',
    '解释为什么评估前缀有合理优先级。',
    '编号顺序承接教师使用频次降序；前128占原库频次44.38%。',
    '44.38%不是当前状态命中率，也不是查询节省。')
add('E 顺序与参数','GRID89','候选换序','默认、随机11/23/37、负载顺序','G1六场景；各单次',
    '检验框架是否只适用于某个候选顺序。',
    '质量依次6/6、6/6、5/6、5/6、6/6；匹配活跃时域查询节省53.453%～65.097%。',
    '不同顺序仍省查询，质量不完全稳定；未按结果换默认。')
add('E 顺序与参数','GRID89','最终版单因素参数检查','p、h、qmin、patience、rho_safe各预定变体','四旧场景；各单次',
    '检验结论对查询前缀、核查量和停止阈值的依赖。',
    '默认质量4/4、查询9108；p256、qmin512、safe0.85/0.95各3/4；其余预定变体4/4，逐项指标全保留。',
    '参数有影响；本轮不据结果重选参数。')
add('F 速度与耗时来源','GRID89','完整控制互斥分解','FRAME与FULL','G2六场景；三重复',
    '解释查询大降幅如何转化为实际完整控制节省。',
    '控制均值83.878→72.623秒/日期；预测27.78→17.36秒；连续优化50.18→49.73秒。预测节省占总差的大部分。',
    '14.301%为日期比例宏平均，不用均值秒数比替换。',['outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md','outputs/grid89_review/per_date.csv'])
add('F 速度与耗时来源','GRID89','高分位延迟与波动','FRAME与FULL','G2；全步骤及搜索步骤分开',
    '检验平均计算改善是否也体现为决策尾延迟。',
    '搜索步P95/P99分别少39.58%/38.26%；全部步P95增加3.36%，P99少8.40%；日期内完整时间CV为2.51%～8.27%。',
    '搜索尾延迟改善，全部步P95没有同步改善。',['outputs/frame_cn_v4_20261018/FRAME_ICAPS2027_中文重写稿_V4.md','outputs/grid89_review/per_date.csv'])
add('F 速度与耗时来源','GRID26','查询、搜索、整体时间的区别','完整回退版GNN/PPO与FULL','四开发周；三次计时',
    '避免把少查询直接当成同比例完整控制提速。',
    'GNN查询少34.09%，搜索少19.55%/17.14%，完整控制仅少1.40%/0.88%。',
    '历史计时教训，版本不同，不混入当前主表。')
add('F 速度与耗时来源','GRID31','公共优化的贡献','原控制器、DC复用、DC+safeQP','三月两次配对；其他三日见GRID32',
    '划清共有工程缓存和FRAME/GNN算法收益。',
    '双复用三月完整控制少22.42%且轨迹相同；其他三日少61.89%～74.46%。',
    '所有主对照共享，不列为FRAME独有贡献。')
add('G GNN与学习门控必要性','GRID37','静态顺序替换GNN','STATIC_ID、GNN0/1、FULL','四旧日期；两重复',
    '检验是否必须靠GNN才能得到计算收益。',
    '静态质量8/8、控制少14.827%；GNN少14.607%/13.101%，未稳定优于静态。',
    '支持把核心定位放在搜索计算分配。')
add('G GNN与学习门控必要性','GRID57','最终旧GNN版本静态消融','压缩GNN与STATIC_ID','八个已暴露日期',
    '检验模型训练相对免费频次顺序的独有增量。',
    '静态质量8/8；GNN平均查询仅少1.897%，正向4/8，混合结果。',
    'GNN作为辅助，不承担主贡献。')
add('G GNN与学习门控必要性','GRID59','更简单先验的GNN','频次GNN两种子与STATIC_ID','四旧日期；单次配对',
    '检验学习评分是否优于原频次先验。',
    '均质量4/4；GNN查询少3.127%/2.675%，但控制慢0.635%/0.506%。',
    '少量查询增益没有转成实际更快。')
add('G GNN与学习门控必要性','GRID75','学习核查门控资格','普通上下文与结构特征','14校准状态；GRID72–75系列',
    '检验额外学习模型及结构特征是否有必要。',
    '随机森林、小树、探针版均未达10%；电网上下文普通输入11.735%通过，结构5.868%未通过。',
    '结构学习增量没有建立；全部失败保留。')
add('G GNN与学习门控必要性','GRID81','学习门控对固定/随机门控','CONTEXT_GATE、FRAME_GUARD、随机0/1','G1六场景；两重复',
    '检验进一步学何时跳过核查能否优于简单参照。',
    '全部质量6/6；学习门控对固定核查时间变化−0.040%，对随机只快0.299%/0.373%，两重复方向混合。',
    '支持最终保留固定规则；随机增量筛查False。')
add('H 失败与修复边界','GRID68','原版提前返回的质量问题','原FRAME与FULL','六个当时未用于本地结果日期',
    '检查原规则是否在更多场景保持质量。',
    '质量4/6、完整5/6，FULL完整6/6；最大完整周成本+0.641%。',
    '修复动机；不能把不同时域的低累计时间当加速。')
add('H 失败与修复边界','GRID69','第一版修复为何未采用','原版与无条件增强核查/补查','6条试点',
    '检验更积极补查会不会引入新的退化。',
    '七月恢复完整、十二月改善，但三月成本+1.374%；最终版收窄触发，全部历史保留。',
    '说明更多查询和更高一步奖励不自动改善闭环。')
add('H 失败与修复边界','GRID55B','电源待机污染与统一补测','原128条与全八臂统一补测槽位','仅受影响日期新增八条',
    '区分工程计时故障与方法退化。',
    '补测路径/成本/查询完全一致；RAW压缩种子1的异常慢计时保留，统一同槽修正后控制少16.065%。',
    'RAW和修正视图并存，不择有利方法补测。')

# Normalize per-date scientific measurements from completed late-stage batches.
latency=defaultdict(list)
latfile=ROOT/'outputs/grid89_review/evidence/existing_all_and_search_latency.csv'
for r in readcsv(latfile):latency[(r['stage'],r['date'],r['method'])].append(r)
dates=[];groups=[];source_csvs=[]
stages=[79,81,82,84,86,87,88,89]
for stage in stages:
    folder=ROOT/'outputs'/('grid89_review' if stage==89 else f'grid{stage:02}')
    pd=folder/'per_date.csv'; pr=folder/'per_run.csv'
    if not pd.exists():continue
    dr=readcsv(pd); runs=readcsv(pr); source_csvs.extend([str(pd),str(pr)])
    runmap=defaultdict(list)
    for r in runs:runmap[(r.get('phase',''),r['date'],r['method'])].append(r)
    for r in dr:
        phase=r.get('phase',''); batch=f'GRID{stage:02}'+({'order':' 换序','parameters':' 参数','meta':' META费用','sealed':' G2主评价'}.get(phase,''))
        rr=runmap[(phase,r['date'],r['method'])]
        assert rr,(batch,r['date'],r['method'])
        steps={int(z['steps']) for z in rr}; assert len(steps)==1
        ts=[float(z['controller_s']) for z in rr]
        assert abs(mean(ts)-float(r['controller_s']))<1e-5
        d={k:None for k in ['search_p95','search_p99','all_p95','all_p99']}
        for key,old in [('all_p95','decision_p95_s'),('all_p99','decision_p99_s'),('search_p95','n1_decision_p95_s'),('search_p99','n1_decision_p99_s')]:
            val=number(r.get(key,r.get(old)))
            if val is None:
                vals=[number(x.get(key,x.get(old))) for x in rr]
                vals=[v for v in vals if v is not None]
                if vals:val=mean(vals)
            if val is None:
                vals=[number(x.get(key)) for x in latency[(f'GRID{stage:02}',r['date'],r['method'])]]
                vals=[v for v in vals if v is not None]
                if vals:val=mean(vals)
            d[key]=val
        method=r['method']
        if phase=='order':base='FULL__'+r['order_mode']
        elif phase in ['parameters','sealed']:base='FULL'
        elif stage==87:base=''
        elif stage in [82,84,86,88] or phase=='meta':base='FRAME_GUARD'
        else:base='FULL'
        costdelta=number(r.get('cost_delta_full')) if truth(r['complete']) else None
        d.update({'batch':batch,'stage':stage,'phase':phase,'date':r['date'],'method':method,'repeats':len(rr),
            'steps':next(iter(steps)),'quality':int(truth(r['quality_pass'])),'complete':int(truth(r['complete'])),
            'cost':number(rr[0].get('cost')),'cost_delta':costdelta,'queries':int(r['queries']),
            'controller':float(r['controller_s']),'forecast':number(r.get('forecast_s')),
            'optimizer':number(r.get('optimizer_s')),'search_other':number(r.get('search_nonforecast_s',r.get('n1_nonforecast_s'))),
            'other':number(r.get('remainder_s')),'min':min(ts),'max':max(ts),'sd':stdev(ts) if len(ts)>1 else None,
            'cv':stdev(ts)/mean(ts) if len(ts)>1 else None,'baseline':base,'source':str(pd).replace('\\','/')})
        dates.append(d)
dates.sort(key=lambda d:(d['stage'],d['phase'],d['method'],d['date']))
for r in dates:
    if not any(g['batch']==r['batch'] and g['method']==r['method'] for g in groups):
        groups.append({'batch':r['batch'],'method':r['method'],'baseline':r['baseline'],'source':r['source']})

dictionary=[
['范围','实验范围','覆盖GRID00–GRID89及GRID29B/29C/35B/40B/42B/55B，共96轮。其他课题FA/LG/IM不混入FRAME。'],
['版本','当前FRAME','当前中文V4的FRAME对应最终FRAME_GUARD，以GRID70实现为锚点。早期FRAME、GNN排序和上下文学习门控单独标识。'],
['场景','D组','原13个2035日期，开发与质量回归，已暴露。早期仅4/7日期的子集在各轮另标。'],
['场景','G1','主种子20261016生成，六个2036日历日期，同一公开电网。后续多轮复用，不能反复当新独立样本。'],
['场景','G2','主种子20261017生成，同电网、同六个日历日期，随机时序不同；方法固定后作为主评价。不是第二电网或六个新增独立日历日期。'],
['场景','旧八日期确认','GRID44与GRID52是不同八日期批次，使用的旧模型不同。GRID55/55B在已暴露日期重新配对，不能算又一次独立确认。'],
['质量','质量达标','当前框架比较要求终止不早于FULL；双方完整周时成本≤FULL的100.5%。0.5%是预设实验容差。早期各轮门槛不同，按原报告保留。'],
['质量','完成整周','达到2017条状态/步记录的整周结束条件；完整周数与质量通过数不同，若FULL提前结束，策略同一时刻结束也可能质量通过。'],
['质量','累计成本','环境原生成本账目，包含网损、调节、储能、弃电等；失败轨迹仍保留累计值，但不拿较低失败累计成本当完整周优势。'],
['质量','完整周成本变化','(本方法完整周成本/FULL完整周成本)−1。负值更低，正值更高；仅双方完整时可比。汇总平均/最大只取这些日期。'],
['质量','奖励遗憾','FULL最佳可用候选的预测奖励减本方法所选奖励。一步代理指标；与整周成本、实际安全保证不同。'],
['质量','同FULL动作','相同观测与基础动作上的选择一致数，分母是存在可用候选的状态数；不当作独立场景数。'],
['速度','查询','一次候选仿真。汇总每日期仅计一次，不把时间重复的相同轨迹重复相加。'],
['速度','完整控制时间','每次控制调用的耗时之和，含候选处理、预测、模型、连续优化、恢复、缓存等；初始化、物理环境推进、保存单列。'],
['速度','搜索耗时','预测仿真加搜索中的其他计算。汇总表通过预测与搜索其他相加；不同早期报告的搜索模块范围按原文。'],
['速度','P95/P99','控制步骤延迟的95/99分位数。全步骤与触发拓扑搜索的步骤分别统计，不能互换；秒为逐日期原单位，汇总转换毫秒。'],
['速度','重复标准差与CV','在同一日期同一方法的计时重复上计算。CV=样本标准差/均值。单次计时留空，不用0表示无波动。'],
['统计','日期宏平均减少','先在日期内平均计时重复，再计算1−本方法/参照，最后日期等权平均；不是总秒数比或查询总量比。'],
['统计','查询宏平均分母','仅参照实际搜索非零且运行时域相同的日期；零查询日期不设成0%参与平均，故总查询比不等于报告宏平均。'],
['统计','速度参照','质量速度汇总中明确列出同批参照。无同批FULL时，82/84/86/88及89-META以FRAME_GUARD作参照；87没有同批参照，速度比例留空。'],
['统计','公式方向','汇总表的减少比例方向为“本行方法相对参照”。会议主张2.228%是FRAME相对各META的均值，见贡献总表和GRID84原表；不得反过来使用。'],
['统计','速度资格','质量不足或时域不同仍列实际时间，但不能据此称全场景同质量提速。原冻结5%筛查不是显著性或等价检验。'],
['方法','FULL','本轮全部代表候选搜索，仍受基础控制器的动作库、可用性与组合规则约束；不是全动作空间或长期全局最优。'],
['方法','FIXED_256 / FIXED_512','只查询序列前min(cap,M)个代表；无额外反馈扩展，无可接纳动作时返回空给共同基础控制器。'],
['方法','NO_LOCAL','取消同站补查，保留该版本全局筛查/核查；版本随实验轮次注明。'],
['方法','GLOBAL_MATCH','保留同一版本全局过程，将同站补查替换为相同追加次数的全局续查。预算在同一状态一致，闭环分歧后整周查询总量可不同。'],
['方法','FIRST_SAFE_BEST','早期达到控制器负载阈值的停止参照，耗尽回退按最佳可用动作处理。是搜索策略适配，不是SoftGNN整套复现。'],
['方法','STATIC_ID','沿用作者动作资产教师频次顺序，过滤/等价合并保留相对顺序；并非任意随机编号。'],
['方法','META_DQN','GRID64/65停止限定接口的ICAPS2022内核适配；与后续META_CONTEXT区分。'],
['方法','META_CONTEXT / META_COST','GRID77后，25维公开上下文，返回/追加64/初始委托完整FRAME。ICAPS2022 DoubleDQN电网适配，不是A*/RRT*原任务完整复现。'],
['方法','META_COST0 / COST1','GRID89命名中COST0=费用0，COST1=费用0.0025；SEED0–4是同一方法的训练随机种子，不是五种算法。'],
['方法','CONTEXT_GATE','固定核查外增加普通电网上下文学习门控，用于跳过部分核查；最终固定FRAME不依赖该学习门控。'],
['方法','RANDOM_GATE0/1','与门控合法范围一致的随机跳过参照，概率按旧校准预设；不是随机候选排序。'],
['方法','LOCAL_GNN / COMPRESSED_GNN','旧GNN排序与软压缩评分版本，均有既定搜索规则；不是当前无需额外搜索训练的FRAME主体。'],
['方法','SOFTGNN','外部图软标签模型的电网任务适配，保留训练修订/种子；不是作者原电网任务完整复现。'],
['归因','公共缓存','DC、safeQP、public_prior元数据复用由各主对照共享。局部缓存节省不等于整套提速，也不计作GNN或FRAME独有贡献。'],
['证据','保存数据审计','核对向量、合法反馈、查询、成本与冻结规则。不是重新训练、独立AC重求解或新的独立评价场景。'],
['原表','原报告指标页','保留各报告的原列名、原值和出处；同一轮的pilot、RAW、修正版都标源文件，不能当额外独立实验合并。'],
['原表','源文冲突','GRID50报告叙述称种子1查询下降，但数值表为7530→7728。历轮摘要按数值表记录，并保留原文。']
]
payload={'as_of':'2026-10-19','history':history,'main':main,'raw_metrics':raw,'dates':dates,'groups':groups,
         'dictionary':dictionary,'source_receipts':sources}
(OUT/'experiment_tables.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_receipts.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'stages':len(history),'contribution_rows':len(main),'raw_report_metric_rows':len(raw),
                  'per_date_rows':len(dates),'method_batch_groups':len(groups),'source_files':len(sources)},ensure_ascii=False))
