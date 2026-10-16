"""Chinese method/pseudocode/evidence notes; keep all user drafts unchanged."""
from pathlib import Path
import csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid82_method'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    assert not OUT.exists();OUT.mkdir()
    R=read(ROOT/'outputs/grid81/review.json');G=[x for x in R['groups'] if x['method']=='FRAME_GUARD'];F=[x for x in R['groups'] if x['method']=='FULL']
    pairs=[(g,next(f for f in F if f['date']==g['date'])) for g in G]
    own_rows=[x for x in R['rows'] if x['method']=='FRAME_GUARD'];full_rows=[x for x in R['rows'] if x['method']=='FULL']
    macro=statistics.mean(1-g['controller_s']/f['controller_s'] for g,f in pairs)
    q=statistics.mean(1-g['queries']/f['queries'] for g,f in pairs if f['queries']>0)
    maxcost=max(g['cost_delta_full'] for g in G if g['cost_delta_full'] is not None)
    rows=[
      dict(claim='分阶段搜索框架具有实测计算收益',evidence='GRID81 FRAME_GUARD与FULL同批两次计时',result=f'质量6/6，完成4/6；活跃同终止长度场景查询宏平均减少{100*q:.3f}%，控制时间宏平均减少{100*macro:.3f}%，最大完整周成本增幅{100*maxcost:.4f}%',paper_role='主结果',boundary='同一公开电网六个新生成场景；质量为预设经验条件'),
      dict(claim='缓存复用避免重复候选预测',evidence='生产代码GRID70缓存委托；GRID81逐调用合法反馈审计',result='每次搜索每个代表最多实际查询一次；委托不重复支付前缀',paper_role='方法性质与可复现接口',boundary='缓存和去重是实现机制，不单独宣称算法首创'),
      dict(claim='初始较安全也可能需要继续核查',evidence='GRID68失效诊断、GRID70修复、GRID76十三日期回归',result='保存旧失败和固定核查修复；不将rho<0.9解释为长期安全保证',paper_role='动机案例与分支设计依据',boundary='修复开发证据，不算独立确认'),
      dict(claim='同站补查应当保留为组成模块',evidence='GRID63/65等每状态额外查询预算的GLOBAL_MATCH',result='现有结果未建立结构选择相对等预算全局扩展的稳定独立优势',paper_role='消融及方法组件',boundary='不能把全部查询节省归因于结构补查；旧版本消融不冒充当前完整版本'),
      dict(claim='学习门控具有独立主要贡献',evidence='GRID81当前门控、始终核查及随机门控0/1',result='未通过两个随机种子的识别增量门槛；相对始终核查控制时间宏平均-0.040%',paper_role='扩展消融，主方法不依赖该门控',boundary='不声称等价、不推广为所有学习门控无价值'),
      dict(claim='最终版本优于既有会议策略',evidence='GRID79旧门控版比较；GRID82最终固定核查版同批36条待运行',result='当前尚未由最终版本的该组比较裁决',paper_role='GRID82完成后更新主对照表',boundary='ICAPS2022算法内核的电网适配；原共同校准质量匹配失败保留'),
      dict(claim='减少搜索查询对应完整控制器提速',evidence='GRID79互斥时间分解；GRID81实际控制时间',result='查询、预测墙钟、优化器墙钟、总控制时间分别报告',paper_role='计算机制分析',boundary='计量分解不等于已识别每个模块的独立因果效应；P95不宣称明显改善'),
      dict(claim='提出了普适的新停止理论或首次使用GNN+RL',evidence='现有研究与历史实验',result='不采用此贡献表述',paper_role='贡献范围管理',boundary='框架差异和应用效果仍需与具体已有工作逐项讨论')]
    with (OUT/'贡献与证据对应表.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    text=r'''# FRAME最终方法说明：分阶段核查与预测反馈复用

## 论文一句话主张

FRAME将电网拓扑候选的预测评估组织为前缀查询、反馈核查和必要补查三个阶段，在保留控制器动作接口的同时减少预测调用，并将查询节省转化为完整控制器的实际提速。

主方法对应冻结代码 **FRAME_GUARD（GRID70）**。当前上下文森林门控保留为扩展消融。这里收敛的是搜索预算控制模块，底层LJN控制器及其既有策略、连续优化和恢复控制保持原状。本文没有新增训练。

## 1. 问题与输入输出

控制器在当前状态s形成基础动作b，并提供一个合法拓扑候选库。搜索模块为这些候选购买一步公开预测，输出附加拓扑动作；没有可接纳候选时返回空，由既有控制器继续处理。

输入包括当前可见状态、基础动作、已去重代表集合、固定候选顺序以及原始动作到代表的别名映射。当前正式比较沿代表ID升序（STATIC_ID），各臂完全相同，不宣称本方法改进了候选排序。

每个代表a实际预测的是b+a。预测反馈包括奖励、最大负载率rho、结束标记、非法/歧义/异常标记。可接纳候选需满足：

\[
0<\rho(s,b+a)<\ell(s),\quad\neg\mathrm{ended},\quad\neg\mathrm{illegal},\quad\neg\mathrm{ambiguous},\quad\neg\mathrm{exception}.
\]

其中ell取当前状态最大负载率和调用方提供阈值中的较小者。初始较安全分支使用底层控制器已有阈值rho_safe=0.9。这个条件用于选择搜索分支，长期运行质量另由闭环评价。

统一排序键为K(a)=(float32(R(a)),-ID(a))，按字典序取最大；这与原控制器的奖励精度和破并列方式一致。E表示已查询代表集合，a*表示E中按K最优的可接纳候选。

## 2. 三个模块

### 前缀查询

沿公共顺序查询前min(128,M)个代表，缓存每份公开反馈，并得到初始候选a0。模块输出为E、反馈缓存和a0。无论后面进入哪个分支，这些预测都直接复用。

### 反馈核查

如果a0存在且rho(a0)<0.9，进入核查分支：每次购买64个未查询候选；累计至少256次查询、连续两批未改善统一排序键且仍有候选未查时，停止全局扩展；否则继续到候选耗尽。

如果初始条件不满足，进入扩展分支：同样按64个候选购买反馈；当本批后出现rho<0.9的可接纳候选时可以停止；否则在已有可接纳候选、累计至少256次查询、连续两批未改善且仍有候选未查时停止。没有可接纳候选时继续到耗尽。

这两个分支的停止条件不同，伪代码分别给出。128、64、256、两批和0.9均沿用冻结设置，不在本轮重新搜索参数。

### 必要补查与交付

初始较安全的核查分支只在核查改变初始候选时补查；初始不够安全的扩展分支在有可接纳候选、仍有未查询代表时补查。补查对象为当前候选所在变电站的剩余代表。

设s*=sub(a*)，pi将原始动作映射到等价代表，则：

\[
\mathcal L(s^*)=\{\pi(a):\operatorname{sub}(a)=s^*\}\setminus E.
\]

集合从原始动作别名计算，以保留去重后的结构归属；补查沿同一公共顺序执行。补查结束后交付已查询集合中按K最优的可接纳动作。

## 3. 完整伪代码

```text
输入：代表顺序 O，代表数 M，原始动作别名 pi，变电站归属 sub
固定：p=128，h=64，q_min=256，patience=2，rho_safe=0.9
E ← 空集；缓存 ← 空；a* ← 空

过程 QUERY(S)：
    保存查询前的最佳排序键
    对 O 顺序中的每个 a∈S：
        若 a 不在缓存中：实际预测 b+a，保存反馈并将 a 加入 E
        若 a 可接纳且 K(a) 更好：更新 a*
    返回最佳排序键是否改善

QUERY(O 的前 min(p,M) 项)；a0 ← a*
initial_safe ← (a0存在 且 rho(a0)<rho_safe)
stale ← 0

当还有未查询代表时：
    improved ← QUERY(下一批至多 h 个代表)
    stale ← 0（improved时），否则 stale+1
    若非 initial_safe 且 a*存在 且 rho(a*)<rho_safe：跳出
    若 a*存在 且 |E|≥q_min 且 stale≥patience 且仍有未查询代表：跳出

need_local ← a*存在 且 |E|<M 且 (非initial_safe 或 a*≠a0)
若 need_local：
    s* ← sub(a*)
    L ← {pi(a):sub(a)=s*} \ E
    QUERY(L)
返回 a*（不存在时返回空）
```

QUERY在每个当前物理状态重新建立缓存；缓存不跨状态复用预测结果。原控制器已有的DC、安全QP和等价元数据缓存另列为各臂共享底座。

## 4. 可直接从过程得到的性质

**单次购买。** 每个代表只在不属于E时实际预测，因此一次搜索的付费预测数不超过M，分支委托不重复支付前缀。

**已查询集合内选优。** a*始终为E中按统一排序键最好的可接纳候选；增加查询不能降低这个排序键。性质只涉及已查询集合与该排序，不承诺找到全库最优动作。

**最坏计算量。** 最坏仍需查询全部M个代表，改进体现在实测查询数和墙钟，不声称降低了穷举的最坏阶数。整体预算还包括基础连续优化、恢复控制和管理开销。

## 5. 贡献如何组织

**贡献一：面向混合电网控制的分阶段核查框架。** 将前缀、反馈核查、必要补查及预测复用衔接为可直接接入控制器的搜索模块。正文写清分支、信息接口和动作交付，解释计算如何被组织。

**贡献二：查询压缩到完整控制提速的闭环证据。** 在相同电网、控制底座、排序和资源计量下，同时评价存活、完整周成本、候选预测数和完整控制时间，并通过消融、外部方法适配和时间分解回答具体问题。

同站补查作为组成模块介绍，其稳定独立优势尚未建立。学习门控放在扩展消融。公共缓存、已有神经控制器和通用停止思想注明来源。论文的新意仍要通过与既有工作的具体差异论证，不能用实验数量代替。

## 6. 已有证据

'''
    text+=f'GRID81中，固定核查版质量条件6/6通过，完成4/6，与FULL相同；完整周成本最大增幅{maxcost*100:.4f}%。活跃且相同时域场景的查询宏平均减少{q*100:.3f}%，完整控制时间按六场景宏平均减少{macro*100:.3f}%。控制时间日期均值为{statistics.mean(g["controller_s"] for g in G):.3f}秒，FULL为{statistics.mean(f["controller_s"] for f in F):.3f}秒。\n\n'
    text+='| 主张 | 对应证据 | 用法 |\n|---|---|---|\n'
    for row in rows:text+=f'|{row["claim"]}|{row["evidence"]}；{row["result"]}|{row["paper_role"]}|\n'
    text+='''
## 7. 本轮外部比较的边界与问题

GRID82只比较固定核查版与Bhatia等ICAPS2022方法内核的既有电网适配，两种冻结模型种子全部保留。沿用同一候选顺序、公共反馈和完整核查委托权限。原场景为anytime planning，本地实现更换了任务状态与动作，因此称为方法适配，不称原任务SOTA完整复现。

GRID77已记录两种子未同时匹配固定核查校准质量；该事实随比较一起保留。如果会议适配没有通过质量条件，结果按质量和计算分别说明，不以失败对手的短轨迹直接宣称同质量加速。

六个2036生成场景已在GRID81暴露；GRID82属于固定版本同场景比较，不新增独立确认。质量参照使用已冻结的GRID81 FULL，新的实际计时只比较本轮同批三臂。无新增训练、调参或日期筛选。

## 8. 与用户稿件的衔接

本文件是最终方法与证据材料，不覆盖手写稿、V4/V5。后续中文正文应将摘要、方法图、算法和主结果统一到固定核查版本；历史门控版的数字单独标明版本。

按留存指南中的“先概览与输入输出，再逐模块解释，最后给伪代码”的组织方式编写。Methods原网页本轮重新访问失败，内置浏览器初始化也失败；本轮使用此前保存的逐节审核说明，不声称重新通读网页。没有重画配图或改写正式投稿版。
'''
    (OUT/'最终方法_伪代码与证据对应.md').write_text(text,encoding='utf-8')
    sources=['work/grid70/policies.py','work/grid63/policies.py','work/grid60/policies.py','work/grid27/policy.py','outputs/grid81/review.json','outputs/grid77/selected.json','outputs/grid65/review.json']
    (OUT/'source_receipt.json').write_text(json.dumps(dict(sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},derived=dict(guard_macro_time_reduction=macro,guard_macro_query_reduction=q,guard_max_complete_cost_delta=maxcost),new_fits=0,user_drafts_modified=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(method='FRAME_GUARD',macro_time=macro,macro_queries=q,max_cost=maxcost)))
if __name__=='__main__':main()
