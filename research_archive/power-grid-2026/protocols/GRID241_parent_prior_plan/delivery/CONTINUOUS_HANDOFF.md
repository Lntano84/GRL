# GRL持续交接：GRID240/241完成，下一步GRID242父先验一致公共输入

先读integration/docs/RESEARCH_STATE.md、NEXT_STEPS.md，以最新状态为准。用户授权自主训练推进直到喊停，不逐轮确认；阶段实质发现简短汇报，无变化安静。当前无持续运行器，完成拟合/评分/审计/归档不重复。

239成对校准未改善平均NET，但种子0/1相对共同GATE保持同即时增益/动作一致且省查询，小改善保留，不挑种子。240只读610原CONT、7共同漏查和4681入口分歧完成：Jan5三相关大损失b8 .037-.047，原208b12 .397-.531，方向约-1，两alpha均误停；3状态占宏平均回本损失95.60%。共漏5/7 b8较低、6/7低于校准正例最小b，却全部在拟合正例b范围内；69条有效无利取消也低于cal范围，最近邻距离也不支持粗暴域外解释。Jun1/Nov16共漏b8反而更强，不能把一因说成全部原因。

原入口仍是208十二天头的b12>0，修正器却把另一八天头的b8作输入/偏置，接口语义并非同源。4681既有NETSTOP中479两者继续、131父继续/inner停、680父停/inner继续、3391两者停。131原规则域外保留不等学出。240没试新策略或替b，241据该结构问题只做父先验一致资格，0新预测/拟合。

241核对24原208 event/amount头的原发布权重、12parent scaler及祖先/入口符号一致。真实旧发布收据是outputs/grid208_public_amount/preheld_publication.json，非pre_score_publication；源码A=GRID208_public_amount。两头均124输入/64树/depth4，parent modelnpz mean/scale83、raw_mean/raw_scale41。全部自身outer4不在标签祖先。不要重训或改原头。

固定唯一新联合候选，详细读241design.json/qualified_pools.json：训练/检查/部署都沿用同一个原208父b12，不是把旧237theta输入b直接替换。父头用原208自己的124维标准化；额外修正器公共41仍使用既有213inner8的mean41/scale41(保持237这部分)，拼b12=42输入43参数、offsetb12。用父b12>0重新筛STOP/辅助CONT，不沿用旧b8筛选。修正器只fit8，风险各域独立归一化结构质量×absNET后半半，原meanNLL+全43单位L2。原221单步长保护自身全部outer12已选STOP回本正例(fit8+cal4)，这是显式联合经验保护变更，不是239的第三臂或纯先验单因素消融，不称cal4独立。总<=11线性+11标量=22、600秒一次不延长，f0_s1或fit8STOP正例0或任一训练域空就预定KEEP。无父预算/新alpha网格/阈值/种子选择。

下一步242只公共再生成及支持资格，0新模型/标量拟合。先封存源码输入推送；消费241qualified_pools/qualification/source_freeze。读192fit_features的raw83 x/ids/原STOP计数、214public_raw41，按原208parent modelnpz标准化为124头输入，固定eventp*amountmu和raw83[:,82] charge生成b12。额外X42用固定inner8 public41标准化加b12。角色仍235固定fit8/cal4，不选新fold。先保存所有父公共预测/资格/来源并推送，再接214既有actual_NET与source_mass标签；0NN/物理。若要核对已发表outerb12，只复现父公共公式，不生成新修正策略分数，不读病例结果。全部新候选输入资格公布后再开始243。

242标签连接后分别准备fit8训练文件和outer12全部已选STOP正例校准文件。fit8风险只能用fit8归一化，cal4标签仅进入已声明联合保护，不进入theta拟合。资格真实拟合池<=11，KEEP规则按计划不换日期/加数据。243另阶段一次<=22拟合，原L-BFGS-B maxiter2000/gtol1e-6/ftol1e-12/maxls50、零43初始化、rawgradinf<.001。全部theta/alpha先发布后外层候选分数，全部分数先发后连既有结果。

未来部署与外层离线评分直接复用原208已计算b12及220public_held_inputs.base_CONT；不再有inner8额外收益头推理或第二支持域。只取消原208继续、最多同一批，不救原停止动作。实际额外public41提取和线性推理未计时，不能称实际控制提速。主要比较共享GATE与新联合候选的质量/查询折中，230历史参考而非纯先验消融；所有旧暴露日期/种子/反例保留，不许给Jan5例外。

报告240/GRID240_report.md和241/GRID241_plan.md。240/241原归档2清单和11协作副本已核对，源码实际A归档不猜别名。当前没有新候选效果。父模型/NET/STOP已outer12，内部cal4不是全链独立OOF/盲测；原outer4标签祖先隔离仍保持。最终8不生成不读标签，不新物理/电网/论文/FRAME，不装依赖。费用标签/控制器/排序别名/锁站/128/64/停滞2/.9不变。

按AGENTS更新docs状态/决定/日志/NEXT/首页、完整归档/D协作hash、git_push.cmd推送远端核对。Githelper返回session必须poll到exit_code后才下一helper/屏障。旧错误和更正非当前故障不重做。权限不足仅申请D工作区写和现有Git网络，拒绝不绕过。小而重复改善保留，不设全赢/5%门槛，不把多查询质量改善当等预算优势，代理不等整周成本/实际时延。用户喊停则停新增并暂停grl，不恢复旧自动化、不归档聊天。
