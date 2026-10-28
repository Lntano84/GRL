# GRL持续交接：GRID235资格完成，下一步GRID236公共先验再生成

先读integration/docs/RESEARCH_STATE.md和NEXT_STEPS.md，以最新状态为准。用户允许自主训练推进直到喊停；阶段实质发现简短汇报，无变化安静。当前无持续运行器，完成拟合/评分/审计/归档不重做。

233/234完成：PUBLIC在14条变化回本记录的原始方向全部更温和，7/8恢复方向主导、6/6误停步长主导；全182仅87方向改善，不能泛化。230仍主内部候选，233保留Jan19/Jun1恢复线索。训练两保护模型STOP分歧871/正例0是规则内生，不能据此直接训练选择器。

235仅资格，拟合0/新预测0。36原outer4与全部标签祖先无交集，原外层没有被用于训练；但若再将原outer12切inner8/4，3474条旧混合fit8行的先验头标签祖先包含拟作cal4的全部4天，旧STOP cal4有1644行被原修正器直接拟合。baseNET/STOP/隐藏表示仍outer12；父222预算也outer12。不是原外层泄漏，而是内部分组不能被当独立校准。固定所有池inner_fold0，没按支持或结果挑fold。

已资格唯一条件层诊断，总操作上限公开：复用24个现有inner0森林头(12event+12amount_depth4)，0森林/NN/物理新拟合，后续最多11个43参数线性修正器，0标量拟合，单次600秒模型窗口不延长。先收集4天未参与修正器拟合的错误，不拟合新alpha/二级gate，也不做新outer4策略比较。上游冻结outer12，所以只能称固定上游条件下修正层留组诊断，不能称全链嵌套、独立OOF或盲测。新先验、筛选、scalers和fit8有变，更不能当与233只差校准的消融。

下一步236新建代码，消费outputs/grid235_calibration_ancestry/fixed_probe_pools.json、qualification/source_freeze，先封存源码及输入推送。只公共再生成：读取grid192_gate_support/fit_features/{pool}.npz的x(83维rawcache)、ids和original_STOP_count，不读取y；读取grid214/public_raw41.npz；每池选定grid213/qualified_inputs/{bundle}.npz的mean83/scale83/mean41/scale41，构建124维头输入。复用grid214/models/{bundle}_event.pkl及_amount_depth4.pkl(源码真实命名)，同一对fit8头对全部ownouter12行生成p和mu4，charge=raw83[:,82]，b=(p*mu4-charge)/(p*mu4+charge)。不混用其他inner头。额外修正器42输入为标准化public41+b(使用同一个inner8 mean41/scale41)，所有12天同源；role由固定fit8/cal4日期给定，outer4不生成新分数。

公共导出字段应含metadata ids、fit/cal role、42输入、b、knowncharge、headCONT(b>0)、原STOP mask、来源hash等，不能含NET/gain/y。STOP mask由192原STOP前缀计数/213资格索引核对，避免读标签求mask。已有214/predictions的cal4公开p/mu4/charge可以逐项重现；或用213 score_X_sha256核对同矩阵。既有森林权重已经发布，先验证资格hash/发布收据。全部236公共预测先保存归档推送，再单独连接214既有连续actual_NET/原source_mass标签，不重新仿真；记录访问字段。不要在公开预测阶段加载casepickle或scoretargets。

236预测发布后再做支持和训练输入资格：用新b>0重新确定STOP/辅助CONT资格，不能沿用224旧筛选或旧b。fit8与cal4标签分文件，237训练worker只读fit8标签；只在fit8各域分别归一化原结构质量×absNET风险半半(沿用224revision02原则)，不能用含cal4的旧整体风险权重。新head选择后若fit8 STOP正例0、或任一训练域为空，固定KEEP，不换fold/补样本；f0_s1 KEEP保留。先公布每池资格与最多模型数<=11，再开始237，不根据cal4效果改变方案。

后续237最多11个新43参残差线性模型，offset同源b，原mean logistic+全43单位L2、原L-BFGS-B maxiter2000/gtol1e-6/ftol1e-12/maxls50，零43初始化，600秒一次不延长；0alpha/预算校准。所有模型数值资格成功且43维rawgradinf<.001后发布权重，4天公共分数发布后才连4天标签，仅分析层错误/正负支持，不做组合策略选择。旧233/230仅历史上下文，原222预算不继承；不能将模型训练8vs12等变化的结果包装成竞争力提升。

235报告outputs/grid235_calibration_ancestry/GRID235_report.md，D协作论文协作资料/FRAME_校准数据来源_GRID235。当前资格清单只承认条件层探针，不承认全链独立。归档从源码实际A取路径，按AGENTS更新状态/决定/日志/NEXT/首页、协作hash及git_push.cmd远端核对。Git helper返回session_id必须poll到exit_code后才下一helper/发布屏障。233发布等待及225/226/229历史错误已保留，非当前故障。

原费用标签/全原日期种子/控制器/排序别名/锁站/128/64/停滞2/.9不变。最终8不生成不读标签，不新物理/电网/论文/FRAME，不装依赖。所有数据已暴露，局部代理不等整周成本或实际时延；小改善保留，不设全赢/5%新门槛。权限不足仅申请D工作区写和现有Git网络，拒绝不绕过。用户喊停则停新增并暂停grl，不恢复旧自动化、不归档聊天。
