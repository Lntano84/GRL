from pathlib import Path
import sys,json,shutil
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A=C.R,C.O,C.A;C.verify();assert C.read(O/'complete.json')['passed'] and C.read(O/'independent_audit.json')['passed'];handoff='''# GRID253后持续交接

先读docs RESEARCH_STATE/NEXT，以最新为准；有运行器跟进，完整项不重做。253仅来源/模型目录/完整计数资格完成，0新拟合/模型前向。原191/235结论已直接复用，不当新性能实验。当前2851拟合记录在原12天NN实际draw均>=14次，4681外层记录draw0，训练与部署停止点来源有明确差别，不称唯一失败原因。经验证21相关checkpoint文件(含全16副本/原12天外层模型)，无满足同时排除当前outer4和记录日期组的8天参考权重。

固定4日期组无序对6个x全3种子=18个参考NN，服务36有方向用途，reference_nodes/uses已证明两组标签日期排除。单个NN原7250updates历史中位73.41秒，18串行估1321秒不可用。完整单候选提案是18参考NN数据基础(6单线程worker、一次600秒)→公开参考选择/输入封存→同250类至多11线性+11保护步长(一次600秒)，全流程<=40拟合，0新森林头。两个组件有真实数据依赖，不是参数/候选搜索，计数不能隐去；各组件<=22、不延长。18NN仅离线生成训练停止点，不替换部署控制器。

下一GRID254只做零优化器执行资格，不训练：按fixed_pipeline_proposal和reference_nodes/uses准备6组fit8归一化/输入与18个固定采样流，原FACTOR12980参数/7250更新/batch32/Adam/SmoothL1/clip10完全保留，全3种子。核对初始hash、原fit8记录与当前outer排除、泛用原NNf隐藏提取接口、6worker隔离/实际缓存内存/严格全局600s截止及不重复完整job保护；源码和输入封存推送。当前32逻辑CPU/约15.2GiB RAM/可用约4.7GiB只是快照，6worker实际资源资格仍pending。失败保留错误/源，不减更新/换种子/续窗救预算；真需超授权才列具体决定。

后续单探针以保留的250(41参数)为主控制，别继续252新gap。训练STOP/CONT来自排除记录日期与outer的NN8；部署原NN12仍不变。canonical38+原父b+nativeSTOPbit含义保留，原父b必须接原f NN隐藏，不可把新NN8隐藏塞旧208头。原22653缓存字节复用，至多9585缺失原NNf隐藏行额外forward，24父头只复用0拟合。参考公开forward共32238行；选择/新输入先发再接原教师NET标签。新参考改变样本、按原两域各半mass*absNET公式重算的权重数值及own12全部selectedSTOP正例集合，不能再说85固定。fit8/cal4日期不变，f0_s1固定KEEP，其他池空域/无fitSTOP正例则预定KEEP，不挑日期/补样本。

父原NNf/208头仍own12含自身标签，内部cal4还可能进入其他记录参考NN祖先；只证明nativeSTOP参考源的记录日期排除和outer隔离，不称全链独立OOF/盲测。历史费用/收益尺度保持。保留230主候选、250低query两种子小改善、252小额恢复及所有反例，不拼策略择种子。

代码输入、18参考NN权重、公开参考选择/新输入、读出权重步长、全部外层预测按逻辑屏障先发后接相应标签。原费用/NET标签/全种子日期/控制器/排序别名/锁站/128/64/停滞2/.9不变；最终8不生成不读标签，不新物理/电网/论文/FRAME，不安装依赖。旧开发已暴露/前缀不独立，局部代理不等整周/实际时延，不把多查询质量改善冒充等预算，不设全赢5%门槛。Git严格等helper退出后下一helper/屏障，按AGENTS更新docs/归档首页，实际源码A归档、git_push.cmd推送，核对报告/D协作副本/清单/本地远端SHA。用户允许持续自主推进至喊停，更新grl起点，不恢复旧自动化、不归档聊天。
''';(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8');man=[]
for p in A.glob('*/archive_manifest.json'):
 for n,h in C.read(p).items():assert C.sha(p.parent/n)==h
 man.append(dict(path=p.relative_to(C.REPO).as_posix(),sha256=C.sha(p)))
copies=[]
for p in (R/'论文协作资料/FRAME_停止样本来源方案_GRID253').iterdir():assert C.sha(p)==C.sha(O/p.name);copies.append(dict(path=p.relative_to(R).as_posix(),sha256=C.sha(p)))
C.save('delivery_check.json',dict(passed=True,archives=man,collaboration=copies,no_new_training_orforward=True));d=A/'delivery';d.mkdir(exist_ok=False)
for p in [O/'delivery_check.json',O/'CONTINUOUS_HANDOFF.md',C.W/'delivery.py']:shutil.copy2(p,d/p.name)
(d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8');print('GRID253_DELIVERY_VERIFIED')
