"""Copy verified configuration materials and register the completed documentation unit."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/meta_unified_20261019'
WORK=Path(__file__).parent
REPO=ROOT/'work/GRL-results-integration'
COLLAB=ROOT/'论文协作资料/META_统一配置_20261019'
ARCHIVE=REPO/'research_archive/power-grid-2026/paper/META_unified_20261019'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def append_once(path,marker,text):
    old=path.read_bytes()
    if marker.encode() not in old:path.write_bytes(old+b'\n'+text.strip().encode('utf-8')+b'\n')


def main():
    check=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    assert check['passed'] and check['new_closed_loop_runs']==check['new_training']==0
    sources=json.loads((OUT/'source_receipts.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    README='''# META统一配置资料入口

先读 `META_统一配置说明.md`。本资料把现有META的训练配方、准确权重、执行接口及评价批次逐项对应。

- `configuration.json`：统一25维、三动作学习方法，显式绑定C0/C0025各五种子。
- `results_by_configuration.json`：138条历史运行和逐配置汇总，批次分别保留。
- `configuration_issues.json`：历史标签继承、费用索引和重复次数的歧义及处理。
- `behavior.json`：逐配置FRAME委托行为。
- `validation.json`：10权重、138运行、612搜索调用通过，20个绑定与原冻结匹配。
- `source_receipts.json`：原始证据SHA256。
- `source/`：配置构建、统一调用入口及归档脚本。

本次新训练0、新闭环0、新仿真0。原校准质量匹配False保留。主结果2.23%仍仅对应GRID84费用0、五种子和两次反向计时；GRID89费用复查是一次计时，不能合并。历史文稿未覆盖。

可执行工作区为 `D:/Codex/GRL研究`。统一入口实际位于 `work/meta_unified_20261019/runtime.py`，调用者提供公共observe接口；这里只存源码副本。依赖原冻结策略和已有本地权重，完整路径/哈希见配置。未在此重复打包大型轨迹、环境和模型。
'''
    (OUT/'README.md').write_text(README,encoding='utf-8',newline='\n')
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='delivery_receipt.json']
    for target in [COLLAB,ARCHIVE]:
        target.mkdir(parents=True,exist_ok=True)
        for path in files:shutil.copy2(path,target/path.name)
        (target/'source').mkdir(exist_ok=True)
        for path in WORK.glob('*.py'):shutil.copy2(path,target/'source'/path.name)
    manifest={p.relative_to(ARCHIVE).as_posix():sha(p) for p in ARCHIVE.rglob('*') if p.is_file()}
    assert all(sha(COLLAB/p)==h for p,h in manifest.items())
    receipt=dict(source_output=OUT.as_posix(),collaboration=COLLAB.as_posix(),archive=ARCHIVE.as_posix(),
                 files=manifest,all_archive_collaboration_bytes_equal=True)
    (OUT/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    marker='META_UNIFIED_20261019'
    append_once(REPO/'docs/DECISIONS.md',marker,'''
## 2026-10-19 — META_UNIFIED_20261019

用户要求先统一META。统一接口META_CONTEXT_V1保留25维、三动作与完整FRAME委托，C0/C0025准确绑定全部五种子和权重哈希。主结果沿用预先校准选定的C0/GRID84两重复；GRID89一重复费用敏感性单列，不根据评价结果重新选费用或拼计时。发现正费用训练设计继承旧阶段/零费用/两重复标签，以实际训练脚本、收据和评价设计澄清，新清单拆分三层；旧冻结文件不改。原calibration_quality_matched=false保留。不把旧12维两动作停止型混入当前均值。仅配置和保存反馈执行核对，无新训练/仿真。
''')
    append_once(REPO/'docs/EXPERIMENT_LOG.md',marker,'''
## 2026-10-19 — META_UNIFIED_20261019 配置一致性核对

核对10权重SHA256、138历史运行元数据、20项原冻结绑定；统一显式入口重执行612次已保存搜索调用、241794条反馈，查询顺序/选择/停止/25维状态/学习动作与原日志一致，129/22日期家族划分和原主结果算术一致。新增训练/仿真/闭环均0。材料：research_archive/power-grid-2026/paper/META_unified_20261019；中文说明、机器配置、逐配置结果、问题记录与源码齐全。
''')
    state=REPO/'docs/RESEARCH_STATE.md'
    text=state.read_text(encoding='utf-8')
    if marker not in text:
        first,body=text.split('\n',1)
        state.write_text(first+'\n\n- **2026-10-19 META_UNIFIED_20261019：** 用户要求先统一META。25维三动作内核、两费用各五种子与准确权重显式绑定；主比较C0/GRID84两重复、费用补充GRID89一重复分别保留。10模型、138运行和612次保存调用一致性检查通过。2.23%及原校准质量不匹配不改，无新训练/仿真。入口[统一说明](../research_archive/power-grid-2026/paper/META_unified_20261019/META_统一配置说明.md)。\n'+body,encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps

1. 用户审阅META统一配置说明：META_CONTEXT_V1，C0主比较、C0025费用补充；旧校准不匹配和批次区别保留。
2. 根据确认的命名、准确权重与结果出处修订论文META段落；当前V4及用户修改稿保留。
3. META口径确定后，再核查一个外部结构启发式ExpertOp4Grid的环境和接口适配条件；当前不启动新训练或闭环。
4. 比例预算与另一电网的迁移作为后续研究建议，尚未设定或启动新协议。配图和英文排版按作者后续指示开展。
''',encoding='utf-8')
    append_once(REPO/'docs/PAPER_MATERIALS_INDEX.md',marker,'''
## META_UNIFIED_20261019

- [META统一配置与结果说明](../research_archive/power-grid-2026/paper/META_unified_20261019/META_统一配置说明.md)：当前学习适配的25维/三动作、费用、五种子、重复次数与结果一一对应；含机器配置和保存调用复核记录。
- D盘协作资料：`D:/Codex/GRL研究/论文协作资料/META_统一配置_20261019`。未覆盖论文V4。
''')
    append_once(REPO/'research_archive/power-grid-2026/README.md',marker,'''
## META_UNIFIED_20261019

[META配置资料](paper/META_unified_20261019/README.md)：统一已有学习方法的特征/动作、准确权重、费用和结果批次；修正新说明中的历史标签歧义，保留原冻结文件和全部较差结果。无新增训练、仿真或闭环。
''')
    print(json.dumps(dict(files=len(manifest),copies_byte_identical=True,archive=ARCHIVE.as_posix()),ensure_ascii=False))


if __name__=='__main__':main()
