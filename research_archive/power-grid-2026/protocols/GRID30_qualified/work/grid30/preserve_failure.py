"""Preserve failed replica setter before changing the qualification harness."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid30';DEST=OUT/'revisions/v0_parameter_setter'
assert not (OUT/'qualification.json').exists();DEST.mkdir(parents=True,exist_ok=False)
files=([ROOT/'work/grid30/qualify.py',ROOT/'work/grid30/cache_safe.py',ROOT/'work/grid30/prepare.py']+
    [OUT/n for n in ['design.json','code_freeze.json','qualify.log','manual.json','parameter_copy_diagnostic.json']])
for p in files:
    shutil.copy2(p,DEST/p.name)
source=(OUT/'runs').resolve();target=(DEST/'runs').resolve();workspace=OUT.resolve()
assert source.is_relative_to(workspace) and target.is_relative_to(workspace)
shutil.move(str(source),str(target))
(DEST/'FAILURE.md').write_text('''# GRID30 V0影子参数setter失败

8项手工检查通过；控制335步后、首次恢复QP之前，影子复制gen_per_bus触发非负setter验证。保存真实公开状态再构建原参数，发现两处负值，最小-0.0006708863656967878。作者更新数组值时原本不走setter；不能把这次拷贝失败读成数值缓存或控制器质量失败。

V1仅改为与作者相同的数组原位值复制，并逐值断言，不裁剪或归零；系数/数学问题/实际控制路径不改。保留全部旧源码/协议/日志/335步片段；未得到有效影子数值/动作结果。
''',encoding='utf-8')
print(json.dumps({'preserved':str(DEST.relative_to(ROOT)),'physical_partial_steps':335}))
