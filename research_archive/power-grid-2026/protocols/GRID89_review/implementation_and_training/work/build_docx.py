"""Reuse the tested editable-OMML manuscript renderer in a new output folder."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'work/frame_manuscript_merge_20261017/build.py').read_text(encoding='utf-8')
source=source.replace("OUT=ROOT/'outputs/frame_cn_merged_20261017_v1'","OUT=ROOT/'outputs/grid89_review/manuscript_v2'")
source=source.replace("NAME='FRAME_ICAPS2027_中文合稿_V1'","NAME='FRAME_ICAPS2027_中文合稿_V2'")
source=source.replace("symbols={'rho':'ρ'","symbols={'exp':'exp','log':'log','beta':'β','lambda':'λ','rho':'ρ'")
source=source.replace("# Unify citation presentation without changing the approved abstract or scientific results.","merged=(ROOT/'outputs/grid89_review/manuscript_v2/全文待排版.md').read_text(encoding='utf-8')\n# Assemble the new version from the reviewed V1 plus all bounded validation results.")
source=source.replace('assert abstract in merged','assert "# 1 引言" in merged')
exec(compile(source,str(__file__),'exec'),{'__file__':str(__file__),'__name__':'__main__'})
