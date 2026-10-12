"""Derive matched original/soft-compressed comparison;no other experiment changes."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
for name in ['prepare.py','audit_pilot.py']:
    s=(ROOT/'work/grid49'/name).read_text(encoding='utf-8')
    s=s.replace('outputs/grid49','outputs/grid51').replace('outputs/grid48','outputs/grid50').replace('work/grid48','work/grid50')
    s=s.replace('FREE_GNN','COMPRESSED_GNN').replace('FREEGNN','COMPRESSEDGNN').replace('free_residual.py','compressed_residual.py')
    s=s.replace('GRID49','GRID51').replace('FREE_RESIDUAL','SOFT_COMPRESSION').replace('free residual','soft-compressed residual').replace('free-residual','soft-compressed-residual')
    s=s.replace('Free residual','Soft-compressed residual')
    assert not (HERE/name).exists();(HERE/name).write_text(s,encoding='utf-8')
s=(ROOT/'work/grid_research_audit/archive_grid49.py').read_text(encoding='utf-8')
s=s.replace('outputs/grid49','outputs/grid51').replace('work/grid49','work/grid51').replace('work/grid48','work/grid50')
s=s.replace('FREE_GNN','COMPRESSED_GNN').replace('FREEGNN','COMPRESSEDGNN').replace('free_residual.py','compressed_residual.py')
s=s.replace('GRID49','GRID51').replace('free-residual','soft-compressed-residual').replace('free residual','soft-compressed residual')
s=s.replace('新无硬界残差两种子','新软压缩残差两种子').replace('2026-10-12','2026-10-12')
p=ROOT/'work/grid_research_audit/archive_grid51.py';assert not p.exists();p.write_text(s,encoding='utf-8')
print('Derived5oldMarchpilot/15conditionaldevelopment,originalfaststop,2compressedscorerseeds;no GRID44reuse.')
