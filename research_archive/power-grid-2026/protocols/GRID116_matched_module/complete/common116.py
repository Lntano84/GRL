"""Fixed GRID115 protocol implementation; saved inputs only."""
from pathlib import Path
import sys
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from common import *
from common import verify as verify113
sys.path.insert(0,str(R/'work/grid115_matched_module_plan'))
from matched_model import MatchedModule,OriginalBackbone
W=Path(__file__).parent;O=R/'outputs/grid116_matched_module';S=R/'outputs/grid113_relational_learning';P115=R/'outputs/grid115_matched_module_plan'
def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for n,h in read(O/name).items():assert sha(R/n)==h,n
    for n,h in read(P115/'input_manifest.json').items():assert sha(R/n)==h,n
    verify113();sealed()
def build(seed,arm):return MatchedModule(seed,arm.removeprefix('BASE_'))
