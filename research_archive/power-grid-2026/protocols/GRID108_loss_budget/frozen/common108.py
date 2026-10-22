"""Fixed loss x budget comparison; original data and architecture only."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid108_loss_budget'
REPO=ROOT/'work/GRL-results-integration';ARCHIVE=REPO/'research_archive/power-grid-2026/protocols/GRID108_loss_budget'
sys.path.insert(0,str(ROOT/'work/grid107_direct_value'))
from common107 import read,save,sha,sealed,loadcases,normalization,normalized,carrier,weight_digest,predict,state_metrics,summarize
def verify():
    for n in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
    sealed()
