"""Frozen BC03 protocol. No model fitting or parameter search."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
WORK = BASE / 'work' / 'bc03'
REPO = BASE / 'work' / 'bc01' / 'Baleen-FAST24'
PYTHON = BASE / 'work' / 'bc01' / 'python311-embed' / 'python.exe'
THETA0 = 0.798545
THETAL = THETA0 / (2 - THETA0)
THETAH = 2 * THETA0 / (1 + THETA0)
T = 346810.0965330601
START = 1572074461.57806
EVAL_START = 86401.23277902603
EVAL_INDEX = 144
WRITE_BUDGET = 148731
ARMS = ['STATIC-BASE', 'STATIC-LOW', 'STATIC-HIGH', 'LEAD', 'DURING', 'PREV-DAY', 'NEXT-DAY']
INTERVALS = {'LEAD': (T-7200, T), 'DURING': (T, T+7200),
             'PREV-DAY': (T-86400-7200, T-86400), 'NEXT-DAY': (T+86400-7200, T+86400)}
CONFIG = REPO / 'runs/example/baleen/prefetch_ml-on-partial-hit/config.json'
REFERENCE = REPO / 'runs/example/baleen/prefetch_ml-on-partial-hit/ml-ap-0.798545_6_lru_366.475GB/full_0_0.1_cache_perf.txt.stats.lzma'

def threshold_at(arm, relative_time):
    if arm not in ARMS:
        raise ValueError(arm)
    if relative_time < EVAL_START or arm == 'STATIC-BASE':
        return THETA0
    if arm == 'STATIC-LOW':
        return THETAL
    if arm == 'STATIC-HIGH':
        return THETAH
    left, right = INTERVALS[arm]
    return THETAL if left <= relative_time < right else THETAH
