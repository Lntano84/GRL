@echo off
REM Two-stage validation, sized per the PI's guidance:
REM   small budget  -> full Monte-Carlo baseline (affordable at small k)
REM   high budget   -> the contrasts that matter (sequential vs static, analytic vs random),
REM                    with the full-pool reference omitted because it costs O(k * |pool| * MC)
setlocal
cd /d "%~dp0..\.."
set PYTHONPATH=%CD%\src
set OUT=%CD%\docs\results

echo ================ scorer diagnosis ================
python scripts\audit\diagnose_scorer.py ^
  --graphs congress_twitter email_eu_core ca_grqc ^
  --fractions 0.01 0.05 0.20 --candidates 50 --mc 200 ^
  --output "%OUT%\scorer_diagnosis.json"

echo ================ stage A: small budget, FULL MC baseline ================
python scripts\experiments\go_no_go.py ^
  --graphs congress_twitter email_eu_core ^
  --fractions 0.01 --pool-size 120 --pool-draws 2 --seeds 20260917 20260918 ^
  --normalisations sum_to_one ^
  --reference-mc 50 --eval-mc 200 --state-mc 25 --stop-mc 25 --shortlist 8 ^
  --require-reference --max-seeds 300 ^
  --output "%OUT%\validation_small.json"

echo ================ stage B: high budget, contrasts only ================
python scripts\experiments\go_no_go.py ^
  --graphs congress_twitter email_eu_core ^
  --fractions 0.20 --pool-size 240 --pool-draws 2 --seeds 20260917 20260918 ^
  --normalisations sum_to_one ^
  --arms delta2_sequential delta2_patience patience_static degree_static delta2_static random_pruning selective_analytic adaptive_selective ^
  --reference-mc 50 --eval-mc 200 --state-mc 25 --stop-mc 25 --shortlist 8 ^
  --max-seeds 300 ^
  --output "%OUT%\validation_large.json"

echo DONE
