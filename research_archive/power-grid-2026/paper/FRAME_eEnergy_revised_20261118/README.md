# Anonymous manuscript source and table data

Compile manuscript.tex with an acmart-enabled TeX installation. The reviewed PDF uses Tectonic 0.17.0 with acmart 1.83.

Table 1 uses equal_budget_states.csv. Restrict to triggered=True. The source groups old13, G1, and G2 correspond to Original, Generated 1, and Generated 2. Compare local_minus_global to ±0.000001. Average that difference within each scenario (week) and then across scenarios within each group for the last column. The file preserves all 185 states.

Table 2 uses paired_results.csv. Run python -X utf8 render_table.py to regenerate closed_loop_table.tex. The CSV also records controller seconds and P95/P99 latencies, completion, and exception counts. A blank percentage denotes unavailable full-week cost or comparable timing. The data reproduce the tables from saved results. End-to-end execution additionally requires the simulator, scenarios, action library, model artifacts, and run configuration.
