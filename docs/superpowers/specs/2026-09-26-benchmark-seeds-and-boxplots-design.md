# Benchmark: 50 seeds + box-plot uncertainty

## Goal
Show run-to-run variability of every scalar metric and raise the sample size
per (algorithm, scenario) from 10 to 50 seeds.

## Seeds
- `benchmark.SEEDS = list(range(1, 51))` — a superset of the old
  `[1..9, 42]`, so the existing 660 runs are reused (resumable sweep).
- `scenarios/freeze_scenarios.py` and `scenarios/freeze_coverage_scenarios.py`
  import `SEEDS` from `benchmark` instead of keeping their own copy.
- `benchmark.main()` preflights the frozen scenario CSVs and exits listing the
  missing ones plus the generator commands, instead of dispatching thousands of
  runs that each fail with `FileNotFoundError`.
- `GT2_OVERRIDES = {}` (was referenced but undefined → `NameError` on any new
  GT2 run). All scenarios use `config/gt2.yaml`.

## Box plots (`plot_benchmark._boxplot`)
- One figure per (scalar metric, scenario): `boxplot_<metric>_<scenario>.png`.
- `ax.boxplot`, one box per algorithm, filled with `colour_map` colours (same as
  the curves), `showmeans=True`, fliers shown.
- No title / legend (algorithm names are the x tick labels); size
  `scenario_panel_size()` so it tiles 6-up under either `PANEL_COLS` layout.
- `None` metric values (e.g. LND when a run never fully dies) are dropped.
- Existing bar charts and mean±std curve bands are unchanged.

## Summary CSV
`summary_by_scenario.csv` gains `<metric>_median`, `_q1`, `_q3` (the box-plot
statistics) and `_ci95` (t-distribution half-width of the 95 % CI of the mean).
Existing `_mean` / `_std` / `energy_per_packet_median` columns are unchanged.

## Testing
- `tests/test_plot_smoke.py` asserts a box-plot PNG and the new CSV columns.
- Render against the existing `results/runs` and inspect panels.
