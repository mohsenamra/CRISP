# CRISP: Conformal Risk-controlled Integer Search-space Pruning

Code and data for the paper

> M. Amra, *Certified search-space reduction for mixed-integer programming: calibrating learned variable fixing with conformal risk control* (submitted to Engineering Applications of Artificial Intelligence).

CRISP fixes the binary variables whose predicted confidence is at least a threshold λ and solves the reduced MILP exactly. It chooses λ on past instances with conformal risk control (CRC) or Learn-then-Test (LTT). The result is a finite-sample, distribution-free guarantee on the optimality gap of the returned solution.

## Repository layout

| Folder | Contents |
|---|---|
| `benchmarks/` | Instance generators (set covering, combinatorial auction, capacitated facility location, maximum independent set), features, GBM and GNN predictors, evaluation of every threshold, CRISP calibration (CRC, LTT, solve-free, fastest-valid rule), baselines, shift / larger-instance / SCIP experiments |
| `benchmarks/models/` | Trained predictors used in the paper |
| `benchmarks/results/` | Raw evaluation records (`*_records.zip`) and summary JSON files |
| `application/` | Multi-period supplier selection and order allocation (SSOA) application: generator, run script, supplier pool, distress probabilities, records and summaries |
| `analysis/` | Derived numbers, figures, the re-analysis of the application and the HiGHS 1% relative-gap baseline |
| `analysis/figures_src/` | TikZ source of Fig. 2 (CRISP framework) |

## Requirements

Python 3.12; the versions used for the paper are pinned in `requirements.txt`. HiGHS (via `highspy`) is the main solver; SCIP (via `PySCIPOpt`) is only needed for the second-solver experiment.

## Where each table and figure comes from

| Paper item | Produced by | File |
|---|---|---|
| Table 2 (families), Table 3 (main results), Tables 7–10 (all configurations) | `benchmarks/bench_analyze.py` (`RUN_ANALYZE.bat`) | `benchmarks/results/summary_bench_fastest.json` |
| Table 4 (hand-set and tuned thresholds) | `benchmarks/baselines.py` (hand-set rows), `analysis/paper_numbers.py` (tuned vs CRISP rows) | `benchmarks/results/summary_baselines.json`, `analysis/results/paper_numbers.json` |
| Table 5 (1% relative-gap baseline) | `analysis/gap_baseline/run_gap1pct.py` | `analysis/results/summary_gap1pct.json` |
| Table 6 (shift, larger instances) and SCIP results | `benchmarks/extra_analyze.py` (`RUN_EXTRA_ANALYZE.bat`) | `benchmarks/results/summary_extra_final.json` |
| Tables 11–13 (application) | `application/app_run.py` (`RUN_SSOA.bat`), then `analysis/analyse_ssoa.py` | `application/results/summary_ssoa.json`, `analysis/results/summary_ssoa_v2.json` |
| Numbers in the text (fallback rates, calibration cost, truncation, chronological test) | `analysis/paper_numbers.py` | `analysis/results/paper_numbers.json` |
| Fig. 1 (positioning map) | `analysis/make_fig_positioning.py` (data), `analysis/make_positioning_tikz.py` (TikZ) | `analysis/figures/` |
| Fig. 2 (framework) | TikZ | `analysis/figures_src/fig_framework_tikz.tex` |
| Figs. 3 and 4 | `analysis/make_figs.py` | `analysis/figures/` |

## Reproducing the results

The `.bat` files were used on Windows; on Linux/macOS run the Python commands inside each file.

**Full re-run (solves all MILPs):**

1. `benchmarks/RUN_BENCHMARKS.bat`: generates the instances, trains the predictors, evaluates every threshold on the test instances and runs the analysis.
2. `benchmarks/RUN_EXTRA.bat`, then `benchmarks/RUN_EXTRA_ANALYZE.bat`: distribution shift, larger instances and SCIP. `RUN_SCIP_RESUME.bat` resumes interrupted SCIP runs.
3. `application/RUN_SSOA.bat`: the application.
4. `python analysis/gap_baseline/run_gap1pct.py`: the relative-gap baseline (records in `analysis/gap_baseline/gap1pct_records.zip`).

**Re-analysis from the stored records (no MILP is solved):** the `*_records.zip` archives contain the outcome of every reduced solve.

- Benchmarks: unzip `benchmarks/results/eval_records.zip` and `benchmarks/results/extra_records.zip` inside `benchmarks/` (this creates `eval/`, `eval_x/` and `preds/`), then run `python bench_analyze.py`, `python baselines.py` and `python extra_analyze.py` there.
- Derived numbers, application and figures: unzip `benchmarks/results/eval_records.zip` and `application/results/ssoa_records.zip` inside `analysis/` (this creates `analysis/eval/<family>/*.pkl` and `analysis/preds/`), then run, inside `analysis/`:
  - `python paper_numbers.py` → `results/paper_numbers.json`
  - `python analyse_ssoa.py` → `results/summary_ssoa_v2.json`
  - `python make_figs.py`, `python make_fig_positioning.py` and `python make_positioning_tikz.py` → `figures/`

### Note on SCIP

On Windows, SCIP crashed intermittently on independent-set instances; these 80 instances were re-run with the same seeds on Linux (`benchmarks/results/scip_indset_linux_records.zip`). In the SCIP experiment, calibration and deployment both use SCIP.

## Data

`application/data/supplier_pool.csv` is derived from the audited financial statements that listed Iranian companies publish on the Codal disclosure system. `application/data/preds_IR.csv.gz` contains out-of-sample financial-distress probabilities produced by a separately trained LightGBM model.

## Citation

If you use this code, please cite the paper (citation will be added on publication).

## License

MIT (see `LICENSE`).
