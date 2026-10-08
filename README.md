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
| `analysis/` | `paper_numbers.py` (derived numbers in the text, Table 3, Fig. 3), `make_figs.py` (Figs. 1–3), `gap_baseline/` (HiGHS 1% relative-gap baseline, Table 4) |

## Requirements

Python 3.12. See `requirements.txt`. HiGHS (via `highspy`) is the main solver; SCIP (via `PySCIPOpt`) is only needed for the second-solver experiment.

## Reproducing the results

The `.bat` files were used on Windows; on Linux/macOS run the Python command inside each file.

1. **Benchmarks** (`benchmarks/`):
   - `RUN_BENCHMARKS.bat`: generates the instances, trains the predictors and evaluates every threshold on the test instances.
   - `RUN_ANALYZE.bat`: CRISP calibration over 2,000 random splits → `results/summary_bench_fastest.json` (Table 2, Appendix A).
   - `python baselines.py`: hand-set and tuned thresholds.
2. **Robustness** (`benchmarks/`): `RUN_EXTRA.bat`, then `RUN_EXTRA_ANALYZE.bat` (distribution shift, larger instances, SCIP). `RUN_SCIP_RESUME.bat` resumes interrupted SCIP runs.
3. **Application** (`application/`): `RUN_SSOA.bat` → `results/summary_ssoa_v2.json` (Table 5).
4. **Derived numbers and figures** (`analysis/`):
   - unzip `benchmarks/results/eval_records.zip` and `application/results/ssoa_records.zip` into `analysis/` (this creates `analysis/eval/<family>/*.pkl`);
   - `python paper_numbers.py` → `results/paper_numbers.json`;
   - `python make_figs.py` → `figures/`.
5. **Relative-gap baseline** (`analysis/gap_baseline/`): `python run_gap1pct.py` (records in `gap1pct_records.zip`, summary in `analysis/results/summary_gap1pct.json`).

The raw records in `*_records.zip` contain the outcome of every reduced solve, so steps 1(b), 2(b) and 4 can be reproduced without solving any MILP.

### Note on SCIP

On Windows, SCIP crashed intermittently on independent-set instances; these 80 instances were re-run with the same seeds on Linux (`benchmarks/results/scip_indset_linux_records.zip`).

## Data

`application/data/supplier_pool.csv` is derived from the audited financial statements that listed Iranian companies publish on the Codal disclosure system. `application/data/preds_IR.csv.gz` contains out-of-sample financial-distress probabilities produced by a separately trained LightGBM model.

## Citation

If you use this code, please cite the paper (citation will be added on publication).

## License

MIT (see `LICENSE`).
