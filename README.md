# KNN remaining-useful-life estimation on NASA C-MAPSS

Work in progress. This project estimates the remaining useful life (RUL) of simulated turbofan
engines from sensor histories, using K-Nearest Neighbours as the central model, and compares it
honestly with other approaches.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Repository, FD001 ingestion, validation, EDA | done, reviewed |
| 2 | FD001: causal features, from-scratch and scikit-learn KNN, equivalence tests, grouped CV | done, approved |
| 3 | FD002-FD004: regime normalisation, two fault modes, RMSE vs NASA selection, short-history probe; review fixes, audits and FD002/FD004 grid extension | done, approved |
| 4 | Official test-set evaluation of the frozen models on FD001-FD004 | done, awaiting review |


Results: `results/phase2/FD001_phase2_report.md`, `results/phase3/phase3_report.md`, and the official test evaluation of the frozen models in `results/phase4/phase4_report.md` (a single evaluation; cross-validation and test numbers are kept in separate tables). Scoring rules and the status of each holdout are in `docs/EVALUATION_PROTOCOL.md`.

## Setup (Windows, VS Code)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then place `CMAPSSData.zip` in `data/raw/` (see `data/README.md`) and run:

```powershell
python scripts\verify_data.py
python scripts\run_eda.py --subset FD001
python -m unittest discover -s tests -t . -v

python scripts\check_equivalence.py                       # scratch KNN vs scikit-learn (about 1 min)
python scripts\run_phase2_experiments.py --sweep-only      # CV sweep, about 4 min on one core
python scripts\run_phase2_experiments.py --reuse-sweep     # analysis, holdout, figures, about 4 min
python scripts\make_phase2_report.py                      # writes results\phase2\FD001_phase2_report.md

# Phase 3. Sweeps resume per fold / per chunk, so an interrupted run can simply be started again.
foreach ($s in "FD001","FD002","FD003","FD004") { python scripts\run_phase3.py sweep --subset $s }   # about 25 min on one core
python scripts\run_phase3.py regimes                          # saves + independently checks the regime detector output
python scripts\run_phase3.py analyze
foreach ($s in "FD002","FD003","FD004") { python scripts\run_phase3.py holdout --subset $s }       # development holdouts, refuse a repeat
foreach ($s in "FD001","FD002") { python scripts\run_phase3.py probe --subset $s }                 # short-history probe
python scripts\run_phase3.py faultmodes
python scripts\run_phase3.py report                           # writes results\phase3\phase3_report.md

# Grid extension for FD002 / FD004 (window 100, k = 200), written to results\phase3_extended\
foreach ($s in "FD002","FD004") { python scripts\run_phase3.py extend --subset $s }                # add --status to see progress
python scripts\run_phase3.py analyze-extended

# Phase 4: official test evaluation (the protocol was frozen once with `freeze`; it cannot be overwritten)
python scripts\run_phase4.py status
python scripts\run_phase4.py evaluate                         # resumable: completed chunks are skipped; predictions are saved before labels are read
python scripts\run_phase4.py report                           # joins labels to the locked predictions; tables, figures, results\phase4\phase4_report.md
python scripts\run_phase4.py verify                           # independent checks (results\phase4\audit\verify.json)

# Audits (see results\phase3\audit\)
foreach ($s in "FD001","FD002","FD003","FD004") { python scripts\audit_phase3.py recompute --subset $s }
python scripts\audit_phase3.py probe --subset FD001
python scripts\audit_phase3.py provenance
```

`pytest` also works once installed (`pytest.ini` is provided). The Phase 1 code has been run on
Linux only; it uses `pathlib` throughout, but it has not been run on Windows.

## Repository layout

```
configs/    YAML configuration (paths, expected subset structure, RUL cap, EDA settings)
data/       data documentation; raw/ is git-ignored
docs/       EVALUATION_PROTOCOL.md (metrics, NASA score, selection rules, holdout status)
figures/    generated figures, one folder per subset
models/     (reserved for saved model artefacts)
notebooks/  (reserved)
results/    generated tables and reports (results/eda/ for Phase 1)
scripts/    verify_data.py, run_eda.py, check_equivalence.py, run_phase2_experiments.py, make_phase2_report.py, run_phase3.py, audit_phase3.py, run_phase4.py, compare_results.py
src/cmapss_knn/
    config.py, logging_utils.py, exceptions.py
    data/        schema, ingestion, validation, targets
    eda/         statistics, plots, pipeline
    features/    sensors.py, windows.py (causal window features, baselining), regimes.py (regime detection and normalisation)
    models/      knn_common.py, knn_scratch.py, knn_sklearn.py, pipeline.py
    evaluation/  metrics.py (RMSE, MAE, NASA score), splits.py (engine folds, cutoffs)
    experiments/ phase2*.py, phase3*.py (sweep, analysis, extension), history_probe.py, fault_modes.py, audit.py, provenance.py, equivalence.py
    app/         (empty until a later phase)
tests/      unittest suite (also pytest-compatible)
```

## How the KNN models work

`ScratchKNNRegressor` (NumPy only) and `SklearnKNNRegressor` share one base class, so they take the same
arguments and reject the same invalid input. Fitting stores the training points. Prediction computes the
distance from a query to every stored point (Euclidean or Manhattan), keeps the k nearest, and returns the
mean of their targets, or a mean weighted by 1/distance. Equal distances are resolved by lowest training
index in the scratch model. Details and measured differences from scikit-learn are in
`results/phase2/equivalence_report.md`.

## Phase 1 outputs

* `results/eda/FD001_eda_report.md`: every number is computed in the run that wrote it.
* `results/eda/fd001_*.csv`, `fd001_eda_summary.json`: tables behind the report.
* `figures/fd001/`: nine figures.

## Data

See `data/README.md` for source, citation, licence status and the readme discrepancy for FD004.
