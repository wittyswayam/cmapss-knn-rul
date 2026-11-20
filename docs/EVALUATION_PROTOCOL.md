# Evaluation protocol (frozen)

This file fixes how models are scored and selected. It was written before any official test file was read,
and it must not be changed after test results have been seen.

## Metrics

Three metrics are reported everywhere, on the same predictions: **RMSE**, **MAE** and the **NASA asymmetric score**.
RMSE and MAE are in cycles. The NASA score is dimensionless.

### Error sign convention

    d = predicted RUL - true RUL

* `d > 0`: the model predicts more remaining life than there is. This is the *optimistic* or *late* error: a
  maintenance warning that comes too late.
* `d < 0`: the model predicts less remaining life than there is. This is the *pessimistic* or *early* error.

### NASA / PHM08 score (Saxena et al., 2008)

For one engine:

    s = exp(-d / 13) - 1     if d < 0   (early)
    s = exp( d / 10) - 1     if d >= 0  (late)

The official score of a model is the **sum** of `s` over the test engines. Lower is better; 0 is perfect.
Both branches grow exponentially, and the late branch uses the smaller constant (10 instead of 13), so it grows
faster: overestimating RUL is penalised more than underestimating it by the same amount.

| d (cycles) | kind | s |
|---|---|---|
| -20 | early (pessimistic) | 3.66 |
| -10 | early (pessimistic) | 1.16 |
| -5 | early (pessimistic) | 0.47 |
| +5 | late (optimistic) | 0.65 |
| +10 | late (optimistic) | 1.72 |
| +20 | late (optimistic) | 6.39 |

### Interpretation and caveats

* The score is dominated by the worst few predictions: one engine that is 40 cycles too optimistic costs
  `exp(4) - 1 = 53.6`, as much as dozens of small errors. It is therefore a high-variance estimate on small evaluation sets.
  This instability is visible in `results/phase3/phase3_report.md`, where the mean NASA score of the same model differs a lot between cross-validation and
  the development holdout while its RMSE moves far less.
* For cross-validation and the development holdouts, results are reported as the **mean score per prediction** (each of the ten cutoffs per engine counts as one prediction; the mean over folds is the unweighted mean of per-fold means) so that
  evaluation sets of different size can be compared. The official test evaluation reports the **sum** over engines.
* Two further diagnostics accompany every table: the **bias** (mean of `d`) and the **late share** (fraction of predictions with `d > 0`).

## Targets and truth

* Models are trained on the RUL clipped at 125 cycles.
* During development, engine-level metrics are computed against the clipped RUL (`*_clip`, used for selection) and also
  against the true RUL (`*_raw`). Raw errors are inflated by a few engines that really have more than 125 cycles left; this is
  a limit of the clipped target, not noise.
* Official test protocol: one prediction per test engine, at its last observed cycle; truth is the value in `RUL_FDxxx.txt`
  (unclipped); NASA score summed over engines.

## Validation protocol (development)

* All splits are by engine. Each subset's training engines are split into tuning engines (80%) and a development holdout (20%).
* Hyper-parameters are chosen by 5-fold cross-validation over tuning engines. Scaling, sensor selection, regime statistics and the
  model are fitted inside each training fold.
* **Engine-level evaluation** (selection criterion): ten random truncation points per engine from cycle 30 on, imitating the test
  protocol. **Window-level** metrics (every 5th cycle) and **short-history stress points** (cycles 5, 10, 20) are secondary.

## Model selection

Two criteria are computed and compared in every subset: lowest mean engine-level **RMSE**, and lowest mean engine-level **NASA score**
(both against the clipped RUL). Which criterion selects the final model is decided by the project owner before the test sets are evaluated.

## Status of the holdouts

| Holdout | Status |
|---|---|
| FD001, 20 engines (Phase 2) | **Exploratory development holdout.** It was evaluated more than once after the search grid had changed, so it is not an independent test and must not be used as one. (An earlier evaluation from the first grid was computed and overwritten without being inspected.) |
| FD002-FD004 (Phase 3) | Exploratory development holdouts. The CV selection was frozen to `results/phase3/selection_*.json` first, and each holdout was evaluated once; `results/phase3/holdout_audit.json` counts evaluations and `run_phase3.py holdout` refuses a repeat without `--force`. |
| Official test files | No model was fitted, tuned or selected using them. Correction of an earlier statement: in Phases 1-3 the loader parsed the test observations and the RUL labels into memory, and the Phase 1 EDA computed and saved descriptive statistics of test observed cycles and true RUL (`results/eda/*_test_engines.csv`). First official evaluation: Phase 4. |

## Phase 4: official test evaluation (frozen)

* Frozen before any test prediction: `results/phase4/frozen_protocol.json` (configurations, training policy, protocol text, SHA-256 of the selection files, of every source file that influences a prediction, and of this document). The evaluation refuses to run if any of them changed.
* Models: the engine-level-RMSE-selected configuration of each subset (the extended grid for FD002 and FD004), plus the NASA-selected FD002 configuration (k=200) as a secondary comparison. Each model is fitted on all training engines of its subset.
* Predictions are made and saved (with a hash in the chunk manifest) before the label files are opened. Labels are joined only in the report step.
* Reported per subset: RMSE, MAE, NASA score (sum over engines = official; mean per engine alongside), bias, share of late predictions. Truth is the unclipped RUL file.
* Descriptive breakdowns (by true-RUL band, by observed-history length, worst engines, a diagnostic against clipped truth) are fixed in code before the results are seen and are not used to change any model or criterion.
* CV and development-holdout numbers use a different protocol (random cutoffs, clipped truth for selection) and are shown separately; they are never mixed with official test numbers.
