# Clean-room reproduction log
Package extracted to a fresh directory; each step below was run with the documented command and compared with the reference results.
| file | result | detail |
|---|---|---|
| results/phase2/equivalence_cases.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/equivalence_report.md | MATCH | byte-identical |
| results/phase2/equivalence_summary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/cv_results_fd001.csv | DIFFERENT | missing in candidate |

Note: a first attempt at the Phase 2 sweep was stopped by the 300 s tool limit (this sandbox ran about 1.7x slower than during the original run) and is repeated below as a detached background job.

| file | result | detail |
|---|---|---|
| results/phase2/cv_results_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| figures/fd001/10_k_metric_weights.png | MATCH | pixel-identical |
| figures/fd001/11_s6_configuration_A_vs_B.png | MATCH | pixel-identical |
| figures/fd001/12_baseline_and_short_history.png | MATCH | pixel-identical |
| figures/fd001/13_baselining_none_vs_causal.png | MATCH | pixel-identical |
| figures/fd001/14_holdout_predicted_vs_true.png | MATCH | pixel-identical |
| results/phase2/FD001_phase2_report.md | DIFFERENT | bytes differ |
| results/phase2/backend_comparison_cv.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/baseline_paired_configs.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/best_by_sensor_config.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/cv_constant_baseline_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/cv_results_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/cv_summary_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/equivalence_cases.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/equivalence_report.md | MATCH | byte-identical |
| results/phase2/equivalence_summary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/history_paired_configs.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/holdout_results.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/phase2_summary.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/s6_low_value_rate_by_rul.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/s6_only_diagnostic.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/s6_paired_configs.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/s6_permutation_control.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase2/top15_configurations.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd001_fold0.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd001_fold1.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd001_fold2.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd001_fold3.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd001_fold4.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd002_fold0.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd003_fold0.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd003_fold1.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd003_fold2.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd003_fold3.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd003_fold4.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv/fd004_fold0.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regimes/fd001_detector.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regimes/fd002_detector.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regimes/fd003_detector.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regimes/fd004_detector.json | MATCH | max abs numeric diff 0.00e+00 |
| figures/phase3/cross_subset_cv.png | MATCH | pixel-identical |
| figures/phase3/fault_mode_structure.png | MATCH | pixel-identical |
| figures/phase3/regime_effect_fd002_s11.png | MATCH | pixel-identical |
| figures/phase3/regime_effect_fd004_s11.png | MATCH | pixel-identical |
| figures/phase3/regime_normalisation_ablation.png | MATCH | pixel-identical |
| figures/phase3/rmse_vs_nasa_selection.png | MATCH | pixel-identical |
| figures/phase3/short_history_probe.png | MATCH | pixel-identical |
| results/phase3/cross_subset_cv_summary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv_summary_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv_summary_fd002.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv_summary_fd003.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/cv_summary_fd004.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/fault_mode_points.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/fault_mode_structure.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/holdout_fd002.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/holdout_fd003.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/holdout_fd004.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/phase3_analysis.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/phase3_report.md | DIFFERENT | bytes differ |
| results/phase3/probe/fd001_fold0.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd001_fold1.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd001_fold2.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd001_fold3.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd001_fold4.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd002_fold0.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd002_fold1.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd002_fold2.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd002_fold3.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe/fd002_fold4.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe_summary_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/probe_summary_fd002.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regime_effect_by_sensor_fd002.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regime_effect_by_sensor_fd004.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regimes_fd002.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/regimes_fd004.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/selection_fd001.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/selection_fd002.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/selection_fd003.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/selection_fd004.json | MATCH | max abs numeric diff 0.00e+00 |
| figures/phase3_extended/extension_curves.png | MATCH | pixel-identical |
| results/phase3_extended/cv/fd002_fold0_k200.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd002_fold0_k200.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd002_fold0_w100.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd002_fold0_w100.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd004_fold0_k200.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd004_fold0_k200.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd004_fold0_w100.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv/fd004_fold0_w100.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv_summary_extended_fd002.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/cv_summary_extended_fd004.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/extended_analysis.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/extended_selection_table.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/phase3_extended_report.md | MATCH | byte-identical |
| results/phase3_extended/selection_extended_fd002.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3_extended/selection_extended_fd004.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/audit/audit_summary.md | MATCH | byte-identical |
| results/phase3/audit/probe_fd001.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/audit/probe_vs_full_history_fd001.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/audit/provenance.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/audit/recompute_fd001.json | MATCH | max abs numeric diff 0.00e+00 |
| results/phase3/audit/recompute_fd003.json | MATCH | max abs numeric diff 0.00e+00 |

## Notes on the three entries marked DIFFERENT

1. `results/phase2/cv_results_fd001.csv`, "missing in candidate": the first attempt at the Phase 2 sweep was stopped by the 300 s tool limit before it wrote its file (this sandbox ran about 1.7x slower at that moment). The sweep was repeated as a detached job and its result is the MATCH entry for the same file earlier in this log (289 s).
2. `results/phase2/FD001_phase2_report.md`: 17 lines differ. After masking the two measured wall-clock values the report quotes (scratch and scikit-learn seconds per fold, in the sentence and in the last two columns of the backend table), every line is identical. This was checked by script.
3. `results/phase3/phase3_report.md`: one line differs. The clean-room run re-evaluated the development holdouts with `--force`, so its holdout audit counter reads 2 where the reference reads 1. All other content is identical. This is the audit guard working as designed.

## Coverage

Fully recomputed in the clean room and matching: tests (139), data verification (all four subsets), scratch-vs-scikit-learn equivalence, the complete Phase 2 sweep and analysis stage (including figures 10-14), Phase 3 sweeps for FD001 and FD003 (all folds), regime detector outputs (all four subsets), development holdouts for FD002-FD004, fault-mode analysis, short-history probe for FD001 (all folds), the Phase 3 analysis and report with all figures, the extended analysis, audit recomputation for FD001 and FD003, the audit summary, and the provenance audit.

Sampled, not fully recomputed (time): Phase 3 sweeps for FD002 and FD004 (fold 0 of 5), their short-history probe (FD002 fold 0 of 5), their extension chunks (fold 0 of 5), and the independent CV recomputation for FD002 and FD004. The remaining folds were copied from the package and are covered by the original run only.

Reproduced package: `cmapss-knn-rul-phase3-review.zip`. The delivered package additionally contains this log, `scripts/compare_results.py` with timing-field handling (it was used from the reference directory), and `tests/test_compare_results.py`.
