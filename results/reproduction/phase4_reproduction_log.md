# Phase 4 clean-room reproduction log

Fresh extraction of the frozen package. The frozen protocol was kept; the saved predictions were deleted and recomputed with the documented commands.

| file | result | detail |
|---|---|---|
| results/phase4/predictions/fd001_primary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/predictions/fd002_nasa_secondary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/predictions/fd002_primary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/predictions/fd003_primary.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/predictions/fd004_primary.csv | MATCH | max abs numeric diff 0.00e+00 |
| figures/phase4/cv_vs_official_test.png | MATCH | pixel-identical |
| figures/phase4/official_error_distribution.png | MATCH | pixel-identical |
| figures/phase4/official_error_vs_history.png | MATCH | pixel-identical |
| figures/phase4/official_nasa_penalties.png | MATCH | pixel-identical |
| figures/phase4/official_pred_vs_true.png | MATCH | pixel-identical |
| results/phase4/cv_vs_official_reference.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_above_cap.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_by_history.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_by_rul_band.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_diagnostic_clipped_truth.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_metrics.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_per_engine.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_posthoc_history_by_cap.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/official_test_worst_engines.csv | MATCH | max abs numeric diff 0.00e+00 |
| results/phase4/phase4_report.md | DIFFERENT | bytes differ |
| results/phase4/frozen_protocol.json | MATCH | max abs numeric diff 0.00e+00 |

## Notes

- The one DIFFERENT entry, `results/phase4/phase4_report.md`, differs in a single line: the official-evaluation counter per chunk reads 2 in this clean room (the audit file was copied with count 1 and the chunks were evaluated again to reproduce them), against 1 in the reference. All tables and all numbers in the report are identical. This is the audit guard working as designed.
- The frozen protocol file was kept and re-validated: the re-run passed the check of the frozen hash, every hashed source file and every selection file before predicting.
- Prediction manifests: `predictions_sha256` and `frozen_sha256` are identical to the reference for all five chunks (compared by script).
- `verify` in the clean room: independent recomputation, report-table check, regeneration of 15 files, literal scan, and labels-read flags all pass.
- Reproduced package: `cmapss-knn-rul-phase4-pre.zip`. The delivered package additionally contains the reproduction logs.
