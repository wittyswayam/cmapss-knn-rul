"""Phase 4 report: join labels to the locked predictions, compute official metrics, tables, figures and the report.

Everything is derived from saved files: the hash-locked prediction chunks, the label files, the frozen protocol, and the CV and
development-holdout result files. Breakdown definitions below were fixed before any test result was inspected.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.ingestion import read_test_labels
from ..evaluation.metrics import nasa_score, nasa_terms, regression_metrics
from ..exceptions import CMAPSSError
from . import phase4_eval as ev
from . import phase4_plots as pl

BANDS = ((0, 30, "RUL 0-30"), (31, 60, "RUL 31-60"), (61, 100, "RUL 61-100"), (101, np.inf, "RUL > 100"))
MODEL_LABEL = {"primary": "KNN selected by CV RMSE (primary)", "nasa_secondary": "KNN selected by CV NASA (secondary)"}
N_WORST = 5


def locked_predictions(cfg: Config) -> tuple[pd.DataFrame, dict]:
    frozen = ev.load_frozen(cfg)
    parts = []
    for c in frozen["chunks"]:
        if ev.chunk_status(cfg, frozen, c) != "complete":
            raise CMAPSSError(f"Chunk {c['id']} is not complete; run `evaluate` first.")
        parts.append(pd.read_csv(ev.chunk_files(cfg, c)[0]))
    return pd.concat(parts, ignore_index=True), frozen


def assign_band(true_rul: pd.Series) -> pd.Series:
    """RUL bands: 0-30, 31-60, 61-100, above 100 (whole-cycle edges)."""
    return pd.cut(true_rul, [b[0] - 0.5 for b in BANDS] + [np.inf], labels=[b[2] for b in BANDS])


def attach_labels(cfg: Config, preds: pd.DataFrame) -> pd.DataFrame:
    """First use of the test labels. Predictions are already locked (hash in the chunk manifests)."""
    df = preds.copy()
    truth = {s: read_test_labels(s, cfg.paths.raw_data_dir, cfg.archive_name) for s in df["subset"].unique()}
    df["true_rul"] = [float(truth[s].loc[u]) for s, u in zip(df["subset"], df["unit"])]
    df["error"] = df["predicted_rul"] - df["true_rul"]  # positive = late (optimistic)
    df["abs_error"] = df["error"].abs()
    df["nasa_term"] = nasa_terms(df["true_rul"], df["predicted_rul"])
    df["true_above_cap"] = df["true_rul"] > cfg.rul_cap
    df["rul_band"] = assign_band(df["true_rul"])
    return df


def metrics_of(g: pd.DataFrame, truth_col: str = "true_rul") -> dict[str, float]:
    m = regression_metrics(g[truth_col], g["predicted_rul"])
    return {"engines": len(g), "RMSE": m["rmse"], "MAE": m["mae"], "NASA score (sum)": nasa_score(g[truth_col], g["predicted_rul"]), "NASA (mean per engine)": m["nasa_mean"],
            "bias": m["bias"], "late share": m["late_share"]}


def official_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"subset": s, "model": m, **metrics_of(g)} for (s, m), g in df.groupby(["subset", "model"], sort=True)]
    return pd.DataFrame(rows)


def breakdowns(df: pd.DataFrame, cap: int) -> dict[str, pd.DataFrame]:
    prim = df[df["model"] == "primary"]
    out = {}
    out["by_rul_band"] = pd.DataFrame([{"subset": s, "band": b, **{k: v for k, v in metrics_of(g).items() if k != "NASA (mean per engine)"}} for (s, b), g in prim.groupby(["subset", "rul_band"], sort=True, observed=True)])
    out["by_history"] = pd.DataFrame([{"subset": s, "history": "shorter than window" if short else "at least window", "window": int(g["window"].iloc[0]), **{k: v for k, v in metrics_of(g).items() if k != "NASA (mean per engine)"}}
                                      for (s, short), g in prim.groupby(["subset", "short_history"], sort=True)])
    worst = []
    for s, g in prim.groupby("subset", sort=True):
        top = g.nlargest(N_WORST, "nasa_term")
        worst += [{"subset": s, "unit": int(r.unit), "observed_cycles": int(r.observed_cycles), "true_rul": r.true_rul, "predicted_rul": r.predicted_rul, "error": r.error, "nasa_term": r.nasa_term,
                   "share of subset NASA sum": r.nasa_term / g["nasa_term"].sum()} for r in top.itertuples()]
    out["worst_engines"] = pd.DataFrame(worst)
    out["above_cap"] = pd.DataFrame([{"subset": s, "true RUL above cap": "yes" if flag else "no", "engines": len(g), "mean true RUL": g["true_rul"].mean(), "mean predicted RUL": g["predicted_rul"].mean(), "bias": g["error"].mean()}
                                     for (s, flag), g in prim.groupby(["subset", "true_above_cap"], sort=True)])
    clipped = prim.assign(true_clipped=np.minimum(prim["true_rul"], cap))
    out["posthoc_history_by_cap"] = posthoc_history_by_cap(prim, cap)
    out["diagnostic_clipped_truth"] = pd.DataFrame([{"subset": s, **{k: v for k, v in metrics_of(g, "true_clipped").items() if k in ("RMSE", "MAE", "NASA (mean per engine)")}} for s, g in clipped.groupby("subset", sort=True)])
    return out


def posthoc_history_by_cap(prim: pd.DataFrame, cap: int) -> pd.DataFrame:
    """POST-HOC (added after the first look at the results): do short histories and the training cap explain errors separately?"""
    rows = []
    for (s, short, above), g in prim.groupby(["subset", "short_history", "true_above_cap"], sort=True):
        clipped = g.assign(true_clipped=np.minimum(g["true_rul"], cap))
        rows.append({"subset": s, "history": "shorter than window" if short else "at least window", "true RUL above cap": "yes" if above else "no", "engines": len(g),
                     "RMSE vs true RUL": metrics_of(g)["RMSE"], "RMSE vs clipped truth": metrics_of(clipped, "true_clipped")["RMSE"], "bias": g["error"].mean()})
    return pd.DataFrame(rows)


def cv_reference(cfg: Config, frozen: dict, official: pd.DataFrame) -> pd.DataFrame:
    rows = []
    res = cfg.paths.results_dir
    p2 = pd.read_csv(res / "phase2" / "holdout_results.csv")
    for c in frozen["chunks"]:
        if c["label"] != "primary":
            continue
        s, ref = c["subset"], c["cv_reference"]
        if s == "FD001":
            hold = float(p2[p2["model"] == "KNN best A"]["eng_rmse_raw"].iloc[0])
        else:
            hold = float(pd.read_csv(res / "phase3" / f"holdout_{s.lower()}.csv").query("model == 'KNN selected by RMSE'")["eng_rmse_raw"].iloc[0])
        t = official[(official["subset"] == s) & (official["model"] == "primary")].iloc[0]
        rows.append({"subset": s, "cv_rmse_clipped_truth": ref["eng_rmse_clip_mean"], "cv_rmse_std_over_folds": ref["eng_rmse_clip_std"], "cv_rmse_raw_truth": ref["eng_rmse_raw_mean"],
                     "cv_mae_clipped_truth": ref["eng_mae_clip_mean"], "cv_nasa_mean_per_prediction": ref["eng_nasa_clip_mean"],
                     "exploratory_holdout_rmse_raw_truth": hold, "official_test_rmse": t["RMSE"], "official_test_mae": t["MAE"], "official_test_nasa_mean_per_engine": t["NASA (mean per engine)"]})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    from .phase3_commands import _md as md3

    return md3(df, fmt)


def build_report(cfg: Config, out: Path, figs: Path) -> dict[str, Path]:
    preds, frozen = locked_predictions(cfg)
    df = attach_labels(cfg, preds)
    out.mkdir(parents=True, exist_ok=True)
    official = official_metrics(df)
    bd = breakdowns(df, cfg.rul_cap)
    ref = cv_reference(cfg, frozen, official)
    cols = ["subset", "model", "unit", "observed_cycles", "window", "short_history", "true_rul", "predicted_rul", "error", "abs_error", "nasa_term", "true_above_cap", "rul_band"]
    df[cols].to_csv(out / "official_test_per_engine.csv", index=False)
    official.to_csv(out / "official_test_metrics.csv", index=False)
    for k, v in bd.items():
        v.to_csv(out / f"official_test_{k}.csv", index=False)
    ref.to_csv(out / "cv_vs_official_reference.csv", index=False)
    prim = df[df["model"] == "primary"]
    saved = {"pred_vs_true": pl.plot_pred_vs_true(prim, cfg.rul_cap, figs / "official_pred_vs_true.png"), "error_hist": pl.plot_error_hist(prim, figs / "official_error_distribution.png"),
             "cv_vs_test": pl.plot_cv_vs_test(ref, figs / "cv_vs_official_test.png"), "error_vs_history": pl.plot_error_vs_history(prim, figs / "official_error_vs_history.png"),
             "nasa_terms": pl.plot_nasa_terms(prim, figs / "official_nasa_penalties.png")}
    manifests = {c["id"]: json.loads(ev.chunk_files(cfg, c)[1].read_text(encoding="utf-8")) for c in frozen["chunks"]}
    audit = json.loads(ev._audit_path(cfg).read_text(encoding="utf-8"))
    backend_diff = {k: format(v["scratch_vs_sklearn_max_abs_diff"], ".1e") for k, v in manifests.items()}
    evaluations = {k: v["evaluations"] for k, v in audit.items()}

    shown = official.assign(model=official["model"].map(MODEL_LABEL))
    L = ["# Phase 4: official test-set results (generated)", "",
         "Generated by `scripts/run_phase4.py report` from saved files only. **Official test results and cross-validation results come from different protocols and are shown in separate tables; they are not directly comparable.**", "",
         "## Integrity", "",
         f"- Protocol frozen at {frozen['frozen_utc']} (SHA-256 `{frozen['frozen_sha256'][:16]}...`) before any test prediction. Models are fitted on all training engines of each subset.",
         "- Predictions were saved and hash-locked in the chunk manifests before the label files were opened (`labels_read` is false in every manifest).",
         f"- Official evaluations per chunk (audit file): {evaluations}.",
         f"- Scratch KNN vs scikit-learn on the test predictions, maximum absolute difference per chunk: {backend_diff}.",
         f"- Disclosure: {frozen['protocol']['data_integrity_disclosure']}", "",
         "## Official test results", "",
         "One prediction per test engine at its last observed cycle; truth is the unclipped RUL file. NASA score (sum) is the official score; lower is better for all metrics. "
         "`bias` > 0 means predictions are on average too optimistic; `late share` is the fraction of engines predicted later than true.", "",
         _md(shown, "{:.2f}"), ""]
    L += ["## Cross-validation and exploratory holdout versus official test (different protocols)", "",
          "CV columns: engine-level CV on training engines, random cutoffs at least 30 cycles in, selection metric against the clipped RUL. The exploratory-holdout column is a small development check (evaluated more than once for FD001). "
          "The official test uses the last observed cycle of each test engine and the unclipped RUL.", "", _md(ref, "{:.2f}"), ""]
    L += ["## Descriptive breakdowns (primary models; fixed before the results were seen; not used to change anything)", "", "### By true RUL band", "", _md(bd["by_rul_band"], "{:.2f}"), "",
          "### By observed history relative to the window length", "", _md(bd["by_history"], "{:.2f}"), "",
          f"### Engines whose true RUL exceeds the training cap ({cfg.rul_cap})", "", _md(bd["above_cap"], "{:.2f}"), "",
          f"### The {N_WORST} engines with the largest NASA penalty per subset", "", _md(bd["worst_engines"], "{:.3f}"), "",
          "### Diagnostic only: metrics if the truth were clipped at the cap (not an official metric)", "", _md(bd["diagnostic_clipped_truth"], "{:.2f}"), "",
          "### Post-hoc (added after the first look at the results): history length and the cap together", "", _md(bd["posthoc_history_by_cap"], "{:.2f}"), ""]
    L += ["## Figures", "", *[f"- `{p.relative_to(cfg.paths.root).as_posix()}`" for p in saved.values()], "",
          "## Saved outputs", "", "- `results/phase4/predictions/*.csv` and manifests: locked predictions (no labels)", "- `results/phase4/official_test_per_engine.csv`: per-engine predictions with labels, errors and NASA terms",
          "- `results/phase4/official_test_metrics.csv`, `official_test_by_rul_band.csv`, `official_test_by_history.csv`, `official_test_worst_engines.csv`, `official_test_above_cap.csv`, `official_test_diagnostic_clipped_truth.csv`, `official_test_posthoc_history_by_cap.csv`, `cv_vs_official_reference.csv`", ""]
    (out / "phase4_report.md").write_text("\n".join(L), encoding="utf-8")
    return {"report": out / "phase4_report.md", **saved}


def run_report(cfg: Config) -> int:
    build_report(cfg, ev.out_dir(cfg), cfg.paths.figures_dir / "phase4")
    return 0


ALLOWED_LITERALS = {  # reviewed: indexes, band-edge offset, marker/figure sizes, bin counts, truncated hash display
    ("phase4_report.py", "locked_predictions"): {0}, ("phase4_report.py", "attach_labels"): {0, 0.5, 2}, ("phase4_report.py", "breakdowns"): {0}, ("phase4_report.py", "cv_reference"): {0},
    ("phase4_report.py", "build_report"): {1, 16},
    ("phase4_plots.py", "plot_pred_vs_true"): {0, 0.7, 1, 2, 8, 9, 11, 16}, ("phase4_plots.py", "plot_error_hist"): {0, 0.8, 1, 1.5, 2, 8, 11, 30},
    ("phase4_plots.py", "plot_cv_vs_test"): {0.27, 4.8, 8, 9.5}, ("phase4_plots.py", "plot_error_vs_history"): {0, 0.7, 1, 2, 8, 11, 16}, ("phase4_plots.py", "plot_nasa_terms"): {1.0, 2, 5, 8, 11},
}
REGEN_OUTPUTS = ("official_test_per_engine.csv", "official_test_metrics.csv", "official_test_by_rul_band.csv", "official_test_by_history.csv", "official_test_worst_engines.csv", "official_test_above_cap.csv",
                 "official_test_diagnostic_clipped_truth.csv", "official_test_posthoc_history_by_cap.csv", "cv_vs_official_reference.csv", "phase4_report.md")


def unreviewed_literals() -> list[dict]:
    import ast

    flagged = []
    for (fname, func), allowed in ALLOWED_LITERALS.items():
        tree = ast.parse((Path(__file__).resolve().parent / fname).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name == func:
                flagged += [{"file": fname, "function": func, "line": n.lineno, "value": n.value} for n in ast.walk(fn)
                            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value not in allowed]
    return flagged


def run_verify(cfg: Config) -> int:
    """Independent checks of the saved Phase 4 results. Writes results/phase4/audit/verify.json."""
    import shutil
    import tempfile

    from ..evaluation.reference import mae_reference, mean_nasa_reference, nasa_score_reference, rmse_reference
    from .provenance import _same_png, md_section, parse_md_tables

    out = ev.out_dir(cfg)
    (out / "audit").mkdir(exist_ok=True)
    preds, frozen = locked_predictions(cfg)  # raises if the frozen file, a source file, a selection file or any chunk hash changed
    manifests = {c["id"]: json.loads(ev.chunk_files(cfg, c)[1].read_text(encoding="utf-8")) for c in frozen["chunks"]}
    audit = json.loads(ev._audit_path(cfg).read_text(encoding="utf-8"))
    integrity = {"frozen_hash_and_sources_verified": True, "chunks_complete": len(manifests), "labels_read_flags": sorted({m["labels_read"] for m in manifests.values()}),
                 "evaluations_per_chunk": {k: v["evaluations"] for k, v in audit.items()}, "max_scratch_vs_sklearn_diff": max(m["scratch_vs_sklearn_max_abs_diff"] for m in manifests.values())}

    # 1. independent recomputation: locked predictions + label files + naive loop-based reference metrics
    metrics = pd.read_csv(out / "official_test_metrics.csv")
    per = pd.read_csv(out / "official_test_per_engine.csv")
    recompute = []
    for c in frozen["chunks"]:
        locked = pd.read_csv(ev.chunk_files(cfg, c)[0]).sort_values("unit")
        labels = read_test_labels(c["subset"], cfg.paths.raw_data_dir, cfg.archive_name).loc[locked["unit"]]
        true, pred = labels.to_numpy().tolist(), locked["predicted_rul"].to_numpy().tolist()
        saved = metrics[(metrics["subset"] == c["subset"]) & (metrics["model"] == c["label"])].iloc[0]
        mine = {"RMSE": rmse_reference(true, pred), "MAE": mae_reference(true, pred), "NASA score (sum)": nasa_score_reference(true, pred), "NASA (mean per engine)": mean_nasa_reference(true, pred)}
        csv_rows = per[(per["subset"] == c["subset"]) & (per["model"] == c["label"])].sort_values("unit")
        recompute.append({"chunk": c["id"], "engines": len(true), "max_abs_diff": max(abs(mine[k] - float(saved[k])) for k in mine),
                          "per_engine_labels_match_label_file": bool(np.allclose(csv_rows["true_rul"].to_numpy(), labels.to_numpy())),
                          "per_engine_predictions_match_locked": bool(np.allclose(csv_rows["predicted_rul"].to_numpy(), locked["predicted_rul"].to_numpy(), atol=0, rtol=0))})

    # 2. the numbers printed in the report's main table equal the metrics file
    report = (out / "phase4_report.md").read_text(encoding="utf-8")
    table = parse_md_tables(md_section(report, "## Official test results"), "## Official test results")[0]
    table_checks = []
    for _, r in table.iterrows():
        label = {v: k for k, v in MODEL_LABEL.items()}[r["model"]]
        s = metrics[(metrics["subset"] == r["subset"]) & (metrics["model"] == label)].iloc[0]
        for col in ("RMSE", "MAE", "NASA score (sum)", "NASA (mean per engine)", "bias", "late share"):
            table_checks.append(abs(float(r[col]) - float(s[col])) <= 0.0051)

    # 3. regeneration from the saved prediction chunks only
    tmp = Path(tempfile.mkdtemp(prefix="cmapss_phase4_"))
    try:
        res2 = tmp / "results"
        (res2 / "phase4").mkdir(parents=True)
        for name in ("predictions", "frozen_protocol.json", "official_test_audit.json"):
            s = out / name
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, res2 / "phase4" / name)
        shutil.copytree(cfg.paths.results_dir / "phase2", res2 / "phase2")
        (res2 / "phase3").mkdir()
        for f in (cfg.paths.results_dir / "phase3").glob("holdout_fd00*.csv"):
            shutil.copy2(f, res2 / "phase3" / f.name)
        for c in frozen["chunks"]:
            dst = tmp / c["selection_file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cfg.paths.root / c["selection_file"], dst)
        (tmp / "docs").mkdir()
        shutil.copy2(cfg.paths.root / "docs" / "EVALUATION_PROTOCOL.md", tmp / "docs")
        cfg2 = replace(cfg, paths=replace(cfg.paths, root=tmp, results_dir=res2, figures_dir=tmp / "figures"))
        build_report(cfg2, res2 / "phase4", tmp / "figures" / "phase4")
        regen = [{"file": f, "identical": (res2 / "phase4" / f).read_bytes() == (out / f).read_bytes()} for f in REGEN_OUTPUTS]
        regen += [{"file": f"figures/phase4/{p.name}", "identical": _same_png(p, cfg.paths.figures_dir / "phase4" / p.name)} for p in sorted((tmp / "figures" / "phase4").glob("*.png"))]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    result = {"integrity": integrity, "independent_recompute": recompute, "report_table_values_checked": len(table_checks), "report_table_all_match": all(table_checks),
              "regeneration": {"files_compared": len(regen), "all_identical": all(r["identical"] for r in regen), "differences": [r for r in regen if not r["identical"]]}, "unreviewed_numeric_literals": unreviewed_literals()}
    (out / "audit" / "verify.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    ok = (all(r["max_abs_diff"] < 1e-9 and r["per_engine_labels_match_label_file"] and r["per_engine_predictions_match_locked"] for r in recompute) and result["report_table_all_match"]
          and result["regeneration"]["all_identical"] and not result["unreviewed_numeric_literals"] and integrity["labels_read_flags"] == [False])
    print(f"independent recompute max diff {max(r['max_abs_diff'] for r in recompute):.1e} | report table values checked {len(table_checks)}, all match: {result['report_table_all_match']} | "
          f"regeneration: {len(regen)} files, identical: {result['regeneration']['all_identical']} | unreviewed literals: {len(result['unreviewed_numeric_literals'])} | labels_read flags: {integrity['labels_read_flags']}")
    return 0 if ok else 1
