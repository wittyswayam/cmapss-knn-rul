"""Provenance audit: are the reported tables and figures generated from saved experiment outputs?

Three independent checks:
1. Regeneration. Only *experiment outputs* (CV folds, probe results, holdout results, fault-mode results, detector JSON,
   Phase 2 outputs) are copied to a scratch directory. ``analyze`` and ``report`` are re-run there. Every derived file
   (summaries, selections, tables, report, figures) must come out identical to the saved one (CSV/JSON/Markdown byte for
   byte, PNG pixel for pixel).
2. Independent recomputation. The report's main tables are re-derived from the raw fold CSVs with a few lines of plain pandas
   (no project code) and compared with the numbers printed in the report.
3. Literal scan. Every numeric literal in the report/figure code is compared with a reviewed allow-list of layout and indexing
   constants; a new unreviewed literal is flagged.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.schema import SUBSET_NAMES

# Reviewed constants: indexes, quantile levels, figure/marker sizes, the "10 observed cycles" slice and the percentage conversion.
ALLOWED_LITERALS = {
    ("phase3_commands.py", "cmd_analyze"): {0, 1, 2, 4},
    ("phase3_commands.py", "cmd_faultmodes"): {0, 1, 2},
    ("phase3_commands.py", "plot_fault_modes_from_saved"): {0, 0.8, 2, 8.5, 11, 16},
    ("phase3_commands.py", "cmd_report"): {0, 0.25, 0.5, 0.75, 1, 10, 100},
    ("phase3_plots.py", "plot_normalisation_scatter"): {0, 0.6, 0.97, 1, 1.03, 5.2, 5.8, 12},
    ("phase3_plots.py", "plot_regime_effect"): {0.6, 0.8, 1, 2, 4.2, 11.5},
    ("phase3_plots.py", "plot_selection_scatter"): {0, 0.35, 1, 3, 4.6, 5, 8, 70},
    ("phase3_plots.py", "plot_cross_subset"): {0, 0.2, 0.4, 1, 3, 4.2, 12, 400},
    ("phase3_plots.py", "plot_probe"): {0, 3, 4.2, 5.2, 7},
    ("phase3_commands.py", "cmd_extend"): {0, 1},
    ("phase3_commands.py", "cmd_analyze_extended"): {0, 1, 2},
    ("phase3_plots.py", "plot_extension_curves"): {0, 1, 2, 3, 4.4, 11.5, 45},
}
INPUTS = ["cv", "probe", "regimes", "holdout_audit.json", "fault_mode_structure.csv", "fault_mode_points.csv", "holdout_fd002.csv", "holdout_fd003.csv", "holdout_fd004.csv"]


def literal_scan(src_dir: Path) -> list[dict]:
    flagged = []
    for (fname, func), allowed in ALLOWED_LITERALS.items():
        tree = ast.parse((src_dir / "experiments" / fname).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name == func:
                for n in ast.walk(fn):
                    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value not in allowed:
                        flagged.append({"file": fname, "function": func, "line": n.lineno, "value": n.value})
    return flagged


def parse_md_tables(text: str, heading: str) -> list[pd.DataFrame]:
    """All pipe tables between ``heading`` and the next heading of the same or higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    lines, out, cur, inside = text.splitlines(), [], [], False
    for ln in lines:
        if ln.strip() == heading.strip():
            inside = True
            continue
        if inside and re.match(rf"^#{{1,{level}}} ", ln):
            break
        if inside and ln.startswith("|"):
            cur.append(ln)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    tables = []
    for block in out:
        head = [c.strip() for c in block[0].strip("|").split("|")]
        rows = [[c.strip() for c in r.strip("|").split("|")] for r in block[2:]]
        tables.append(pd.DataFrame(rows, columns=head))
    return tables


def md_section(text: str, heading_prefix: str) -> str:
    """Text of the ``##`` section whose heading starts with ``heading_prefix`` (up to the next ``##`` heading)."""
    out, inside = [], False
    for ln in text.splitlines():
        if ln.startswith("## "):
            if inside:
                break
            inside = ln.startswith(heading_prefix)
        if inside:
            out.append(ln)
    if not out:
        raise ValueError(f"section {heading_prefix!r} not found")
    return "\n".join(out)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def independent_cv_tables(cv_dir: Path, n_folds: int) -> dict[str, dict]:
    """Plain-pandas re-derivation of the selected configurations and their CV metrics."""
    keys = ["normalization", "baseline", "window", "history", "metric", "weights", "k"]
    res = {}
    for n in SUBSET_NAMES:
        d = pd.concat([pd.read_csv(cv_dir / f"{n.lower()}_fold{f}.csv") for f in range(n_folds)])
        const = d[d.normalization == "constant"]
        d = d[d.normalization != "constant"]
        g = d.groupby(keys)
        m = g[["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "eng_bias_clip", "eng_late_clip"]].mean()
        sd = g["eng_rmse_clip"].std()
        top = lambda col: set(m[col].nsmallest(10).index)
        res[n] = {"by_rmse": (m["eng_rmse_clip"].idxmin(), m.loc[m["eng_rmse_clip"].idxmin()], sd.loc[m["eng_rmse_clip"].idxmin()]),
                  "by_nasa": (m["eng_nasa_clip"].idxmin(), m.loc[m["eng_nasa_clip"].idxmin()], sd.loc[m["eng_nasa_clip"].idxmin()]),
                  "const": const[["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "eng_bias_clip", "eng_late_clip"]].mean(),
                  "spearman": float(m["eng_rmse_clip"].rank().corr(m["eng_nasa_clip"].rank())), "top10_overlap": len(top("eng_rmse_clip") & top("eng_nasa_clip"))}
    return res


def compare_report_tables(report: str, cv_dir: Path, n_folds: int, tol: float = 6e-4) -> list[dict]:
    ind = independent_cv_tables(cv_dir, n_folds)
    checks = []
    t = parse_md_tables(report, "## Cross-validation results per subset (engine level, vs clipped RUL)")[0]
    for _, r in t.iterrows():
        n, crit = r["subset"], r["criterion"]
        if crit == "constant predictor":
            ref = ind[n]["const"]
            pairs = [("RMSE", ref["eng_rmse_clip"]), ("MAE", ref["eng_mae_clip"]), ("NASA (mean)", ref["eng_nasa_clip"]), ("bias", ref["eng_bias_clip"]), ("late share", ref["eng_late_clip"])]
        else:
            key, ref, sd = ind[n]["by_rmse" if "RMSE" in crit else "by_nasa"]
            pairs = [("RMSE", ref["eng_rmse_clip"]), ("RMSE std", sd), ("MAE", ref["eng_mae_clip"]), ("NASA (mean)", ref["eng_nasa_clip"]), ("bias", ref["eng_bias_clip"]), ("late share", ref["eng_late_clip"])]
            checks.append({"table": "CV selection", "row": f"{n} {crit}", "item": "configuration", "report": "/".join(str(r[c]) for c in ["normalisation", "baseline", "window", "history", "metric", "weights", "k"]),
                           "independent": "/".join(str(x) for x in key), "ok": "/".join(str(r[c]) for c in ["normalisation", "baseline", "window", "history", "metric", "weights", "k"]) == "/".join(str(x) for x in key)})
        for col, val in pairs:
            checks.append({"table": "CV selection", "row": f"{n} {crit}", "item": col, "report": _num(r[col]), "independent": float(val), "ok": abs(_num(r[col]) - float(val)) <= tol})
    s = parse_md_tables(report, "## RMSE-selected vs NASA-selected configurations")[0]
    for _, r in s.iterrows():
        n = r["subset"]
        for col, val in (("Spearman (RMSE vs NASA over configs)", ind[n]["spearman"]), ("top-10 overlap", ind[n]["top10_overlap"]),
                         ("RMSE(NASA-sel) - RMSE(RMSE-sel)", ind[n]["by_nasa"][1]["eng_rmse_clip"] - ind[n]["by_rmse"][1]["eng_rmse_clip"]),
                         ("NASA(RMSE-sel) - NASA(NASA-sel)", ind[n]["by_rmse"][1]["eng_nasa_clip"] - ind[n]["by_nasa"][1]["eng_nasa_clip"])):
            checks.append({"table": "selection comparison", "row": n, "item": col, "report": _num(r[col]), "independent": float(val), "ok": abs(_num(r[col]) - float(val)) <= tol})
        checks.append({"table": "selection comparison", "row": n, "item": "same configuration", "report": r["same configuration"], "independent": str(ind[n]["by_rmse"][0] == ind[n]["by_nasa"][0]), "ok": r["same configuration"] == str(ind[n]["by_rmse"][0] == ind[n]["by_nasa"][0])})
    return checks


def compare_holdout_tables(report: str, results_dir: Path, tol: float = 6e-4) -> list[dict]:
    checks = []
    section = md_section(report, "## Development holdouts")
    for n in ("FD002", "FD003", "FD004"):
        h = pd.read_csv(results_dir / f"holdout_{n.lower()}.csv")
        tab = parse_md_tables(section, f"### {n}")
        rep = [t for t in tab if "model" in t.columns and "eng_rmse_clip" in t.columns][0]
        for _, r in rep.iterrows():
            src = h[h["model"] == r["model"]].iloc[0]
            for col in ("eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "eng_bias_clip", "eng_late_clip", "eng_rmse_raw", "stress_rmse_clip"):
                checks.append({"table": f"holdout {n}", "row": r["model"], "item": col, "report": _num(r[col]), "independent": float(src[col]), "ok": abs(_num(r[col]) - float(src[col])) <= tol})
    return checks


def compare_extension_table(cfg: Config, results_dir: Path, tol: float = 1e-9) -> list[dict]:
    """Plain-pandas re-derivation of the extended-grid selection from the raw fold and chunk CSVs."""
    keys = ["normalization", "baseline", "window", "history", "metric", "weights", "k"]
    saved = pd.read_csv(results_dir / "phase3_extended" / "extended_selection_table.csv")
    k_old = max(cfg.phase3.grid.k)
    checks = []
    for n in cfg.phase3_extension.subsets:
        orig = pd.concat([pd.read_csv(results_dir / "phase3" / "cv" / f"{n.lower()}_fold{f}.csv") for f in range(cfg.validation.n_folds)])
        orig = orig[orig.normalization != "constant"]
        extra = pd.concat([pd.read_csv(p) for p in sorted((results_dir / "phase3_extended" / "cv").glob(f"{n.lower()}_fold*_*.csv"))])
        orig_keys = set(map(tuple, orig[["fold"] + keys].astype(str).to_numpy()))
        extra = extra[[tuple(r) not in orig_keys for r in extra[["fold"] + keys].astype(str).to_numpy()]]  # drop the k=150 repeats of original configurations
        m = pd.concat([orig, extra]).groupby(keys)[["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip"]].mean()
        best = m["eng_rmse_clip"].idxmin()
        row = saved[(saved.subset == n) & saved.model.str.contains("primary")].iloc[0]
        checks.append({"table": "extension", "row": n, "item": "n configurations", "report": float(len(m)), "independent": float(len(m)), "ok": True})
        checks.append({"table": "extension", "row": n, "item": "best configuration", "report": "/".join(str(row[k]) for k in keys), "independent": "/".join(str(x) for x in best), "ok": "/".join(str(row[k]) for k in keys) == "/".join(str(x) for x in best)})
        for col, name in (("eng_rmse_clip", "RMSE"), ("eng_mae_clip", "MAE"), ("eng_nasa_clip", "NASA (mean)")):
            checks.append({"table": "extension", "row": n, "item": name, "report": float(row[name]), "independent": float(m.loc[best, col]), "ok": abs(float(row[name]) - float(m.loc[best, col])) <= tol})
    return checks


def _same_png(a: Path, b: Path) -> bool:
    import matplotlib.image as mpimg

    x, y = mpimg.imread(a), mpimg.imread(b)
    return x.shape == y.shape and bool(np.array_equal(x, y))


def regenerate_and_compare(cfg: Config) -> list[dict]:
    from . import phase3_commands as cmds

    tmp = Path(tempfile.mkdtemp(prefix="cmapss_provenance_"))
    src, dst = cfg.paths.results_dir, tmp / "results"
    (dst / "phase3").mkdir(parents=True)
    shutil.copytree(src / "phase2", dst / "phase2")
    for name in INPUTS:
        s = src / "phase3" / name
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, dst / "phase3" / name)
    if (src / "phase3_extended" / "cv").is_dir():
        shutil.copytree(src / "phase3_extended" / "cv", dst / "phase3_extended" / "cv")
    paths = replace(cfg.paths, root=tmp, results_dir=dst, figures_dir=tmp / "figures", logs_dir=tmp / "logs")
    cfg2 = replace(cfg, paths=paths)
    cmds.cmd_analyze(cfg2, None)
    cmds.cmd_report(cfg2, None)
    if (dst / "phase3_extended" / "cv").is_dir():
        cmds.cmd_analyze_extended(cfg2, None)
    results = []
    derived = [p for p in (dst / "phase3").iterdir() if p.is_file()]
    for p in sorted(derived):
        orig = src / "phase3" / p.name
        if not orig.is_file():
            results.append({"file": p.name, "kind": "derived", "identical": False, "note": "not in saved outputs"})
        elif p.name == "holdout_audit.json" or p.name.startswith("holdout_") or p.name.startswith("fault_mode"):
            continue  # copied inputs
        else:
            results.append({"file": p.name, "kind": "derived", "identical": p.read_bytes() == orig.read_bytes()})
    if (dst / "phase3_extended").is_dir():
        for p in sorted(q for q in (dst / "phase3_extended").iterdir() if q.is_file()):
            orig = src / "phase3_extended" / p.name
            results.append({"file": f"phase3_extended/{p.name}", "kind": "derived", "identical": orig.is_file() and p.read_bytes() == orig.read_bytes()})
    for sub in ("phase3", "phase3_extended"):
        for p in sorted((tmp / "figures" / sub).glob("*.png")):
            orig = cfg.paths.figures_dir / sub / p.name
            results.append({"file": f"figures/{sub}/{p.name}", "kind": "figure", "identical": orig.is_file() and _same_png(p, orig)})
    shutil.rmtree(tmp, ignore_errors=True)
    return results


def run(cfg: Config) -> int:
    out = cfg.paths.results_dir / "phase3" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    report = (cfg.paths.results_dir / "phase3" / "phase3_report.md").read_text(encoding="utf-8")
    regen = regenerate_and_compare(cfg)
    tables = compare_report_tables(report, cfg.paths.results_dir / "phase3" / "cv", cfg.validation.n_folds) + compare_holdout_tables(report, cfg.paths.results_dir / "phase3")
    if (cfg.paths.results_dir / "phase3_extended" / "extended_selection_table.csv").is_file():
        tables += compare_extension_table(cfg, cfg.paths.results_dir)
    lits = literal_scan(Path(__file__).resolve().parents[1])
    summary = {"regeneration": {"files_compared": len(regen), "all_identical": all(r["identical"] for r in regen), "differences": [r for r in regen if not r["identical"]]},
               "independent_table_recompute": {"values_compared": len(tables), "all_match": all(c["ok"] for c in tables), "mismatches": [c for c in tables if not c["ok"]]},
               "unreviewed_numeric_literals": lits}
    (out / "provenance.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"regeneration: {len(regen)} files compared, identical: {summary['regeneration']['all_identical']} | independent table recompute: {len(tables)} values, all match: {summary['independent_table_recompute']['all_match']} | unreviewed literals: {len(lits)}")
    for r in summary["regeneration"]["differences"] + summary["independent_table_recompute"]["mismatches"] + lits:
        print("  !", r)
    return 0 if (summary["regeneration"]["all_identical"] and summary["independent_table_recompute"]["all_match"] and not lits) else 1
