"""Phase 4: official test-set evaluation with frozen models (prediction side).

Integrity rules implemented here:

* ``freeze`` writes ``results/phase4/frozen_protocol.json`` *before* any test prediction exists. It records the model
  configurations, the training policy, the evaluation protocol, and SHA-256 hashes of the selection files and of every
  source file that influences a prediction. The file carries its own hash. It cannot be overwritten.
* ``evaluate_chunk`` refuses to run if the frozen file or any hashed source file changed since the freeze.
* A chunk is one (subset, model). It fits the model on **all** training engines, predicts the last observed cycle of every
  test engine, and saves the predictions. The RUL label file is never opened here; labels are only joined later,
  in the report step, after the predictions are hash-locked in the chunk manifest.
* Every real evaluation of a chunk is counted in ``official_test_audit.json``; completed chunks are skipped on resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from ..data.ingestion import read_inputs
from ..data.schema import ID_COLUMN
from ..data.targets import add_train_rul
from ..data.validation import _frame_checks
from ..exceptions import CMAPSSError
from ..features.regimes import RegimeNormalizer
from ..models.pipeline import RULKNNPipeline
from .phase3 import CONFIG_KEYS
from .phase3_commands import pipeline_config_from_selection

logger = logging.getLogger(__name__)

CHUNKS: tuple[tuple[str, str], ...] = (("FD001", "primary"), ("FD002", "primary"), ("FD002", "nasa_secondary"), ("FD003", "primary"), ("FD004", "primary"))
SOURCE_FILES = ("data/ingestion.py", "data/schema.py", "data/targets.py", "features/windows.py", "features/regimes.py", "features/sensors.py",
                "models/pipeline.py", "models/knn_common.py", "models/knn_sklearn.py", "models/knn_scratch.py",
                "evaluation/metrics.py", "evaluation/reference.py", "experiments/phase4_eval.py")
PROTOCOL_TEXT = {
    "model_selection": "Engine-level cross-validation RMSE (vs clipped RUL) on training engines is the primary selection criterion. MAE and the NASA score are reported alongside. "
                       "The NASA-selected FD002 model (k=200) is evaluated as a secondary comparison only.",
    "training": "Every model is fitted on ALL training engines of its subset (including the engines held out as exploratory development holdouts). "
                "Regime statistics, sensor selection and feature scaling are fitted on those training rows only.",
    "prediction": "One prediction per test engine, at its last observed cycle, from that engine's observed history only (histories start at cycle 1).",
    "truth_and_metrics": "Truth is the unclipped value in RUL_FDxxx.txt. Reported: RMSE, MAE, NASA score (sum over engines, official) and its mean per engine, bias, share of late predictions.",
    "no_tuning": "No hyper-parameter, feature, grid or protocol change is made after test results have been seen. Diagnostic breakdowns are descriptive only.",
    "data_integrity_disclosure": "In Phases 1-3 the loader parsed the test observations and the RUL label file into memory, and the Phase 1 EDA computed and saved descriptive statistics of test "
                                 "observed cycles and true RUL (results/eda/*_test_engines.csv). No test label was used to fit, tune or select any model; all selection used training engines only. "
                                 "Statements in earlier messages that the test files were 'not read' were too strong and are corrected here.",
}


class FrozenMismatch(CMAPSSError):
    """The frozen protocol, a hashed source file or a chunk manifest no longer matches."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def out_dir(cfg: Config) -> Path:
    return cfg.paths.results_dir / "phase4"


def src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def selection_source(cfg: Config, subset: str, label: str) -> tuple[Path, str]:
    r = cfg.paths.results_dir
    if subset in cfg.phase3_extension.subsets:
        return r / "phase3_extended" / f"selection_extended_{subset.lower()}.json", ("rmse_selected_primary" if label == "primary" else "nasa_selected_secondary")
    if label != "primary":
        raise ValueError(f"{subset} has no secondary model")
    return r / "phase3" / f"selection_{subset.lower()}.json", "rmse_selected"


def freeze(cfg: Config) -> Path:
    path = out_dir(cfg) / "frozen_protocol.json"
    if path.is_file():
        raise FrozenMismatch(f"{path} already exists; the protocol is frozen and is never overwritten.")
    chunks = []
    for subset, label in CHUNKS:
        src, key = selection_source(cfg, subset, label)
        sel = json.loads(src.read_text(encoding="utf-8"))[key]
        chunks.append({"id": f"{subset}:{label}", "subset": subset, "label": label, "config": {k: sel[k] for k in CONFIG_KEYS},
                       "selection_file": str(src.relative_to(cfg.paths.root)), "selection_key": key, "selection_file_sha256": _sha(src),
                       "cv_reference": {k: v for k, v in sel.items() if k.startswith(("eng_", "win_", "stress_"))}})
    payload = {"protocol": PROTOCOL_TEXT, "chunks": chunks, "seed": cfg.seed, "rul_cap": cfg.rul_cap, "baseline_cycles": cfg.features.baseline_cycles, "stats": list(cfg.features.stats),
               "source_sha256": {f: _sha(src_root() / f) for f in SOURCE_FILES}, "protocol_doc_sha256": _sha(cfg.paths.root / "docs" / "EVALUATION_PROTOCOL.md")}
    payload["frozen_sha256"] = _canonical_hash(payload)
    payload["frozen_utc"] = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_frozen(cfg: Config) -> dict:
    path = out_dir(cfg) / "frozen_protocol.json"
    if not path.is_file():
        raise FrozenMismatch("No frozen protocol found; run `freeze` first.")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in frozen.items() if k not in ("frozen_sha256", "frozen_utc")}
    if _canonical_hash(body) != frozen["frozen_sha256"]:
        raise FrozenMismatch("frozen_protocol.json was modified after it was written.")
    for f, h in frozen["source_sha256"].items():
        if _sha(src_root() / f) != h:
            raise FrozenMismatch(f"Source file {f} changed after the freeze.")
    if _sha(cfg.paths.root / "docs" / "EVALUATION_PROTOCOL.md") != frozen["protocol_doc_sha256"]:
        raise FrozenMismatch("docs/EVALUATION_PROTOCOL.md changed after the freeze.")
    for c in frozen["chunks"]:
        if _sha(cfg.paths.root / c["selection_file"]) != c["selection_file_sha256"]:
            raise FrozenMismatch(f"{c['selection_file']} changed after the freeze.")
    return frozen


def chunk_files(cfg: Config, chunk: dict) -> tuple[Path, Path]:
    base = out_dir(cfg) / "predictions" / f"{chunk['subset'].lower()}_{chunk['label']}"
    return base.with_suffix(".csv"), base.with_suffix(".json")


def chunk_status(cfg: Config, frozen: dict, chunk: dict) -> str:
    csv, man = chunk_files(cfg, chunk)
    if not csv.is_file() or not man.is_file():
        return "pending"
    m = json.loads(man.read_text(encoding="utf-8"))
    if m["frozen_sha256"] != frozen["frozen_sha256"] or m["predictions_sha256"] != _sha(csv):
        raise FrozenMismatch(f"{man.name}: manifest does not match the frozen protocol or the predictions file was altered.")
    return "complete"


def _audit_path(cfg: Config) -> Path:
    return out_dir(cfg) / "official_test_audit.json"


def record_evaluation(cfg: Config, chunk_id: str, note: str) -> int:
    p = _audit_path(cfg)
    audit = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    e = audit.setdefault(chunk_id, {"evaluations": 0, "log": []})
    e["evaluations"] += 1
    e["log"].append({"time_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "note": note})
    p.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return e["evaluations"]


def find_chunk(frozen: dict, chunk_id: str) -> dict:
    for c in frozen["chunks"]:
        if c["id"] == chunk_id:
            return c
    raise ValueError(f"unknown chunk {chunk_id!r}; expected one of {[c['id'] for c in frozen['chunks']]}")


def evaluate_chunk(cfg: Config, chunk_id: str, force: bool = False) -> str:
    """Fit on all training engines and save last-cycle predictions for one (subset, model). No labels are read."""
    frozen = load_frozen(cfg)
    chunk = find_chunk(frozen, chunk_id)
    if chunk_status(cfg, frozen, chunk) == "complete" and not force:
        return "skipped"
    subset = chunk["subset"]
    t0 = time.time()
    train_raw, test = read_inputs(subset, cfg.paths.raw_data_dir, cfg.archive_name)
    for label, df in (("train", train_raw), ("test", test)):
        bad = [c.name for c in _frame_checks(label, df) if not c.passed]
        if bad:
            raise CMAPSSError(f"{subset} {label} data failed checks: {bad}")
    spec = cfg.subsets[subset]
    if test[ID_COLUMN].nunique() != spec.n_test_units or train_raw[ID_COLUMN].nunique() != spec.n_train_units:
        raise CMAPSSError(f"{subset}: engine counts differ from the configured specification")
    train = add_train_rul(train_raw, cap=cfg.rul_cap)
    pcfg = pipeline_config_from_selection(chunk["config"], cfg)
    model = RULKNNPipeline(pcfg).fit(train)
    pred = model.predict_last_cycle(test)
    check = RULKNNPipeline(replace(pcfg, backend="scratch")).fit(train, normalizer=model.normalizer_).predict_last_cycle(test)  # second implementation, same data
    obs = test.groupby(ID_COLUMN)["cycle"].max()
    out = pd.DataFrame({"subset": subset, "model": chunk["label"], "unit": pred.index.to_numpy(), "observed_cycles": obs.reindex(pred.index).to_numpy(),
                        "window": pcfg.window.window, "predicted_rul": pred.to_numpy()})
    out["short_history"] = out["observed_cycles"] < out["window"]
    if len(out) != spec.n_test_units:
        raise CMAPSSError(f"{subset}: expected {spec.n_test_units} predictions, got {len(out)}")
    csv, man = chunk_files(cfg, chunk)
    csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    n_eval = record_evaluation(cfg, chunk_id, "forced repeat" if force else "first evaluation")
    man.write_text(json.dumps({"chunk": chunk_id, "frozen_sha256": frozen["frozen_sha256"], "config": chunk["config"], "n_train_engines": int(train_raw[ID_COLUMN].nunique()),
                               "n_train_rows": len(train_raw), "n_test_engines": len(out), "n_regimes": (model.normalizer_.n_regimes_ if model.normalizer_ else 1),
                               "scratch_vs_sklearn_max_abs_diff": float((pred - check).abs().max()), "predictions_sha256": _sha(csv), "labels_read": False,
                               "evaluation_number": n_eval, "seconds": round(time.time() - t0, 1), "completed_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    logger.info("%s done in %.0fs (%d test engines; scratch-vs-sklearn max diff %.2e)", chunk_id, time.time() - t0, len(out), float((pred - check).abs().max()))
    return "ran"
