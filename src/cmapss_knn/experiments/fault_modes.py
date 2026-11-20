"""Unsupervised look for fault-mode structure.

FD003 and FD004 mix two fault modes (HPC and fan degradation) but give no per-engine label. A descriptive
check: summarise each engine by its *end-of-life signature* (mean of its last 10 cycles minus mean of its first
20 cycles, per sensor), split the engines into two groups with k-means, and ask how clean the split is
(silhouette score) compared with what the same number of engines and sensors would give if the signatures were
a single Gaussian cloud. FD001 and FD002 (one fault mode) are the controls. This can show whether the
data look two-modal; it cannot say which group is which fault, and it is not used by any model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..data.schema import CYCLE_COLUMN, ID_COLUMN


def end_of_life_signature(train: pd.DataFrame, sensors: list[str], early: int = 20, late: int = 10) -> pd.DataFrame:
    """Per engine: mean of the last ``late`` cycles minus mean of the first ``early`` cycles, per sensor."""
    g = train.groupby(ID_COLUMN)
    first = train[train[CYCLE_COLUMN] <= early].groupby(ID_COLUMN)[sensors].mean()
    last_cycle = g[CYCLE_COLUMN].transform("max")
    last = train[train[CYCLE_COLUMN] > last_cycle - late].groupby(ID_COLUMN)[sensors].mean()
    return last - first


def split_in_two(sig: pd.DataFrame, seed: int, n_null: int = 200) -> dict:
    Z = ((sig - sig.mean()) / sig.std()).dropna(axis=1).to_numpy()
    labels = KMeans(2, n_init=10, random_state=seed).fit_predict(Z)
    sil = float(silhouette_score(Z, labels))
    rng = np.random.default_rng(seed)
    null = np.array([silhouette_score(x, KMeans(2, n_init=3, random_state=0).fit_predict(x)) for x in (rng.normal(size=Z.shape) for _ in range(n_null))])
    centers = pd.DataFrame(Z).groupby(labels).mean()
    diff = (centers.loc[0] - centers.loc[1]).abs()
    cols = list(sig.columns[sig.std().notna()])
    return {"labels": labels, "silhouette": sil, "null_silhouette_mean": float(null.mean()), "null_silhouette_max": float(null.max()),
            "share_of_null_below": float((null < sil).mean()), "sizes": np.bincount(labels).tolist(),
            "top_sensors": [cols[i] for i in np.argsort(-diff.to_numpy())[:3]], "pca": PCA(2, random_state=seed).fit_transform(Z)}
