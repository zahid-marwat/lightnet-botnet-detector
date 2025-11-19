"""Feature selection helpers."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from src.config import FeatureConfig, ProjectConfig


def variance_filter(X: pd.DataFrame, variance_threshold: float) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=variance_threshold)
    X_sel = selector.fit_transform(X)
    cols = X.columns[selector.get_support(indices=True)]
    return pd.DataFrame(X_sel, columns=cols)


def rank_features_rf(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int,
    importance_threshold: float,
    random_state: int,
) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    importances = importances[importances > importance_threshold]
    return importances.head(top_k)


def select_top_features(X: pd.DataFrame, y: pd.Series, cfg: ProjectConfig, logger=None) -> pd.DataFrame:
    filtered = variance_filter(X, cfg.features.variance_threshold)
    importances = rank_features_rf(
        filtered,
        y,
        top_k=cfg.features.top_k,
        importance_threshold=cfg.features.importance_threshold,
        random_state=cfg.training.random_state,
    )
    if logger:
        logger.info("Feature ranking complete. Top features: %s", ", ".join(importances.index))
    return filtered[importances.index]
