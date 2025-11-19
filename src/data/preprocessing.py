"""Data ingestion and preprocessing utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
try:  # pragma: no cover - dependency guard
    from imblearn.combine import SMOTETomek  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install 'imbalanced-learn' to use the preprocessing module."
    ) from exc
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig
from src.features.selection import select_top_features
from src.utils.logger import setup_logger
from src.utils.paths import PROJECT_ROOT


def _load_raw_frames(pattern: str, chunk_size: int, sample_frac: float) -> pd.DataFrame:
    files = sorted(PROJECT_ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}'. Place raw CSVs in data/raw.")

    frames = []
    for path in files:
        reader = pd.read_csv(path, chunksize=chunk_size)
        for chunk in reader:
            if sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)
            frames.append(chunk)

    return pd.concat(frames, ignore_index=True)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return df


def _prepare_feature_matrix(df: pd.DataFrame, cfg: ProjectConfig) -> Tuple[pd.DataFrame, pd.Series]:
    label_col = cfg.data.label_column
    df = df.dropna(subset=[label_col])

    categorical_cols = [c for c in cfg.data.categorical_features if c in df.columns]
    numeric_cols = [c for c in cfg.data.numeric_features if c in df.columns]

    num_df = df[numeric_cols].copy()
    num_df = num_df.fillna(num_df.median())
    scaler = StandardScaler()
    num_scaled = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)

    cat_df = df[categorical_cols].astype(str).fillna("missing")
    cat_encoded = pd.get_dummies(cat_df, drop_first=False)

    X = pd.concat([num_scaled, cat_encoded], axis=1)
    y = df[label_col].astype(str)
    return X, y



def preprocess(config: ProjectConfig) -> Path:
    logger = setup_logger(log_dir=config.paths.logs_dir)
    logger.info("Loading raw IoT-PoT data from %s", config.data.raw_glob)

    df = _load_raw_frames(
        pattern=config.data.raw_glob,
        chunk_size=config.data.chunk_size,
        sample_frac=config.data.sample_frac,
    )
    df = _clean_columns(df)
    logger.info("Raw shape: %s", df.shape)

    X, y = _prepare_feature_matrix(df, config)
    logger.info("Feature matrix shape before selection: %s", X.shape)

    X = select_top_features(X, y, config, logger)
    logger.info("Feature matrix shape after selection: %s", X.shape)

    sampler = SMOTETomek(random_state=config.training.random_state)
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    logger.info("Balanced sample size: %s", X_balanced.shape[0])

    processed_path = Path(config.data.processed_file)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df = pd.DataFrame(X_balanced, columns=X.columns)
    processed_df[config.data.label_column] = y_balanced.values
    processed_df.to_parquet(processed_path, index=False)

    metadata = {
        "n_rows": int(processed_df.shape[0]),
        "n_features": int(processed_df.shape[1] - 1),
        "label_distribution": y_balanced.value_counts().to_dict(),
        "selected_features": X.columns.tolist(),
    }
    metadata_path = processed_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved processed dataset to %s", processed_path)

    return processed_path
