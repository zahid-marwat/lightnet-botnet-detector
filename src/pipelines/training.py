"""CLI-friendly training pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config
from src.data.preprocessing import preprocess
from src.models.pso_lightgbm import PSOLightGBMTuner
from src.utils.logger import setup_logger


def _load_processed_or_run(config: ProjectConfig, logger) -> pd.DataFrame:
    processed_path = Path(config.data.processed_file)
    if not processed_path.exists():
        logger.info("Processed file %s missing. Running preprocessing...", processed_path)
        preprocess(config)
    logger.info("Loading processed dataset from %s", processed_path)
    return pd.read_parquet(processed_path)


def run_training_pipeline(config: ProjectConfig, stage: str = "all") -> None:
    logger = setup_logger(log_dir=config.paths.logs_dir)

    if stage in {"all", "preprocess"}:
        preprocess(config)
        if stage == "preprocess":
            logger.info("Preprocessing complete; exiting because stage=preprocess")
            return

    df = _load_processed_or_run(config, logger)
    label_col = config.data.label_column
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        stratify=y,
        random_state=config.training.random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config.training.val_size,
        stratify=y_train,
        random_state=config.training.random_state,
    )

    logger.info(
        "Dataset split: train=%s, val=%s, test=%s",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    tuner = PSOLightGBMTuner(config)
    best_params = tuner.fit(X_train, y_train)
    model = tuner.train_best_model(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    val_preds = model.predict(X_val)
    report = classification_report(y_val, val_preds, output_dict=True)

    config.paths.models_path.mkdir(parents=True, exist_ok=True)
    config.paths.metrics_path.mkdir(parents=True, exist_ok=True)
    model_path = config.paths.models_path / "lightgbm_pso.pkl"
    metrics_path = config.paths.metrics_path / "validation_report.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Saved model to %s", model_path)
    logger.info("Validation macro F1: %.4f", report["macro avg"]["f1-score"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PSO-tuned LightGBM model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "train"],
        help="Run only a subset of the pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training_pipeline(config, stage=args.stage)


if __name__ == "__main__":
    main()
