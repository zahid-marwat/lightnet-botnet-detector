"""Evaluation pipeline for PSO-tuned LightGBM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config
from src.pipelines.training import _load_processed_or_run
from src.utils.logger import setup_logger


def run_evaluation(config: ProjectConfig) -> None:
    logger = setup_logger(log_dir=config.paths.logs_dir)
    df = _load_processed_or_run(config, logger)
    label_col = config.data.label_column
    X = df.drop(columns=[label_col])
    y = df[label_col]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        stratify=y,
        random_state=config.training.random_state,
    )

    model_path = config.paths.models_path / "lightgbm_pso.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run training first.")

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
    }

    metrics_path = config.paths.metrics_path / "test_report.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Test macro F1: %.4f", report["macro avg"]["f1-score"])
    logger.info("Saved evaluation metrics to %s", metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PSO-tuned LightGBM on held-out test data")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_evaluation(config)


if __name__ == "__main__":
    main()
