"""Configuration dataclasses and loader for the PSO-LightGBM project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import yaml


@dataclass
class DataConfig:
    raw_glob: str
    processed_file: str
    label_column: str
    categorical_features: List[str]
    numeric_features: List[str]
    sample_frac: float = 1.0
    chunk_size: int = 1_000_000
    target_classes: List[str] | None = None


@dataclass
class FeatureConfig:
    top_k: int
    importance_threshold: float
    variance_threshold: float


@dataclass
class PSOSearchSpace:
    num_leaves: List[int]
    max_depth: List[int]
    learning_rate: List[float]
    feature_fraction: List[float]
    bagging_fraction: List[float]
    min_data_in_leaf: List[int]


@dataclass
class TrainingConfig:
    test_size: float
    val_size: float
    random_state: int
    n_particles: int
    n_iterations: int
    cv_folds: int
    scoring: str
    early_stopping_rounds: int
    n_estimators: int = 800
    use_gpu: bool = False
    gpu_platform_id: int = 0
    gpu_device_id: int = 0
    show_progress: bool = False


@dataclass
class PathsConfig:
    artifacts_dir: str
    models_dir: str
    metrics_dir: str
    logs_dir: str

    def ensure(self) -> None:
        for folder in (self.artifacts_path, self.models_path, self.metrics_path, self.logs_path):
            folder.mkdir(parents=True, exist_ok=True)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def metrics_path(self) -> Path:
        return Path(self.metrics_dir)

    @property
    def logs_path(self) -> Path:
        return Path(self.logs_dir)


@dataclass
class ProjectConfig:
    data: DataConfig
    features: FeatureConfig
    search_space: PSOSearchSpace
    training: TrainingConfig
    paths: PathsConfig


def _expand_path_config(raw: Dict[str, str]) -> PathsConfig:
    root = Path(__file__).resolve().parents[1]
    return PathsConfig(
        artifacts_dir=str(root / raw["artifacts_dir"]),
        models_dir=str(root / raw["models_dir"]),
        metrics_dir=str(root / raw["metrics_dir"]),
        logs_dir=str(root / raw["logs_dir"]),
    )


def load_config(config_path: Union[str, Path]) -> ProjectConfig:
    """Load YAML config and cast into strongly-typed dataclasses."""

    with Path(config_path).open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)

    data_cfg = DataConfig(**raw_cfg["data"])
    feature_cfg = FeatureConfig(**raw_cfg["features"])
    search_space = PSOSearchSpace(**raw_cfg["search_space"])
    training_cfg = TrainingConfig(**raw_cfg["training"])
    paths_cfg = _expand_path_config(raw_cfg["paths"])
    paths_cfg.ensure()

    return ProjectConfig(
        data=data_cfg,
        features=feature_cfg,
        search_space=search_space,
        training=training_cfg,
        paths=paths_cfg,
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_default.yaml"
