"""Particle Swarm Optimization wrapper for LightGBM."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from lightgbm import LGBMClassifier
from lightgbm.callback import CallbackEnv
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.config import ProjectConfig
from src.utils.logger import setup_logger


@dataclass
class SwarmState:
    positions: np.ndarray
    velocities: np.ndarray
    personal_best_positions: np.ndarray
    personal_best_scores: np.ndarray
    global_best_position: np.ndarray
    global_best_score: float


class PSOLightGBMTuner:
    """Basic PSO optimizer to fine-tune LightGBM hyperparameters."""

    def __init__(
        self,
        config: ProjectConfig,
        inertia: float = 0.7,
        cognitive_coef: float = 1.5,
        social_coef: float = 1.5,
    ) -> None:
        self.config = config
        self.scoring_name = config.training.scoring
        self.inertia = inertia
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.use_gpu = config.training.use_gpu
        self.gpu_platform_id = config.training.gpu_platform_id
        self.gpu_device_id = config.training.gpu_device_id
        self.show_progress = config.training.show_progress
        self.logger = setup_logger(log_dir=config.paths.logs_dir)
        self.rng = np.random.default_rng(config.training.random_state)
        self.param_keys = [
            "num_leaves",
            "max_depth",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "min_data_in_leaf",
        ]
        self.bounds = self._build_bounds()
        self.best_params_: Dict[str, float | int] | None = None
        self.best_estimator_: LGBMClassifier | None = None
        self.checkpoint_paths_: List[Path] = []

    def _build_bounds(self) -> np.ndarray:
        search_space = self.config.search_space
        bounds = np.array(
            [
                search_space.num_leaves,
                search_space.max_depth,
                search_space.learning_rate,
                search_space.feature_fraction,
                search_space.bagging_fraction,
                search_space.min_data_in_leaf,
            ],
            dtype=float,
        )
        return bounds

    def _init_swarm(self) -> SwarmState:
        n_particles = self.config.training.n_particles
        dimensions = len(self.param_keys)
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        positions = self.rng.uniform(low, high, size=(n_particles, dimensions))
        velocities = np.zeros_like(positions)
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(n_particles, -np.inf)
        global_best_position = positions[0].copy()
        global_best_score = -np.inf
        return SwarmState(
            positions,
            velocities,
            personal_best_positions,
            personal_best_scores,
            global_best_position,
            global_best_score,
        )

    def _vector_to_params(self, vector: np.ndarray, n_classes: int) -> Dict[str, float | int]:
        space = dict(zip(self.param_keys, vector))
        params = {
            "num_leaves": int(round(space["num_leaves"])),
            "max_depth": int(round(space["max_depth"])),
            "learning_rate": float(space["learning_rate"]),
            "feature_fraction": float(space["feature_fraction"]),
            "bagging_fraction": float(space["bagging_fraction"]),
            "min_data_in_leaf": int(round(space["min_data_in_leaf"])),
        }
        params["max_depth"] = max(3, params["max_depth"])
        params["num_leaves"] = max(8, params["num_leaves"])
        params["min_data_in_leaf"] = max(5, params["min_data_in_leaf"])

        objective = "binary"
        extra_args: Dict[str, float | int] = {}
        if n_classes > 2:
            objective = "multiclass"
            extra_args["num_class"] = n_classes
        params.update(
            {
                "objective": objective,
                "n_estimators": self.config.training.n_estimators,
                "random_state": self.config.training.random_state,
                # Reduce CPU contention when GPU acceleration is enabled.
                "n_jobs": 1 if self.use_gpu else -1,
                "class_weight": "balanced",
                "verbosity": -1,
                **extra_args,
            }
        )
        params["min_child_samples"] = params["min_data_in_leaf"]
        if self.use_gpu:
            params.update(
                {
                    "device_type": "gpu",
                    "gpu_platform_id": self.gpu_platform_id,
                    "gpu_device_id": self.gpu_device_id,
                    "gpu_use_dp": False,
                    "max_bin": 255,
                    "bin_construct_sample_cnt": 2_000_000,
                    "force_col_wise": True,
                }
            )
            params["subsample_for_bin"] = params["bin_construct_sample_cnt"]
        return params

    def _make_checkpoint_callback(self, prefix: str) -> Callable[[CallbackEnv], None]:
        save_dir = self.config.paths.models_path
        save_dir.mkdir(parents=True, exist_ok=True)

        def _callback(env: CallbackEnv) -> None:
            iteration = env.iteration + 1
            filename = f"{prefix}_epoch{iteration:04d}.txt"
            destination = save_dir / filename
            env.model.save_model(str(destination), num_iteration=iteration)
            self.checkpoint_paths_.append(destination)

        _callback.order = 10
        return _callback

    def _evaluate_particle(self, X, y, vector: np.ndarray) -> float:
        params = self._vector_to_params(vector, n_classes=len(np.unique(y)))
        model = LGBMClassifier(**params)
        cv_n_jobs = 1 if self.use_gpu else -1
        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.config.training.cv_folds,
            scoring=self.scoring_name,
            n_jobs=cv_n_jobs,
        )
        return float(scores.mean())

    def fit(self, X, y) -> Dict[str, float | int]:
        state = self._init_swarm()
        iterations = self.config.training.n_iterations
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        iterator = range(iterations)
        if self.show_progress:
            iterator = tqdm(iterator, desc="PSO tuning", unit="iter")

        for iteration in iterator:
            for idx in range(self.config.training.n_particles):
                vector = state.positions[idx]
                score = self._evaluate_particle(X, y, vector)
                if score > state.personal_best_scores[idx]:
                    state.personal_best_scores[idx] = score
                    state.personal_best_positions[idx] = vector.copy()
                if score > state.global_best_score:
                    state.global_best_score = score
                    state.global_best_position = vector.copy()

            r1 = self.rng.random(size=state.positions.shape)
            r2 = self.rng.random(size=state.positions.shape)
            cognitive = self.cognitive_coef * r1 * (state.personal_best_positions - state.positions)
            social = self.social_coef * r2 * (state.global_best_position - state.positions)
            state.velocities = self.inertia * state.velocities + cognitive + social
            state.positions = state.positions + state.velocities
            state.positions = np.clip(state.positions, low, high)

            self.logger.info(
                "Iteration %s/%s | best %s = %.4f",
                iteration + 1,
                iterations,
                self.scoring_name,
                state.global_best_score,
            )

        self.best_params_ = self._vector_to_params(state.global_best_position, len(np.unique(y)))
        self.logger.info("Best params: %s", self.best_params_)
        return self.best_params_

    def train_best_model(
        self,
        X,
        y,
        *,
        n_estimators: int | None = None,
        save_each_epoch: bool = False,
        checkpoint_prefix: str | None = None,
    ) -> LGBMClassifier:
        if self.best_params_ is None:
            raise RuntimeError("fit() must be called before training the final model")

        params = self.best_params_.copy()
        if n_estimators is not None:
            params["n_estimators"] = n_estimators

        model = LGBMClassifier(**params)
        callbacks = []
        if save_each_epoch:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = checkpoint_prefix or f"pso_lightgbm_{timestamp}"
            self.checkpoint_paths_ = []
            callbacks.append(self._make_checkpoint_callback(prefix))

        fit_kwargs = {"callbacks": callbacks} if callbacks else {}
        model.fit(X, y, **fit_kwargs)
        self.best_estimator_ = model
        return model


def run_pso_lightgbm(X, y, config: ProjectConfig) -> Tuple[LGBMClassifier, Dict[str, float | int]]:
    tuner = PSOLightGBMTuner(config)
    best_params = tuner.fit(X, y)
    model = tuner.train_best_model(X, y)
    return model, best_params
