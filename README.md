# PSO-Tuned LightGBM for Robust IoT Botnet Detection

> **Course**: CS-871 Machine Learning (MSAI) · SEECS, NUST  \
> **Date**: 24 Oct 2025  \
> **Team**: Zahid Ullah Khan (495181), Sikander Khan (500067)

## 1. Motivation

IoT infrastructures confront massive, heterogeneous traffic that makes them vulnerable to botnet attacks. Heavy deep-learning models struggle to run on constrained devices, so we investigate **LightGBM** enhanced with **Particle Swarm Optimization (PSO)** for hyper-parameter tuning. The objective is to deliver an intrusion detection workflow that remains lightweight, interpretable, and deployable across real-world IoT edge scenarios.

## 2. Repository Layout

```
ML Project/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment_default.yaml
├── data/
│   ├── raw/                # place IoT-PoT CSV chunks or parquet files here
│   └── processed/          # auto-generated balanced datasets + feature stores
├── artifacts/
│   └── models/             # persisted LightGBM models + PSO search logs
├── logs/                   # training & evaluation logs
├── reports/                # figures (confusion matrices, feature importances)
├── docs/
│   └── project_overview.md
├── src/
│   ├── config.py
│   ├── data/
│   │   └── preprocessing.py
│   ├── features/
│   │   └── selection.py
│   ├── models/
│   │   └── pso_lightgbm.py
│   ├── pipelines/
│   │   ├── training.py
│   │   └── evaluation.py
│   └── utils/
│       ├── logger.py
│       └── paths.py
├── tests/
│   └── test_config.py
├── 1. Pre_processing.ipynb
├── 2. Training.ipynb
└── 3. Testing and evaluation.ipynb
```

> Existing folders `Raw data sets/` and `processed Data sets/` remain untouched. Copy/move their contents into `data/raw/` and `data/processed/` respectively when running the automated pipelines.

## 3. Dataset Expectations

| Aspect | Details |
| --- | --- |
| Source | IoT-PoT (73M+ labelled flows) |
| Mandatory Columns | Listed in `docs/project_overview.md` (Table 3) |
| Label | `attack_label` (`NORMAL`, `DDOS_HTTP`, …) |
| Imbalance Handling | SMOTETomek + optional undersampling |
| Feature Selection | Top-k importance via Random Forest + manual filtering |

Because the full dataset is massive, the repo defaults to **chunked ingestion**. You can specify a `sample_frac` in the config to downsample during prototyping.

## 4. Workflow Overview

1. **Pre-processing (Notebook `1. Pre_processing.ipynb`)**
   - Load chunks from `data/raw/`.
   - Clean header anomalies, drop duplicates, engineer network-flow ratios.
   - Encode categorical features (IP octets, protocol) using target-aware encoders.
   - Balance data with SMOTETomek.
   - Persist balanced dataset to `data/processed/iot_pot_balanced.parquet` and artifact metadata JSON.

2. **Training (Notebook `2. Training.ipynb` or `python -m src.pipelines.training`)**
   - Split processed data into train/validation/test.
   - Run PSO search over LightGBM hyper-parameters.
   - Produce model card, feature importance plot, and save model to `artifacts/models/lightgbm_pso.pkl`.

3. **Testing & Evaluation (Notebook `3. Testing and evaluation.ipynb` or `python -m src.pipelines.evaluation`)**
   - Reload saved artifacts.
   - Score held-out test set, compute metrics (precision/recall/F1, ROC-AUC).
   - Generate confusion matrix, per-class performance table, and threshold analysis.

## 5. Quickstart

### 5.1. Environment

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=psolgbm
```

### 5.2. Configuration

Edit `configs/experiment_default.yaml` to point to your raw data paths or adjust PSO search ranges.

### 5.3. CLI Pipelines

```bash
python -m src.pipelines.training --config configs/experiment_default.yaml
python -m src.pipelines.evaluation --config configs/experiment_default.yaml
```

### 5.4. Notebook Flow

Open the notebooks in order (`1.` → `2.` → `3.`). Each notebook mirrors the CLI steps and automatically reloads helper modules from `src/`.

## 6. Research Traceability

`docs/project_overview.md` stores the full proposal (motivation, background, references, Tables 1–3, and research gaps) so the narrative remains version-controlled.

## 7. Next Steps

- Integrate adversarial validation and drift detection for streaming IoT traffic.
- Package the PSO tuner as a reusable pip module.
- Add CI that lints notebooks (`nbqa black`, `ruff`) and runs smoke tests on a 1% IoT-PoT sample.
