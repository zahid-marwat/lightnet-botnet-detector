# Project Overview — PSO-Tuned LightGBM for Robust IoT Botnet Detection

## Institution & Course
- **National University of Sciences & Technology (NUST)**
- **School of Electrical Engineering and Computer Science (SEECS)**
- **Course**: CS-871 Machine Learning (3+0) – MSAI
- **Date**: 24 Oct 2025
- **Group Members**: Zahid Ullah Khan (495181), Sikander Khan (500067)

## Project Title
**PSO-Tuned LightGBM for Robust IoT Botnet Detection**

## Introduction
The proliferation of IoT deployments expands the attack surface for increasingly sophisticated botnets. Lightweight, resource-conscious models are required because many IoT devices lack the compute budget for deep neural networks. This project proposes using **Light Gradient Boosting Machine (LightGBM)** tuned with **Particle Swarm Optimization (PSO)** to deliver a scalable intrusion detection system that balances accuracy, latency, and deployment footprint.

## Background & Prior Work
1. **Upadhyay et al. (2024)** — ANN-based model with 99.99% accuracy but heavier compute profile.
2. **Himani et al. (2023)** — Compared KNN vs. SVM for DDoS mitigation; KNN achieved ~90% accuracy.
3. **Koroniotis et al. (2020)** — Particle Deep Framework (PDF) with PSO tuning; high accuracy yet prohibitive compute requirements.
4. **Infantia et al. (2025)** — Compared RF, Bagging, Gaussian NB; RF ≈93.9% accuracy but limited scalability.
5. **Sharma et al. (2024)** — Benchmarked lightweight models on IoT-PoT dataset; LightGBM led with 99.56% accuracy.

## Table 1 — Properties of Compared Models

| Property | KNN | ANN | LightGBM | Particle Deep Framework |
| --- | --- | --- | --- | --- |
| Scalability for Large IoT Datasets | ✗ | ✓ | ✓ | ✓ |
| Meets Low / Real-Time Requirements | ✗ | ✓ | ✓ | ✓ |
| Low Memory / Lightweight Deployment | ✗ | ✗ | ✓ | ✗ |
| Handles High Dimensionality & Nonlinearity | ✓ | ✓ | ✓ | ✓ |
| Requires Feature Scaling / Normalization | ✓ | ✓ | ✗ | ✓ |
| Handles Class Imbalance Natively | ✗ | ✗ | ✓ | ✗ |
| Provides Interpretability / Feature Importance | ✗ | ✗ | ✓ | ✗ |
| Low Training Complexity / Cost | ✓ | ✗ | ✓ | ✗ |

## Research Gaps in LightGBM Workflows
1. **Hyper-parameter Sensitivity** — Parameters (`num_leaves`, `max_depth`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `min_data_in_leaf`) need smarter tuning strategies such as PSO, Optuna, Bayesian Optimization, or Grid Search.
2. **Dataset Size & Balancing** — Prior work sampled only 120k flows from IoT-PoT while the dataset contains ≈73M samples. Severe label imbalance demands SMOTE/Tomek Links and broader attack sub-category coverage.
3. **Feature Engineering** — 22 raw features were used wholesale, risking overfitting; feature filtering via domain knowledge or feature-importance ranking is needed.

## Table 2 — IoT-PoT Dataset Samples

| Category | Sub-Category | Count |
| --- | --- | --- |
| **DDOS** | Total | 38,532,480 |
| | HTTP | 19,771 |
| | TCP | 19,547,603 |
| | UDP | 18,965,106 |
| **DOS** | Total | 33,005,194 |
| | HTTP | 29,706 |
| | TCP | 12,315,997 |
| | UDP | 20,659,491 |
| **NORMAL** | Total | 9,543 |
| **RECONNAISSANCE** | Total | 1,821,639 |
| | OS Fingerprint | 358,275 |
| | Service Scan | 1,463,364 |
| **THEFT** | Total | 1,587 |
| | Data Exfiltration | 118 |
| | Keylogging | 1,469 |
| **Grand Total** | | **73,370,443** |

## Table 3 — IoT-PoT Feature Inventory

| # | Feature | # | Feature |
| --- | --- | --- | --- |
| 1 | Source IP Address | 12 | Acknowledgment Number |
| 2 | Destination IP Address | 13 | Timestamp |
| 3 | Source Port | 14 | Window Size |
| 4 | Destination Port | 15 | TTL (Time to Live) |
| 5 | Sequence Number | 16 | IP Version |
| 6 | Port Number | 17 | Header Length |
| 7 | Protocol | 18 | IP Fragmentation |
| 8 | Packet Size | 19 | Packet Count |
| 9 | Flow Duration | 20 | IP Options |
| 10 | Payload | 21 | Attack Label |
| 11 | TCP Flags | 22 | Additional Flow Feature |

## References
1. R. Upadhyay et al., "Machine Learning Based Prediction of Malicious Intrusions in IoT Empowered Cybersecurity Application," IEEE, 2024.
2. H. Sivaraman et al., "Performance Evaluation and Analysis of IoT Network using KNN and SVM," IEEE, 2023.
3. M. Koroniotis et al., "A New Network Forensic Framework Based on Deep Learning for Internet of Things Networks: A Particle Deep Framework," IEEE Access 8:116–127, 2020.
4. H. N. Infantia et al., "Empowering IoT Cyber Network Attacks Using Machine Learning," Proc. 8th ICOEI, IEEE, 2025, pp. 90–96.
5. A. Sharma et al., "Detecting IoT Botnets with Advanced Machine Learning Techniques," Proc. 9th ICCES, IEEE, 2024.
