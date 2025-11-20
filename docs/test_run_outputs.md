# LightNet Botnet Detector Test Run Explainer

This document walks through the recent notebook run step-by-step and translates the console output into plain English insights.

## 1. Data Intake
- **Message:** `Loaded 314,055 rows from ... Dataset(Over Sampled).csv`
- **Meaning:** We pulled a spreadsheet-like file containing 314,055 network traffic records. Each row logs how a device communicated over the network.
- **Label counts:**
  - `attack    304,834`
  - `normal      9,221`
- **Interpretation:** The file is heavily imbalanced: attacks make up ~97% of rows while normal traffic is only ~3%. Training directly on such skewed data would bias the model toward always predicting "attack".

## 2. Balancing the Dataset
- **Message:** `Balanced subset size: 18,442` with `9,221` attack and `9,221` normal rows.
- **Meaning:** To make the learning fair, we sampled an equal number of attack and normal rows (9,221 each). This balanced subset is what the model actually sees during training/testing, and the total size matches 18,442 rows.

## 3. Train / Validation / Test Split
- **Message:** `Splits -> train: 13,277, val: 1,476, test: 3,689`
- **Meaning:**
  - **Train set (13,277 rows):** Used to let the algorithm learn patterns.
  - **Validation set (1,476 rows):** Used during tuning to double-check whether new settings genuinely help.
  - **Test set (3,689 rows):** Held back until the end to get an unbiased performance estimate.

## 4. PSO Tuning Status Messages
Example snippet:
```
2025-11-19 12:59:44 | INFO | psolgbm | Iteration 1/25 | best f1_macro = 0.9999
...
2025-11-19 13:04:49 | INFO | psolgbm | Best params: {...}
```
- **Meaning:** Particle Swarm Optimization (PSO) tried 25 rounds of hyperparameter combinations for LightGBM. After each round the script logs the best "macro F1" score it has seen so far. Macro F1 combines precision and recall for both classes equally. A value of `0.9999` indicates near-perfect balance between correctly detecting attacks and normal traffic on the validation folds.
- **Best parameters:** The PSO search settled on a LightGBM configuration with values such as `num_leaves: 94`, `max_depth: 10`, etc. These control tree complexity, learning speed, and data sampling.

## 5. LightGBM Warnings
Example warnings:
```
[LightGBM] [Warning] min_data_in_leaf is set=429, min_child_samples=20 will be ignored.
```
- **Meaning:** LightGBM offers multiple ways to control the same behaviour. When both settings are supplied, it keeps one and ignores the redundant one. These warnings just clarify which switch won; they do **not** indicate a failure.

## 6. MLP-Inspired Overrides
- **Message:** `Best hyperparameters with MLP overrides ... {'num_leaves': 200, 'learning_rate': 0.004174, 'reg_alpha': 0.0001, ...}`
- **Meaning:** After PSO finished, we manually forced three values inspired by a previous MLP experiment: more tree leaves, a slower learning rate, and extra regularization. The rest of the PSO-selected settings (depth, sampling fractions, etc.) stay in place.

## 7. Training Diagnostics
- The LightGBM info lines (`Number of data points`, `Start training from score ...`, `No further splits ...`) describe how the model builds decision trees internally. They simply confirm the algorithm finished without crashing.

## 8. Final Test Results
```
Test macro F1: 0.9984
precision  recall  f1-score  support
attack  0.9968  1.0000  0.9984  1,845
normal  1.0000  0.9967  0.9984  1,844
accuracy 0.9984 (on 3,689 samples)
```
- **Interpretation:**
  - **Accuracy 99.84%:** Out of 3,689 unseen rows, the model misclassified roughly six events.
  - **Precision vs. recall:** Both attack and normal classes scored 99.6%+ on precision (few false alarms) and recall (rarely missed true events).
  - **Macro F1 0.9984:** Average of the two class F1 scores; indicates balanced excellence rather than one class dominating.

## 9. Key Takeaways for Non-Experts
1. **Balanced training** fixed the original skew, so the model treats attacks and normal traffic equally.
2. **Tuning** iteratively searched for the best LightGBM configuration and then blended in trusted MLP-inspired tweaks.
3. **Warnings** are informational and just tell us which duplicate settings LightGBM chose to honour.
4. **Performance** on the held-out test slice is effectively perfect, showing the pipeline can both detect attacks and avoid false alarms on this dataset. Future checks should still include fresh, real-world traffic to ensure the model generalizes beyond the lab sample.
