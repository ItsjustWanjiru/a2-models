# a2-models

DSA 8401 MSc Data Science and Analytics, Assignment 2: The Cost of Being Wrong.

Extends the Assignment 1 fraud pipeline (`lab2_extended.ipynb`, Sections 1 to 5) with a
cost-sensitive evaluation of `is_fraud` as the adverse outcome: purged/blocked time-series
CV, a model ladder, imbalance handling, Optuna tuning, calibration, and SHAP/fairness
(Sections 6 to 13 of the same notebook).

## Contents

| File | What it is |
|---|---|
| `lab2_extended.ipynb` | The full, executed notebook. Sections 1 to 5 are Assignment 1; Sections 6 to 13 are Assignment 2. |
| `mobile_money_statements.csv` | Raw input data (180,000 rows). |
| `Assignment2_Report.docx` | The 4 page written report summarising Sections 6 to 13. |
| `requirements.txt` | Pinned package versions used to run the notebook. |
| `model_comparison.csv` | The Section 7 model ladder table (LogReg, RandomForest, XGBoost; PR-AUC and ROC-AUC). |
| `pr_curves.png` | Out-of-fold PR curves and PR-AUC by model (Section 7, Figure 1 in the report). |
| `calibration_curve.png` | Reliability diagram, raw vs calibrated (Section 11, Figure 2 in the report). |
| `optuna_study.db` | The SQLite Optuna study, 60 trials, reopenable with `optuna.load_study(study_name="a2_xgb_tuning", storage="sqlite:///optuna_study.db")` (Section 10). |

Running the notebook end to end also regenerates all of the above plus:

- `fraud_pipeline.joblib` — the honest, 17-feature Assignment 1 pipeline, persisted and reloaded (Section 5d)
- `optuna_trials.csv` — the Optuna study as a flat trials table

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute lab2_extended.ipynb --output lab2_extended.ipynb --ExecutePreprocessor.timeout=1800
```

A fresh run takes roughly 15 to 20 minutes, most of it in the Section 7 model ladder,
the Section 8 SMOTE leak demo, and the Section 10 Optuna study (60 trials, 2 folds
each, pruned). Everything else finishes in seconds. Re-running is deterministic
(`random_state=42` throughout, Optuna `TPESampler(seed=42)`); exact numbers may drift
by a few thousandths of a point across scikit-learn/xgboost versions, not more.

## Where each grading item lives

- **Evaluation plan** — Section 6 (`purged_blocked_splits`, fold report table)
- **Ensemble models** — Section 7 (model ladder table, PR curves)
- **Cost-based decisions** — Section 9 (cost sweep, threshold vs naive 0.5)
- **Imbalance and tuning** — Sections 8 to 10 (SMOTE in-fold vs leaky, Optuna study)
- **Calibration** — Section 11 (reliability diagram, Brier score, recomputed threshold)
- **Explainability and fairness** — Section 12 (SHAP, region/segment subgroup tables)
- **Reproducibility and report** — this README, `requirements.txt`, `Assignment2_Report.docx`

## Known limitation

Assignment 1's RFM features are computed once per customer over that customer's entire
pre `SCORING_TS` history, so they carry some hindsight bias into early-period rows.
Section 6 measures this directly (customer overlap between purged train/test folds is
99.5 to 99.9%) and the report treats it as a disclosed limitation rather than something
the CV design can fully remove without changing Assignment 1's features, which the brief
does not allow. See Section 6 and the Reproducibility section of the report for detail.
