# Regularized Regression with Hyperparameter Tuning

A simple scikit-learn project that fits **Ridge regression**, tunes `alpha` with cross-validation, and compares it against ordinary **LinearRegression**.

- **Dataset:** sklearn Diabetes (442 samples, 10 features) — built in, no download needed
- **Task:** predict disease progression (regression)

## Project Structure

```
ridge_regression_project/
├── README.md
├── requirements.txt
├── src/
│   └── train.py              # full pipeline (all 5 steps)
└── outputs/                  # generated after running
    ├── alpha_sweep.csv       # train/test R² for each alpha
    ├── comparison.csv        # Linear vs Ridge test metrics
    ├── results.md            # auto-generated results report
    ├── alpha_vs_r2.png       # alpha vs R² plot
    ├── coefficients.png      # coefficient comparison plot
    └── final_ridge_model.joblib
```

## Setup & Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

## Implementation Steps

| # | Step | How it's done |
|---|------|---------------|
| 1 | Load dataset and split data | `load_diabetes`, 80/20 `train_test_split` (random_state=42) |
| 2 | Fit Ridge with different alpha values | 13 alphas from 1e-3 to 1e3 (`np.logspace`), with `StandardScaler` in a Pipeline |
| 3 | Select best alpha via cross-validation | `GridSearchCV`, 5-fold, scoring = negative RMSE |
| 4 | Compare with ordinary LinearRegression | Same pipeline, test-set R², RMSE, MAE |
| 5 | Document results and save final model | `results.md`, plots, `joblib.dump` of the best model |

## Results (test set)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| LinearRegression | 0.4526 | 53.85 | 42.79 |
| Ridge (alpha ≈ 31.6) | 0.4599 | 53.49 | 42.95 |

Best alpha from 5-fold CV: **31.62**.

**Takeaway:** Ridge gives a small improvement in R² and RMSE over plain LinearRegression. The dataset is small and low-dimensional, so the gain is modest. Regularization matters more when features are many or correlated. MAE is marginally higher for Ridge, so the gain is not uniform across metrics.

## Use the Saved Model

```python
import joblib
from sklearn.datasets import load_diabetes

model = joblib.load("outputs/final_ridge_model.joblib")
X, _ = load_diabetes(return_X_y=True, as_frame=True)
print(model.predict(X.head()))
```

The scaler is part of the saved pipeline, so raw features can be passed directly.

## Notes

- Features are standardized before Ridge, because the L2 penalty is scale-sensitive.
- Scaling is fit inside the pipeline, so there is no data leakage during cross-validation.
