"""
Regularized Regression with Hyperparameter Tuning
Dataset : sklearn Diabetes (built-in, no download needed)
Steps   : 1 Load+split | 2 Ridge over alphas | 3 Best alpha via CV
          4 Compare with LinearRegression | 5 Document + save model
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "R2": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
    }


# ---------- Step 1: Load dataset and split data ----------
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"[1] Data loaded: train={X_train.shape}, test={X_test.shape}")

# ---------- Step 2: Fit Ridge with different alpha values ----------
alphas = np.logspace(-3, 3, 13)
rows = []
for a in alphas:
    m = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))])
    m.fit(X_train, y_train)
    rows.append({
        "alpha": a,
        "train_R2": r2_score(y_train, m.predict(X_train)),
        "test_R2": r2_score(y_test, m.predict(X_test)),
    })
alpha_df = pd.DataFrame(rows)
alpha_df.to_csv(OUT / "alpha_sweep.csv", index=False)
print("[2] Alpha sweep done:\n", alpha_df.round(4).to_string(index=False))

# ---------- Step 3: Select best alpha via cross-validation ----------
pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
grid = GridSearchCV(
    pipe,
    param_grid={"ridge__alpha": alphas},
    cv=5,
    scoring="neg_root_mean_squared_error",
)
grid.fit(X_train, y_train)
best_alpha = grid.best_params_["ridge__alpha"]
best_cv_rmse = -grid.best_score_
print(f"[3] Best alpha = {best_alpha:.4f} (5-fold CV RMSE = {best_cv_rmse:.3f})")

# ---------- Step 4: Compare with ordinary LinearRegression ----------
lin = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
lin.fit(X_train, y_train)
best_ridge = grid.best_estimator_

comparison = pd.DataFrame({
    "LinearRegression": evaluate(lin, X_test, y_test),
    f"Ridge (alpha={best_alpha:.3g})": evaluate(best_ridge, X_test, y_test),
}).T
comparison.to_csv(OUT / "comparison.csv")
print("[4] Test-set comparison:\n", comparison.round(4))

# ---------- Step 5: Document results and save final model ----------
joblib.dump(best_ridge, OUT / "final_ridge_model.joblib")

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alpha_df["alpha"], alpha_df["train_R2"], "o-", label="Train R²")
ax.semilogx(alpha_df["alpha"], alpha_df["test_R2"], "s-", label="Test R²")
ax.axvline(best_alpha, color="red", ls="--", label=f"Best alpha = {best_alpha:.3g}")
ax.set_xlabel("alpha (log scale)")
ax.set_ylabel("R²")
ax.set_title("Ridge: effect of alpha")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "alpha_vs_r2.png", dpi=150)

coef_df = pd.DataFrame({
    "LinearRegression": lin.named_steps["lr"].coef_,
    "Ridge": best_ridge.named_steps["ridge"].coef_,
}, index=X.columns)
coef_df.plot.bar(figsize=(8, 4), title="Coefficients: Linear vs Ridge")
plt.tight_layout()
plt.savefig(OUT / "coefficients.png", dpi=150)

report = f"""# Results

- Dataset: sklearn Diabetes ({X.shape[0]} samples, {X.shape[1]} features)
- Split: 80% train / 20% test (random_state={SEED})
- Best alpha (5-fold CV): **{best_alpha:.4f}**
- Best CV RMSE: **{best_cv_rmse:.3f}**

## Test-set comparison

{comparison.round(4).to_markdown()}

## Coefficients

{coef_df.round(3).to_markdown()}

Saved model: `outputs/final_ridge_model.joblib`
"""
(OUT / "results.md").write_text(report)
print("[5] Results + model saved in", OUT)
