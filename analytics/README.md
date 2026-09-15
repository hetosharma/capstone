# Module 2 — Analytics and Modeling

`pipeline.py` performs the full EDA/modeling workflow in one deterministic script. It calls `sns.load_dataset("titanic")` at most once, immediately saves the result to `analytics/titanic.csv`, and uses that CSV on later runs if the network/cache is unavailable.

The workflow reports shape, `info`, descriptive statistics, missing percentages, threshold-based cleaning, IQR outlier counts, fare mean/median/mode and skew, sex/class survival rates, the required six-column correlation matrix and strongest pairs, four data-story figures, z-score checks, a stratified split, train-only preprocessing in a `ColumnTransformer`, three classifiers, confusion matrices and ROC/AUC, imbalance comparisons including train-fold-only SMOTE, Random Forest `GridSearchCV` with `oob_score=True`, and a linear regression side-task with MAE/RMSE/R²/adjusted R² and residual diagnostics.

Run:

```bash
python analytics/pipeline.py
```

Generated charts and `artifacts/best_pipeline.joblib` are reproducible outputs. The saved classifier artifact is the complete preprocessing-plus-estimator pipeline and can predict from raw columns.


