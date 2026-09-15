"""Reproducible Titanic EDA, classification, imbalance, tuning, and regression."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "titanic.csv"
FIG_DIR = ROOT / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
FIG_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_once() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    try:
        df = sns.load_dataset("titanic")
    except Exception as exc:
        raise RuntimeError(
            "titanic.csv is missing and sns.load_dataset('titanic') could not reach the dataset. "
            "Run once with internet access to create the committed fallback."
        ) from exc
    df.to_csv(CSV_PATH, index=False)
    return df


def clean_eda(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Under 5%: drop rows; 5–30%: impute; above 30%: keep as an explicit category.
    if "embarked" in out:
        out = out.dropna(subset=["embarked"])
    if "age" in out:
        out["age"] = out["age"].fillna(out["age"].median())
    if "deck" in out:
        out["deck"] = out["deck"].fillna("Missing")
    return out


def report_eda(df: pd.DataFrame) -> None:
    print("\n=== INFO ===")
    df.info()
    print("\n=== DESCRIBE ===\n", df.describe(include="all"))
    print("\nSHAPE:", df.shape)
    missing = (df.isna().mean().mul(100).round(2)).loc[lambda s: s > 0]
    print("\nMissing percentages:\n", missing.to_string())
    for column in ["age", "fare"]:
        values = df[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        print(f"{column} IQR outliers: {count}")
    fare = df["fare"].dropna()
    mode = fare.mode().iloc[0]
    print(f"Fare mean={fare.mean():.3f}, median={fare.median():.3f}, mode={mode:.3f}; right-skew is expected when mean > median > mode.")

    print("\nSurvival by sex:\n", df.groupby("sex", observed=True)["survived"].mean())
    print("\nSurvival by pclass:\n", df.groupby("pclass")["survived"].mean())
    print("\nSurvival by sex and pclass:\n", df.groupby(["sex", "pclass"], observed=True)["survived"].mean())

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    print("\nRequired six-column correlation matrix:\n", corr)
    pairs = []
    for i, left in enumerate(corr_cols):
        for right in corr_cols[i + 1 :]:
            pairs.append((abs(corr.loc[left, right]), left, right, corr.loc[left, right]))
    print("Two strongest absolute correlations:", sorted(pairs, reverse=True)[:2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Required Titanic numeric correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    for column in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[column], kde=True, ax=axes[0])
        sns.boxplot(x=df[column], ax=axes[1])
        fig.suptitle(f"{column.title()} distribution and outliers")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{column}_distribution.png", dpi=150)
        plt.close(fig)

    charts = [
        (sns.barplot, {"data": df, "x": "sex", "y": "survived", "hue": "pclass"}, "survival_by_sex_class.png", "Women survived at a higher rate than men in each class, while first-class passengers generally had the strongest outcomes. This chart combines gender and class rather than treating them as isolated effects."),
        (sns.boxplot, {"data": df, "x": "survived", "y": "fare", "hue": "sex"}, "fare_by_survival.png", "Survivors tend to have higher fares, especially among women. Fare is a proxy for cabin/class resources, so this supports the class effect without proving causation."),
        (sns.scatterplot, {"data": df, "x": "age", "y": "fare", "hue": "survived", "style": "sex", "alpha": 0.65}, "age_fare_survival.png", "Survival is not explained by age alone: the survivor and non-survivor points overlap. The color separation is more visible at higher fares, suggesting an interaction between age, class proxy, and sex."),
        (sns.countplot, {"data": df, "x": "pclass", "hue": "survived"}, "class_counts_survival.png", "The passenger population is concentrated in lower classes, but the survival counts and rates differ sharply by class. Reading this with the grouped rates avoids mistaking group size for probability."),
    ]
    for plot_fn, kwargs, filename, interpretation in charts:
        plt.figure(figsize=(7, 4))
        plot_fn(**kwargs)
        plt.title(interpretation.split(".")[0])
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=150)
        plt.close()
        print(f"\nChart interpretation ({filename}): {interpretation}")

    standardized = df[["age", "fare"]].copy()
    standardized = (standardized - standardized.mean()) / standardized.std()
    print("\nZ-score means:\n", standardized.mean(), "\nZ-score stds:\n", standardized.std())


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, ["age", "sibsp", "parch", "fare"]), ("categorical", categorical, ["sex", "embarked"])])


def classification(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Pipeline]:
    features = ["age", "sibsp", "parch", "fare", "sex", "embarked"]
    model_df = df[features + ["survived"]].copy()
    X, y = model_df[features], model_df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nClass balance:\n", y.value_counts(normalize=True).rename("proportion"))
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    rows = []
    plt.figure(figsize=(8, 6))
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, prob):.3f}")
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "auc": roc_auc_score(y_test, prob)})
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["not survived", "survived"]).plot()
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()
        if name == "Decision Tree":
            transformed = pipeline.named_steps["preprocess"].transform(X_train)
            names = pipeline.named_steps["preprocess"].get_feature_names_out()
            plt.figure(figsize=(18, 9))
            plot_tree(estimator, feature_names=names, class_names=["not survived", "survived"], filled=True, max_depth=3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "decision_tree.png", dpi=150)
            plt.close()
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("Classifier ROC comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_comparison.png", dpi=150)
    plt.close()
    metrics = pd.DataFrame(rows).set_index("model")
    print("\nClassifier comparison:\n", metrics)

    base = Pipeline([("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    balanced = Pipeline([("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    smote = ImbPipeline([("preprocess", make_preprocessor()), ("smote", SMOTE(random_state=42)), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    imbalance_rows = []
    for label, pipe in [("baseline", base), ("class_weight_balanced", balanced), ("SMOTE_train_only", smote)]:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        imbalance_rows.append({"variant": label, "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred)})
    imbalance = pd.DataFrame(imbalance_rows).set_index("variant")
    print("\nImbalance comparison (SMOTE is inside the training pipeline):\n", imbalance)

    grid = GridSearchCV(
        Pipeline([("preprocess", make_preprocessor()), ("model", RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True))]),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10], "model__max_features": ["sqrt", "log2"]},
        cv=5, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_pipeline: Pipeline = grid.best_estimator_
    print("Grid best params:", grid.best_params_)
    print("OOB score:", best_pipeline.named_steps["model"].oob_score_)
    joblib.dump(best_pipeline, ARTIFACT_DIR / "best_pipeline.joblib")
    reloaded = joblib.load(ARTIFACT_DIR / "best_pipeline.joblib")
    print("Reloaded pipeline prediction on raw row:", reloaded.predict(X_test.iloc[[0]]).tolist())
    return metrics, imbalance, X_test, best_pipeline


def regression(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["fare", "survived", "pclass", "age", "sibsp", "parch", "sex", "embarked"]].dropna().copy()
    X = pd.get_dummies(work.drop(columns="fare"), columns=["sex", "embarked"], drop_first=True)
    y = work["fare"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    n, p = len(y_test), X_test.shape[1]
    r2 = r2_score(y_test, pred)
    metrics = pd.DataFrame([{"MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred, squared=False), "R2": r2, "Adjusted_R2": 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)}])
    residuals = y_test - pred
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=pred, y=residuals)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("predicted fare")
    plt.ylabel("residual")
    plt.title("Fare residual plot — inspect for non-random spread/heteroscedasticity")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fare_residuals.png", dpi=150)
    plt.close()
    print("\nRegression metrics:\n", metrics)
    print("Residual conclusion: inspect the saved residual plot; a funnel-shaped spread indicates heteroscedasticity, while an even cloud supports constant variance.")
    return metrics


def main() -> None:
    df = load_once()
    report_eda(df)
    cleaned = clean_eda(df)
    clf, imbalance, _, _ = classification(cleaned)
    reg = regression(cleaned)
    summary = {"classification": clf.reset_index().to_dict(orient="records"), "imbalance": imbalance.reset_index().to_dict(orient="records"), "regression": reg.to_dict(orient="records")}
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFinal recommendation: deploy the classifier with the strongest validation F1/AUC balance; review the generated metrics table and choose the model whose false-negative trade-off fits the product risk.")


if __name__ == "__main__":
    main()

