#!/usr/bin/env python3
"""
Assignment 4 — Heart Disease Classification Analysis
University of Naples Federico II — Statistical Data Analysis
Student: Mohammad Waliullah Shah | ID: P37000087
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Non-interactive backend for saving figures in any environment
plt.switch_backend("Agg")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths & constants ─────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"
PLOT_DIR = BASE_DIR / "eda_plots"
REPORT_PATH = BASE_DIR / "assignment4_report.txt"

CATEGORICAL_COLS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

PAIR_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Short teaching lines printed after each model (student voice)
WHY_LR = (
    "Think of logistic regression like a dimmer switch: it combines your features "
    "with weights and pushes the score through a curve that outputs a probability "
    "of disease. It is the clean linear baseline everyone compares against."
)
PROF_LR = (
    "Markers want to see that you understand linear decision boundaries, "
    "interpretability, and that you report more than accuracy on imbalanced data."
)

WHY_LDA = (
    "LDA is like drawing the straightest fair boundary between two clouds of points "
    "when each cloud is roughly bell-shaped with similar spread. It borrows strength "
    "across features by assuming a shared covariance structure."
)
PROF_LDA = (
    "They check that you know LDA needs approximate normality and homogeneity of "
    "covariance across classes, and that it is a generative linear classifier."
)

WHY_QDA = (
    "QDA is like giving each class its own stretchy shape: if one disease group "
    "varies more in some directions, QDA uses class-specific covariance matrices. "
    "It can curve the boundary because the separation rule becomes quadratic."
)
PROF_QDA = (
    "They look for the contrast with LDA: when variances differ, QDA is more "
    "flexible, but it needs enough data to estimate more parameters and can overfit."
)

WHY_TREE = (
    "A decision tree is a flowchart of yes/no questions. It catches non-linear patterns "
    "and interactions without assuming a straight line; depth 5 caps complexity so "
    "we do not memorize noise."
)
PROF_TREE = (
    "They expect feature importances, a discussion of overfitting vs depth, and "
    "honesty that trees do not need scaling but we used scaled inputs for a fair "
    "comparison with linear models."
)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load heart.csv."""
    return pd.read_csv(path)


def impute_ca_thal_median(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing ca / thal with column medians."""
    out = df.copy()
    imputer = SimpleImputer(strategy="median")
    out[["ca", "thal"]] = imputer.fit_transform(out[["ca", "thal"]])
    return out


def add_binary_target(df: pd.DataFrame, col: str = "target") -> pd.DataFrame:
    """0 = no disease, 1 = disease (labels 1–4 collapsed)."""
    out = df.copy()
    out["target_binary"] = (out[col] > 0).astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Predictors only (exclude original multiclass target and binary label)."""
    return [c for c in df.columns if c not in ("target", "target_binary")]


# ── 1. EDA ────────────────────────────────────────────


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of binary disease vs no disease."""
    counts = df["target_binary"].value_counts().sort_index()
    labels = ["No disease (0)", "Disease (1)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Number of patients")
    ax.set_title("Class distribution (binary target)")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 2, str(int(v)), ha="center", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pie_target_binary(df: pd.DataFrame, out_path: Path) -> None:
    """Pie chart of class shares."""
    counts = df["target_binary"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=["No disease", "Disease"],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
        startangle=90,
    )
    ax.set_title("Share of each class (binary target)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Pearson correlation among features and binary outcome."""
    fcols = feature_columns(df)
    corr = df[fcols + ["target_binary"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Feature correlation matrix (Pearson)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_grid(df: pd.DataFrame, out_path: Path) -> None:
    """Pair plot on a readable subset of variables."""
    plot_df = df[PAIR_FEATURES + ["target_binary"]].copy()
    g = sns.pairplot(
        plot_df,
        hue="target_binary",
        diag_kind="hist",
        plot_kws={"alpha": 0.6, "s": 25},
        corner=False,
    )
    g.fig.suptitle(
        "Pair plot (selected features; hue = binary disease)",
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()
    g.savefig(out_path, dpi=120)
    plt.close("all")


def plot_bar_charts_categorical(df: pd.DataFrame, out_path: Path) -> None:
    """Bar charts for coded categorical attributes vs disease."""
    n = len(CATEGORICAL_COLS)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, CATEGORICAL_COLS):
        ct = (
            df.groupby([col, "target_binary"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], width=0.85)
        ax.set_title(f"{col} vs disease (binary)")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(["No disease", "Disease"], title="Class")
        ax.tick_params(axis="x", rotation=0)
    for j in range(len(CATEGORICAL_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """EDA figures 01–05; returns dataframe ready for modeling."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df_imp = impute_ca_thal_median(df)
    df_eda = add_binary_target(df_imp)
    plot_class_distribution(df_eda, PLOT_DIR / "01_class_distribution.png")
    plot_pie_target_binary(df_eda, PLOT_DIR / "02_pie_chart_target.png")
    plot_correlation_heatmap(df_eda, PLOT_DIR / "03_correlation_heatmap.png")
    plot_pair_grid(df_eda, PLOT_DIR / "04_pair_plot.png")
    plot_bar_charts_categorical(df_eda, PLOT_DIR / "05_bar_charts_categorical.png")
    return df_eda


# ── 2. PREPROCESSING ────────────────────────────────


def run_preprocessing(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Train/test split (stratified) and StandardScaler on all features.
    Categoricals are already numerically coded as in the UCI-style file; scaling puts
    every column on a comparable scale for LR / LDA / QDA (trees are insensitive but
    we keep one shared matrix for fair comparison).
    """
    feats = feature_columns(df)
    X = df[feats].to_numpy(dtype=float)
    y = df["target_binary"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, feats


# ── 3. MODEL BUILDING ───────────────────────────────


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall, F1 for the positive class (disease)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }


def _print_model_block(
    title: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    why: str,
    prof: str,
) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"\nWhy this model:\n  {why}")
    print(f"\nWhat the professor looks for:\n  {prof}\n")


def run_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Fit LR, LDA, QDA, CART with class balancing.
    LR and CART use class_weight='balanced'. For LDA/QDA we prefer balanced
    sample_weight when supported; otherwise equal priors (0.5, 0.5).
    """
    balanced_priors = np.array([0.5, 0.5])
    sw = compute_sample_weight(class_weight="balanced", y=y_train)

    models: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    probas: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    # --- Logistic Regression ---
    lr = LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    y_hat = lr.predict(X_test)
    p_hat = lr.predict_proba(X_test)[:, 1]
    models["Logistic Regression"] = lr
    preds["Logistic Regression"] = y_hat
    probas["Logistic Regression"] = p_hat
    metrics["Logistic Regression"] = {
        **_binary_metrics(y_test, y_hat),
        "auc": roc_auc_score(y_test, p_hat),
    }
    _print_model_block("Logistic Regression", y_test, y_hat, WHY_LR, PROF_LR)

    # --- LDA: shared covariance, linear boundary ---
    lda = LinearDiscriminantAnalysis(priors=balanced_priors)
    try:
        lda.fit(X_train, y_train, sample_weight=sw)
    except TypeError:
        lda = LinearDiscriminantAnalysis(priors=balanced_priors)
        lda.fit(X_train, y_train)
    y_hat = lda.predict(X_test)
    p_hat = lda.predict_proba(X_test)[:, 1]
    models["LDA"] = lda
    preds["LDA"] = y_hat
    probas["LDA"] = p_hat
    metrics["LDA"] = {**_binary_metrics(y_test, y_hat), "auc": roc_auc_score(y_test, p_hat)}
    _print_model_block(
        "Linear Discriminant Analysis (LDA)",
        y_test,
        y_hat,
        WHY_LDA,
        PROF_LDA,
    )

    # --- QDA: class-specific covariance; quadratic boundary ---
    qda = QuadraticDiscriminantAnalysis(reg_param=0.02, priors=balanced_priors)
    try:
        qda.fit(X_train, y_train, sample_weight=sw)
    except TypeError:
        qda = QuadraticDiscriminantAnalysis(reg_param=0.02, priors=balanced_priors)
        qda.fit(X_train, y_train)
    y_hat = qda.predict(X_test)
    p_hat = qda.predict_proba(X_test)[:, 1]
    models["QDA"] = qda
    preds["QDA"] = y_hat
    probas["QDA"] = p_hat
    metrics["QDA"] = {**_binary_metrics(y_test, y_hat), "auc": roc_auc_score(y_test, p_hat)}
    _print_model_block(
        "Quadratic Discriminant Analysis (QDA)",
        y_test,
        y_hat,
        WHY_QDA,
        PROF_QDA,
    )

    # --- Decision Tree (CART) ---
    tree = DecisionTreeClassifier(
        max_depth=5,
        criterion="gini",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    p_hat = tree.predict_proba(X_test)[:, 1]
    models["Decision Tree (CART)"] = tree
    preds["Decision Tree (CART)"] = y_hat
    probas["Decision Tree (CART)"] = p_hat
    metrics["Decision Tree (CART)"] = {
        **_binary_metrics(y_test, y_hat),
        "auc": roc_auc_score(y_test, p_hat),
    }
    _print_model_block("Decision Tree (CART)", y_test, y_hat, WHY_TREE, PROF_TREE)

    return {
        "models": models,
        "preds": preds,
        "probas": probas,
        "metrics": metrics,
        "y_test": y_test,
        "feature_names": feature_names,
    }


# ── 4. MODEL COMPARISON ─────────────────────────────


def plot_confusion_matrices(
    preds: dict[str, np.ndarray],
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    """2×2 grid of confusion matrices on the test set."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            colorbar=False,
            values_format="d",
        )
        ax.set_title(name)
    plt.suptitle("Confusion matrices (test set)", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    probas: dict[str, np.ndarray],
    y_test: np.ndarray,
    metrics: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """All models on one ROC plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = metrics[name]["auc"]
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — test set")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Gini-based importance from the fitted CART model."""
    imp = tree.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        imp[order][::-1],
        color="#55A868",
    )
    ax.set_xlabel("Importance (Gini decrease)")
    ax.set_title("Decision Tree — feature importance (test uses scaled features)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_comparison_table(metrics: dict[str, dict[str, float]]) -> None:
    """Console table: Model | Accuracy | Precision | Recall | F1 | AUC."""
    print(f"\n{'=' * 72}")
    print("  Final comparison (test set)")
    print(f"{'=' * 72}")
    header = f"{'Model':<24} {'Acc':>9} {'Prec':>9} {'Rec':>9} {'F1':>9} {'AUC':>9}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<24} "
            f"{m['accuracy']:9.4f} "
            f"{m['precision']:9.4f} "
            f"{m['recall']:9.4f} "
            f"{m['f1']:9.4f} "
            f"{m['auc']:9.4f}"
        )
    print(f"{'=' * 72}\n")


def run_model_comparison(bundle: dict[str, Any]) -> None:
    preds = bundle["preds"]
    probas = bundle["probas"]
    metrics = bundle["metrics"]
    y_test = bundle["y_test"]
    feature_names = bundle["feature_names"]
    models = bundle["models"]

    plot_confusion_matrices(preds, y_test, PLOT_DIR / "06_confusion_matrices.png")
    plot_roc_curves(probas, y_test, metrics, PLOT_DIR / "07_roc_curves.png")
    plot_feature_importance(
        models["Decision Tree (CART)"],
        feature_names,
        PLOT_DIR / "08_feature_importance.png",
    )
    print_comparison_table(metrics)


# ── 5. INTERPRETATION (written report) ───────────────


def write_full_report(
    df_eda: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
) -> None:
    """
    Academic-style report (structure aligned with Assignment 3).
    Header: student | ID | UNINA.
    """
    n = len(df_eda)
    vc = df_eda["target_binary"].value_counts().sort_index()
    lines: list[str] = []

    lines.append("University of Naples Federico II (UNINA)")
    lines.append("Department of Economics and Statistics")
    lines.append("Course: Statistical Data Analysis")
    lines.append("Academic Year: 2025-2026")
    lines.append("Professors: Prof. Roberta Siciliano & Prof. Giulia Vannucci")
    lines.append("Mohammad Waliullah Shah | P37000087 | UNINA")
    lines.append("")
    lines.append("Assignment 4: Heart Disease Classification Analysis")
    lines.append("")

    lines.append("## Abstract")
    lines.append(
        "This report presents a full classification workflow for heart disease "
        f"presence using a dataset of {n} patients and 13 predictors. After exploratory "
        "analysis and median imputation for missing angiographic values, the outcome "
        "was binarized (no disease vs disease). Four classifiers were compared on a "
        "stratified 80/20 split with standardized features: Logistic Regression, "
        "Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), "
        "and a depth-limited Decision Tree (CART). Models used class rebalancing "
        "because the two classes are not perfectly equal. Performance was assessed with "
        "accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and "
        "AUC, plus tree-based feature importance. The results highlight how linear "
        "generative models and discriminative logistic regression behave on the same "
        "scaled feature space, and how a tree captures non-linear splits."
    )
    lines.append("")

    lines.append("## 1. Introduction")
    lines.append(
        "Early detection of heart disease supports better clinical decisions. The aim "
        "of this assignment is to learn a binary classifier from routinely collected "
        "clinical and stress-test variables, compare several statistical learning "
        "methods, and interpret their behavior with respect to scaling, class "
        "imbalance, and decision boundaries."
    )
    lines.append("")

    lines.append("## 2. Dataset and EDA")
    lines.append(
        f"The heart dataset contains {n} observations. The original target included "
        "severity levels 0–4; for this assignment, label 0 means no disease and levels "
        "1–4 are collapsed to a single “disease present” class. Missing values in "
        "`ca` and `thal` were replaced by column medians so that all patients enter "
        "the modeling stage."
    )
    lines.append(
        f"Class counts after binarization: no disease = {int(vc.get(0, 0))}, disease = "
        f"{int(vc.get(1, 0))}. Exploratory plots include class distribution, a "
        "correlation heatmap, a pair plot on a subset of continuous variables, and "
        "bar charts for coded categorical attributes. The heatmap helps detect "
        "multicollinearity; the pair plot gives a visual sense of overlap between "
        "classes in low-dimensional projections."
    )
    lines.append("")

    lines.append("## 3. Preprocessing")
    lines.append(
        "Predictors were those 13 feature columns (categorical factors are already "
        "numerically coded as in the source file). Data were split 80% train / 20% "
        "test with stratification on the binary outcome (`random_state=42`). "
        "`StandardScaler` was fit on the training set only and applied to the test "
        "set to avoid information leakage. This puts variables on comparable units, "
        "which matters strongly for logistic regression, LDA, and QDA; trees are "
        "scale-invariant, but a single scaled design matrix keeps the comparison fair."
    )
    lines.append(
        "Class imbalance was handled with `class_weight='balanced'` in Logistic "
        "Regression and the Decision Tree. For LDA and QDA, balanced `sample_weight` "
        "vectors were used when supported by the installed scikit-learn version; "
        "otherwise equal class priors (0.5, 0.5) were kept as a fallback so the "
        "generative fit still reflects concern for unequal class prevalence."
    )
    lines.append("")

    lines.append("## 4. Models and Results")
    lines.append(
        "**Logistic Regression** estimates a linear score and passes it through a "
        "logistic link; its decision boundary is linear in the scaled feature space. "
        "**LDA** assumes multivariate normal classes with a common covariance matrix, "
        "which yields a linear boundary when priors are equal; it is appropriate when "
        "features are roughly normal within each class and dispersion is similar. "
        "**QDA** relaxes common covariance, allowing quadratic boundaries—useful when "
        "variability differs materially between disease groups, at the cost of more "
        "parameters. **CART** with `max_depth=5` and Gini impurity models non-linear "
        "interactions via recursive partitioning."
    )
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |"
        )
    lines.append("")

    lines.append("## 5. Discussion")
    lines.append(
        "**Confusion matrices** summarize where each model trades off false alarms "
        "and missed cases. **ROC and AUC** integrate performance across thresholds; "
        "they are especially informative alongside imbalanced outcomes. **Feature "
        "importance** from the tree highlights which variables dominated splits under "
        "the chosen depth constraint; importances are relative and sum to one."
    )
    lines.append(
        "**Effect of scaling:** distance-based and likelihood-based linear models are "
        "not invariant to measurement units; scaling prevents a variable with large "
        "variance from dominating solely because of its scale. **Decision boundaries:** "
        "LR and LDA produce linear separators (in different ways: discriminative vs "
        "generative); QDA can curve the boundary; the tree axis-aligned partitions "
        "approximate non-linear regions without a single smooth surface."
    )
    lines.append("")

    lines.append("## 6. Conclusion")
    lines.append(
        "The analysis compared four transparent classifiers on standardized inputs with "
        "class rebalancing. The numeric table and ROC figure give a concise ranking by "
        "multiple metrics; no single number should be read in isolation on medical "
        "data, but together they describe discrimination ability and calibration of "
        "errors for this test split."
    )
    lines.append("")

    lines.append("## References")
    lines.append(
        "- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). "
        "*An Introduction to Statistical Learning*. Springer."
    )
    lines.append(
        "- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "
        "*The Elements of Statistical Learning*. Springer."
    )
    lines.append(
        "- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
        "*Journal of Machine Learning Research*."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_assignment_footer(saved_rel_paths: list[str]) -> None:
    """Standard closing banner for the assignment."""
    print("=" * 48)
    print("  Assignment 4 - Classification Analysis")
    print("  Student: Mohammad Waliullah Shah")
    print("  ID: P37000087")
    print("  University of Naples Federico II")
    print("  Course: Statistical Data Analysis")
    print("=" * 48)
    print("  Saved Files:")
    for p in saved_rel_paths:
        print(f"  - {p}")
    print("=" * 48)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    df_raw = load_raw_data(DATA_PATH)
    df_eda = run_eda(df_raw)

    X_train, X_test, y_train, y_test, _scaler, feat_names = run_preprocessing(df_eda)

    bundle = run_models(X_train, X_test, y_train, y_test, feat_names)
    run_model_comparison(bundle)

    write_full_report(df_eda, bundle["metrics"])

    saved = [
        "assignment4_heart_disease.py",
        "eda_plots/01_class_distribution.png",
        "eda_plots/02_pie_chart_target.png",
        "eda_plots/03_correlation_heatmap.png",
        "eda_plots/04_pair_plot.png",
        "eda_plots/05_bar_charts_categorical.png",
        "eda_plots/06_confusion_matrices.png",
        "eda_plots/07_roc_curves.png",
        "eda_plots/08_feature_importance.png",
        "assignment4_report.txt",
    ]
    print_assignment_footer(saved)


if __name__ == "__main__":
    main()
