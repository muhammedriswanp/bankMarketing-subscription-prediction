import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.preprocessing import build_preprocessor
from src.model import (
    baseline_model,
    decision_tree_model,
    random_forest_model,
    gradient_boosting_model,
    adaboost_model,
    xgboost_model,
)
from src.utils import load_data

DROP_FEATURES = ["duration"]

warnings.filterwarnings("ignore")


def get_feature_names(fitted_preprocessor, numeric_features, categorical_features):
    num_names = numeric_features
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features)
    return np.concatenate([num_names, cat_names])


def main():
    DATA_DIR = Path(__file__).parent / "data"
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"

    print("=" * 70)
    print("BANK MARKETING SUBSCRIPTION PREDICTION: Comparison of All Models")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH, TEST_PATH)
    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── 2. Build preprocessor ─────────────────────────────────────────────────
    print("\n[2] Building preprocessor...")
    all_numeric = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    all_categorical = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [c for c in all_numeric if c not in DROP_FEATURES]
    categorical_features = [c for c in all_categorical if c not in DROP_FEATURES]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    print(f"    Numeric: {len(numeric_features)}  |  Categorical: {len(categorical_features)}")

    # ── 3. Define models ──────────────────────────────────────────────────────
    models = {
        "LogisticRegression (baseline)": baseline_model(preprocessor),
        "DecisionTree": decision_tree_model(preprocessor),
        "RandomForest + GridSearch": random_forest_model(preprocessor),
        "GradientBoosting + GridSearch": gradient_boosting_model(preprocessor),
        "AdaBoost + GridSearch": adaboost_model(preprocessor),
        "XGBoost + GridSearch": xgboost_model(preprocessor),
    }

    # ── 4. Train & evaluate ───────────────────────────────────────────────────
    results = []
    best_model = None
    best_f1 = 0
    best_model_name = ""
    best_tree_model = None
    best_tree_f1 = 0
    print("\n[3] Training & evaluating models...\n" + "-" * 70)

    for name, model in models.items():
        print(f"\n>>> {name}")
        try:
            model.fit(X_train, y_train)

            if hasattr(model, "best_score_"):
                cv_f1 = model.best_score_
            else:
                cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            results.append(
                {
                    "Model": name,
                    "CV F1": round(cv_f1, 4),
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "ROC-AUC": round(roc_auc, 4),
                }
            )

            if cv_f1 > best_f1:
                best_f1 = cv_f1
                best_model = model
                best_model_name = name

            if hasattr(model, "best_estimator_"):
                inner_clf = model.best_estimator_.named_steps["classifier"]
                if hasattr(inner_clf, "feature_importances_") and cv_f1 > best_tree_f1:
                    best_tree_f1 = cv_f1
                    best_tree_model = model

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print(f"    CV F1: {cv_f1:.4f}  |  Acc: {acc:.4f}  |  Prec: {prec:.4f}  |  Rec: {rec:.4f}  |  F1: {f1:.4f}  |  ROC-AUC: {roc_auc:.4f}")
            print(f"    Confusion Matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

            if hasattr(model, "best_params_"):
                print(f"    Best params: {model.best_params_}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"Model": name, "CV F1": None, "Accuracy": None, "Precision": None, "Recall": None, "F1": None, "ROC-AUC": None})

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # ── 6. Save results ───────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nResults saved to output/model_comparison.csv")

    # ── 7. Feature Importance (Best Tree-Based Model) ──────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (BEST TREE-BASED MODEL)")
    print("=" * 70)

    preprocessor.fit(X_train, y_train)
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    if best_tree_model is not None:
        clf = best_tree_model.best_estimator_.named_steps["classifier"]
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Top 5 features:")
        for i in range(min(5, len(indices))):
            print(f"  {i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    else:
        print("No tree-based model available for feature importance.")

    # ── 8. Save final model ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING BEST MODEL (no retraining)")
    print("=" * 70)
    from joblib import dump
    if hasattr(best_model, "best_estimator_"):
        dump(best_model.best_estimator_, "model.pkl")
    else:
        dump(best_model, "model.pkl")
    print(f"Saved model.pkl (best model: {best_model_name}, F1 = {best_f1:.4f})")

if __name__ == "__main__":
    main()
