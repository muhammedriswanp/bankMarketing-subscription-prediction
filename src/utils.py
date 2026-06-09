import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns=["y"])
    y_train = (train["y"] == "yes").astype(int)
    X_test = test.drop(columns=["y"])
    y_test = (test["y"] == "yes").astype(int)
    return X_train, X_test, y_train, y_test

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)

def load_pipeline(path):
    return joblib.load(path)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    if hasattr(model, "best_score_"):
        print(f"Best CV F1: {model.best_score_:.4f}")
        print(f"Best params: {model.best_params_}")
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"Cross-validation F1 (5-fold): {scores.mean():.4f}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

