from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    _has_xgboost = True
except ImportError:
    _has_xgboost = False


def baseline_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])


def decision_tree_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])


def random_forest_model(preprocessor):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    param_grid = {
        "classifier__max_depth": [10, None],
        "classifier__min_samples_split": [2, 10],
        "classifier__n_estimators": [100, 200]
    }
    return GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)


def gradient_boosting_model(preprocessor):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]
    }
    return GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)


def adaboost_model(preprocessor):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(random_state=42))
    ])
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.5, 1.0]
    }
    return GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)


def xgboost_model(preprocessor):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0))
    ])
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]
    }
    return GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
