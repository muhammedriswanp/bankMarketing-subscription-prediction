from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def baseline_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

logistic_regression_model = baseline_model

def sgd_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SGDClassifier(max_iter=1000, random_state=42, class_weight="balanced", loss="log_loss"))
    ])

def decision_tree_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

def random_forest_model(preprocessor):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5, 10]
    }
    return GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)

def gradient_boosting_model(preprocessor):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
    param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__learning_rate": [0.05, 0.1, 0.2]
    }
    return GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
