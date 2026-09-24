import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


def run_modeling():
    df = pd.read_csv("analytics/titanic.csv")
    df = df.drop(columns=["deck"])

    X = df.drop(columns=["survived"])
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["age", "fare", "sibsp", "parch"]
    cat_cols = ["sex", "embarked", "pclass"]

    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(max_depth=4),
        "Random Forest": RandomForestClassifier(n_estimators=100),
    }

    print("--- Model Performance Metrics ---")
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = (
            pipe.predict_proba(X_test)[:, 1]
            if hasattr(pipe, "predict_proba")
            else preds
        )

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"{name} -> Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    dt_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=3)),
        ]
    )
    dt_pipe.fit(X_train, y_train)
    plt.figure(figsize=(12, 8))
    plot_tree(dt_pipe.named_steps["clf"], filled=True)
    plt.savefig("analytics/decision_tree.png")
    plt.close()

    rf_param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [3, 5, None],
    }
    rf_tune_pipe = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(oob_score=True, random_state=42),
            ),
        ]
    )
    grid_search = GridSearchCV(
        rf_tune_pipe, rf_param_grid, cv=3, scoring="f1"
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print("\n--- Hyperparameter Tuning ---")
    print("Best Parameters:", grid_search.best_params_)

    reg_X = df[["age", "sibsp", "parch", "pclass"]].dropna()
    reg_y = df.loc[reg_X.index, "fare"]

    reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
        reg_X, reg_y, test_size=0.2, random_state=42
    )
    reg_model = LinearRegression()
    reg_model.fit(reg_X_train, reg_y_train)
    reg_preds = reg_model.predict(reg_X_test)

    mae = mean_absolute_error(reg_y_test, reg_preds)
    rmse = np.sqrt(mean_squared_error(reg_y_test, reg_preds))
    r2 = r2_score(reg_y_test, reg_preds)

    print("\n--- Linear Regression Task ---")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2 Score: {r2:.4f}")

    joblib.dump(best_rf, "analytics/pipeline.joblib")
    print("\nFitted model pipeline saved to analytics/pipeline.joblib")


if __name__ == "__main__":
    run_modeling()