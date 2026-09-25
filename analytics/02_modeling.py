import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


def build_preprocessor():
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
        [
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    return preprocessor


def evaluate_classifier(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    auc = roc_auc_score(
        y_test,
        probabilities,
    )
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    return {
        "name": name,
        "pipeline": pipeline,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": matrix,
        "probabilities": probabilities,
    }


def run_modeling():
    df = pd.read_csv("analytics/titanic.csv")

    if "deck" in df.columns:
        df = df.drop(columns=["deck"])

    X = df.drop(columns=["survived"])
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("--- Stratified Train/Test Split ---")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Training survival rate: {y_train.mean():.4f}")
    print(f"Testing survival rate: {y_test.mean():.4f}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    print("\n--- Model Performance Metrics ---")

    baseline_results = []

    for name, model in models.items():
        pipeline = Pipeline(
            [
                ("prep", build_preprocessor()),
                ("clf", model),
            ]
        )

        result = evaluate_classifier(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        baseline_results.append(result)

    baseline_table = pd.DataFrame(
        [
            {
                "Model": result["name"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "ROC_AUC": result["roc_auc"],
            }
            for result in baseline_results
        ]
    )

    print("\n--- Baseline Model Comparison ---")
    print(baseline_table.round(4).to_string(index=False))

    baseline_table.to_csv(
        "analytics/classifier_comparison.csv",
        index=False,
    )

    plt.figure(figsize=(10, 7))

    for result in baseline_results:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            result["probabilities"],
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f'{result["name"]} (AUC={result["roc_auc"]:.3f})',
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Classification Models")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "analytics/roc_curves.png",
        dpi=200,
    )

    plt.close()

    print(
        "\nROC curve saved to analytics/roc_curves.png"
    )

    decision_tree_pipeline = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=4,
                    random_state=42,
                ),
            ),
        ]
    )

    decision_tree_pipeline.fit(
        X_train,
        y_train,
    )

    feature_names = (
        decision_tree_pipeline
        .named_steps["prep"]
        .get_feature_names_out()
    )

    plt.figure(
        figsize=(18, 10)
    )

    plot_tree(
        decision_tree_pipeline.named_steps["clf"],
        feature_names=feature_names,
        class_names=[
            "Did not survive",
            "Survived",
        ],
        filled=True,
        rounded=True,
        fontsize=7,
    )

    plt.title("Decision Tree Classifier")
    plt.tight_layout()

    plt.savefig(
        "analytics/decision_tree.png",
        dpi=200,
    )

    plt.close()

    print(
        "\nDecision tree saved to analytics/decision_tree.png"
    )

    print("\n--- Class Weight Balanced Comparison ---")

    balanced_models = {
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree Balanced": DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest Balanced": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
    }

    balanced_results = []

    for name, model in balanced_models.items():
        pipeline = Pipeline(
            [
                ("prep", build_preprocessor()),
                ("clf", model),
            ]
        )

        result = evaluate_classifier(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        balanced_results.append(result)

    print("\n--- SMOTE Comparison ---")

    smote_models = {
        "Logistic Regression SMOTE": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree SMOTE": DecisionTreeClassifier(
            max_depth=4,
            random_state=42,
        ),
        "Random Forest SMOTE": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    smote_results = []

    for name, model in smote_models.items():
        pipeline = ImbPipeline(
            [
                ("prep", build_preprocessor()),
                ("smote", SMOTE(random_state=42)),
                ("clf", model),
            ]
        )

        result = evaluate_classifier(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        smote_results.append(result)

    imbalance_results = balanced_results + smote_results

    imbalance_table = pd.DataFrame(
        [
            {
                "Method": (
                    "Class Weight"
                    if "Balanced" in result["name"]
                    else "SMOTE"
                ),
                "Model": result["name"]
                .replace(" Balanced", "")
                .replace(" SMOTE", ""),
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "ROC_AUC": result["roc_auc"],
            }
            for result in imbalance_results
        ]
    )

    print("\n--- Class Weight vs SMOTE Comparison ---")
    print(
        imbalance_table.round(4).to_string(
            index=False
        )
    )

    imbalance_table.to_csv(
        "analytics/imbalance_comparison.csv",
        index=False,
    )

    print(
        "\nSMOTE was applied only to the training data inside the pipeline."
    )

    print("\n--- Random Forest Grid Search ---")

    rf_pipeline = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    random_state=42,
                    oob_score=True,
                    bootstrap=True,
                ),
            ),
        ]
    )

    parameter_grid = {
        "clf__n_estimators": [
            50,
            100,
            200,
        ],
        "clf__max_depth": [
            3,
            5,
            10,
            None,
        ],
        "clf__max_features": [
            "sqrt",
            "log2",
        ],
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_rf = grid_search.best_estimator_

    print(
        "Best Parameters:",
        grid_search.best_params_,
    )

    print(
        f"Best Cross-Validation F1: "
        f"{grid_search.best_score_:.4f}"
    )

    tuned_predictions = best_rf.predict(
        X_test
    )

    tuned_probabilities = best_rf.predict_proba(
        X_test
    )[:, 1]

    tuned_accuracy = accuracy_score(
        y_test,
        tuned_predictions,
    )

    tuned_precision = precision_score(
        y_test,
        tuned_predictions,
        zero_division=0,
    )

    tuned_recall = recall_score(
        y_test,
        tuned_predictions,
        zero_division=0,
    )

    tuned_f1 = f1_score(
        y_test,
        tuned_predictions,
        zero_division=0,
    )

    tuned_auc = roc_auc_score(
        y_test,
        tuned_probabilities,
    )

    tuned_matrix = confusion_matrix(
        y_test,
        tuned_predictions,
    )

    oob_score = (
        best_rf
        .named_steps["clf"]
        .oob_score_
    )

    print("\n--- Tuned Random Forest Test Performance ---")
    print(f"Accuracy: {tuned_accuracy:.4f}")
    print(f"Precision: {tuned_precision:.4f}")
    print(f"Recall: {tuned_recall:.4f}")
    print(f"F1 Score: {tuned_f1:.4f}")
    print(f"ROC AUC: {tuned_auc:.4f}")
    print("Confusion Matrix:")
    print(tuned_matrix)
    print(f"OOB Score: {oob_score:.4f}")

    print("\n--- Regression: Predicting Fare ---")

    regression_data = df[
        [
            "fare",
            "age",
            "sibsp",
            "parch",
            "pclass",
        ]
    ].dropna()

    regression_X = regression_data[
        [
            "age",
            "sibsp",
            "parch",
            "pclass",
        ]
    ]

    regression_y = regression_data["fare"]

    reg_X_train, reg_X_test, reg_y_train, reg_y_test = (
        train_test_split(
            regression_X,
            regression_y,
            test_size=0.2,
            random_state=42,
        )
    )

    regression_model = LinearRegression()

    regression_model.fit(
        reg_X_train,
        reg_y_train,
    )

    regression_predictions = regression_model.predict(
        reg_X_test
    )

    mae = mean_absolute_error(
        reg_y_test,
        regression_predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            reg_y_test,
            regression_predictions,
        )
    )

    r2 = r2_score(
        reg_y_test,
        regression_predictions,
    )

    n = len(reg_y_test)
    p = reg_X_test.shape[1]

    adjusted_r2 = (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )

    residuals = (
        reg_y_test
        - regression_predictions
    )

    print("\n--- Linear Regression Metrics ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Adjusted R2: {adjusted_r2:.4f}")

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        regression_predictions,
        residuals,
        alpha=0.6,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title(
        "Linear Regression Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        "analytics/regression_residuals.png",
        dpi=200,
    )

    plt.close()

    print(
        "\nResidual plot saved to analytics/regression_residuals.png"
    )

    absolute_residual_correlation = np.corrcoef(
        regression_predictions,
        np.abs(residuals),
    )[0, 1]

    if abs(absolute_residual_correlation) >= 0.30:
        heteroscedasticity_statement = (
            "The residual spread changes with predicted fare, "
            "which provides evidence consistent with heteroscedasticity."
        )
    else:
        heteroscedasticity_statement = (
            "The residual spread does not show strong evidence "
            "of heteroscedasticity based on the relationship "
            "between predicted fare and absolute residuals."
        )

    print("\n--- Heteroscedasticity Assessment ---")
    print(heteroscedasticity_statement)

    print(
        f"Correlation between predicted fare and "
        f"absolute residuals: "
        f"{absolute_residual_correlation:.4f}"
    )

    final_classification_table = baseline_table.copy()

    tuned_row = pd.DataFrame(
        [
            {
                "Model": "Tuned Random Forest",
                "Accuracy": tuned_accuracy,
                "Precision": tuned_precision,
                "Recall": tuned_recall,
                "F1": tuned_f1,
                "ROC_AUC": tuned_auc,
            }
        ]
    )

    final_classification_table = pd.concat(
        [
            final_classification_table,
            tuned_row,
        ],
        ignore_index=True,
    )

    final_regression_table = pd.DataFrame(
        [
            {
                "Model": "Linear Regression",
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted_R2": adjusted_r2,
            }
        ]
    )

    final_classification_table.to_csv(
        "analytics/final_classification_comparison.csv",
        index=False,
    )

    final_regression_table.to_csv(
        "analytics/final_regression_comparison.csv",
        index=False,
    )

    print("\n--- Final Classification Comparison ---")
    print(
        final_classification_table.round(4).to_string(
            index=False
        )
    )

    print("\n--- Final Regression Comparison ---")
    print(
        final_regression_table.round(4).to_string(
            index=False
        )
    )

    best_baseline = baseline_table.loc[
        baseline_table["F1"].idxmax()
    ]

    print("\n--- Final Model Summary ---")
    print(
        f"The baseline classifier with the highest F1 score was "
        f"{best_baseline['Model']} with an F1 score of "
        f"{best_baseline['F1']:.4f}."
    )

    print(
        f"The tuned Random Forest achieved an F1 score of "
        f"{tuned_f1:.4f} and ROC AUC of {tuned_auc:.4f}."
    )

    print(
        f"The tuned Random Forest OOB score was "
        f"{oob_score:.4f}, providing an additional internal "
        f"validation measure."
    )

    print(
        f"The Linear Regression model achieved an R2 score of "
        f"{r2:.4f} with an RMSE of {rmse:.2f} for fare prediction."
    )

    print(
        "The final results provide both classification performance "
        "for survival prediction and regression performance for fare prediction."
    )

    joblib.dump(
        best_rf,
        "analytics/pipeline.joblib",
    )

    print(
        "\nFitted preprocessing + Random Forest "
        "pipeline saved to analytics/pipeline.joblib"
    )

    loaded_pipeline = joblib.load(
        "analytics/pipeline.joblib"
    )

    raw_input = pd.DataFrame(
        [
            {
                "pclass": 3,
                "sex": "male",
                "age": 30,
                "sibsp": 0,
                "parch": 0,
                "fare": 15.0,
                "embarked": "S",
                "name": "Demo Passenger",
                "ticket": "DEMO",
                "cabin": np.nan,
                "boat": np.nan,
                "body": np.nan,
                "home.dest": np.nan,
                "class": "Third",
                "who": "man",
                "adult_male": True,
                "alive": "no",
                "alone": True,
            }
        ]
    )

    raw_prediction = loaded_pipeline.predict(
        raw_input
    )[0]

    raw_probability = loaded_pipeline.predict_proba(
        raw_input
    )[0, 1]

    print("\n--- Saved Pipeline Reload Test ---")

    print(
        f"Raw input prediction: "
        f"{int(raw_prediction)}"
    )

    print(
        f"Survival probability: "
        f"{raw_probability:.4f}"
    )

    print(
        "Pipeline reload and raw-input "
        "prediction successful."
    )

    print("\nModeling completed successfully.")


if __name__ == "__main__":
    run_modeling()