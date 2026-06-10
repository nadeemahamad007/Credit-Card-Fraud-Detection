import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DROP_COLUMNS = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "job",
    "dob",
    "trans_num",
]

CATEGORICAL_FEATURES = ["merchant", "category", "gender", "state"]
NUMERICAL_FEATURES = [
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train = train_df.drop(columns=DROP_COLUMNS + ["is_fraud"])
    y_train = train_df["is_fraud"]
    x_test = test_df.drop(columns=DROP_COLUMNS + ["is_fraud"])
    y_test = test_df["is_fraud"]
    return x_train, x_test, y_train, y_test


def evaluate_model(model_name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series):
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }

    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, predictions)
    return metrics, report, matrix


def save_confusion_matrix(matrix, output_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_importance(best_pipeline: Pipeline, output_dir: Path) -> None:
    model = best_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = best_pipeline.named_steps["preprocessor"]
    encoded_features = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    feature_names = NUMERICAL_FEATURES + list(encoded_features)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    importance_df.head(15).to_csv(output_dir / "top_features.csv", index=False)

    top_features = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#2a6f97")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "top_features.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare fraud detection models.")
    parser.add_argument("--train-path", default="data/fraudTrain.csv")
    parser.add_argument("--test-path", default="data/fraudTest.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = load_data(Path(args.train_path), Path(args.test_path))
    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        ),
    }

    metrics_rows = []
    reports = {}
    best_pipeline = None
    best_model_name = None
    best_f1 = -1.0

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(x_train, y_train)
        metrics, report, matrix = evaluate_model(model_name, pipeline, x_test, y_test)
        metrics_rows.append(metrics)
        reports[model_name] = report
        save_confusion_matrix(
            matrix,
            output_dir / f"{model_name}_confusion_matrix.png",
            f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        )

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_pipeline = pipeline
            best_model_name = model_name

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_score", ascending=False)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    summary = {
        "best_model": best_model_name,
        "metrics": metrics_df.to_dict(orient="records"),
        "classification_reports": reports,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_pipeline is not None:
        joblib.dump(best_pipeline, output_dir / "best_fraud_model.joblib")
        save_feature_importance(best_pipeline, output_dir)

    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")


if __name__ == "__main__":
    main()
