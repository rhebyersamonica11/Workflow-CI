import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def prepare_data(data_path):
    """
    Memuat dataset dan membagi menjadi train-test set.
    """

    df = pd.read_csv(data_path)

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_baseline_model(X_train, y_train):
    """
    Melatih model Logistic Regression baseline.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def main():

    # Membuat eksperimen MLflow
    mlflow.set_experiment("Eksperimen_Loan_Scoring_Model")

    # Lokasi dataset hasil preprocessing
    data_path = os.path.join(
        os.path.dirname(__file__),
        "namadataset_preprocessing",
        "loan_clean.csv"
    )

    # Split data
    X_train, X_test, y_train, y_test = prepare_data(data_path)

    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):

        # ==========================
        # Log parameter
        # ==========================
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # ==========================
        # Training
        # ==========================
        model = train_baseline_model(X_train, y_train)

        # ==========================
        # Prediksi
        # ==========================
        y_pred = model.predict(X_test)

        # ==========================
        # Evaluasi
        # ==========================
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ==========================
        # Log Metrics
        # ==========================
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # ==========================
        # Simpan model
        # ==========================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("=" * 50)
        print("TRAINING BERHASIL")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()