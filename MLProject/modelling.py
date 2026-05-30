import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Aktifkan autologging untuk mencatat semua detail model secara otomatis
mlflow.sklearn.autolog()

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_baseline_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    return acc, prec, rec

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Eksperimen_Loan_Scoring_Model")
    
    data_path = os.path.join(os.path.dirname(__file__), 'namadataset_preprocessing', 'loan_clean.csv')
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        model = train_baseline_model(X_train, y_train)
        acc, _, _ = evaluate_model(model, X_test, y_test)
        print(f"Baseline run selesai dengan autologging. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()