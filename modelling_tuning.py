import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Setup Path
base_dir = os.path.dirname(os.path.abspath(__file__))
ml_project_path = os.path.join(base_dir, 'MLProject')
if ml_project_path not in sys.path:
    sys.path.append(ml_project_path)

from modelling import prepare_data

# 2. Setup Otentikasi DagsHub untuk CI/CD
# Kita set variabel lingkungan yang dibaca oleh SDK DagsHub/MLflow
token = os.environ.get("DAGSHUB_TOKEN")
if token:
    # Set kredensial agar MLflow dan DagsHub mengenalinya tanpa interaksi
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rhebyersamonica11"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    # Menentukan URL tracking secara manual menghindari deteksi otomatis yang memicu OAuth
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/rhebyersamonica11/msml-loan-scoring.mlflow"
else:
    # Import dagshub hanya jika dijalankan secara lokal
    import dagshub
    dagshub.init(repo_owner='rhebyersamonica11', repo_name='msml-loan-scoring', mlflow=True)

import mlflow
import mlflow.sklearn

def run_experiment():
    data_path = os.path.join(base_dir, "namadataset_preprocessing", "loan_clean.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset tidak ditemukan di {data_path}")
        return

    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    mlflow.set_experiment("Loan_Approval_Classification_Advance")
    
    configs = [
        {"name": "RF_Tuning_1", "n": 50, "depth": 5}, 
        {"name": "RF_Tuning_2", "n": 150, "depth": 12}
    ]
    
    for cfg in configs:
        with mlflow.start_run(run_name=cfg["name"]):
            rf = RandomForestClassifier(n_estimators=cfg["n"], max_depth=cfg["depth"], random_state=42)
            rf.fit(X_train, y_train)
            
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            mlflow.log_metric("accuracy", acc)
            
            # Log artefak
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, preds))
            mlflow.log_artifact(report_path)
            
            plt.figure(figsize=(10, 6))
            pd.Series(rf.feature_importances_, index=X_train.columns).sort_values().plot(kind='barh')
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
            
            mlflow.sklearn.log_model(sk_model=rf, artifact_path="model")
            print(f"[+] Eksperimen {cfg['name']} Selesai.")

if __name__ == "__main__":
    run_experiment()