import dagshub
import mlflow
import mlflow.sklearn
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Tambahkan path agar bisa import fungsi dari file modelling.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modelling import prepare_data

# 1. Inisialisasi DagsHub untuk Tracking Online
dagshub.init(repo_owner='rhebyersamonica11', repo_name='msml-loan-scoring', mlflow=True)

def run_experiment():
    # Persiapan Data
    data_path = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing", "loan_clean.csv")
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    mlflow.set_experiment("Loan_Approval_Classification_Advance")
    
    # Konfigurasi Hyperparameter Tuning
    configs = [
        {"name": "RF_Tuning_1", "n": 50, "depth": 5}, 
        {"name": "RF_Tuning_2", "n": 150, "depth": 12}
    ]
    
    for cfg in configs:
        # 2. Mulai Eksperimen (Manual Logging)
        with mlflow.start_run(run_name=cfg["name"]):
            rf = RandomForestClassifier(n_estimators=cfg["n"], max_depth=cfg["depth"], random_state=42)
            rf.fit(X_train, y_train)
            
            # 3. Manual Logging Metrik
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            mlflow.log_metric("accuracy", acc)
            
            # 4. Artefak Tambahan 1: Laporan Klasifikasi (TXT)
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, preds))
            mlflow.log_artifact(report_path)
            
            # 5. Artefak Tambahan 2: Plot Feature Importance (PNG)
            plt.figure(figsize=(10, 6))
            pd.Series(rf.feature_importances_, index=X_train.columns).sort_values().plot(kind='barh')
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            
            # 6. Log Model
            mlflow.sklearn.log_model(
                sk_model=rf, 
                artifact_path="model"
            )
            
            print(f"[+] {cfg['name']} Selesai: Akurasi {acc:.4f}, Artefak berhasil di-log.")

if __name__ == "__main__":
    run_experiment()