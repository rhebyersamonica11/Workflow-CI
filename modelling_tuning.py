import dagshub
import mlflow
import mlflow.sklearn
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- PERBAIKAN: Menambahkan folder MLProject ke sys.path ---
# Mengambil path absolut dari direktori file ini
base_dir = os.path.dirname(os.path.abspath(__file__))
ml_project_path = os.path.join(base_dir, 'MLProject')

if ml_project_path not in sys.path:
    sys.path.append(ml_project_path)

# Sekarang import dari folder MLProject akan berhasil
from modelling import prepare_data

# Tambahkan ini untuk membaca token dari env
token = os.environ.get("DAGSHUB_TOKEN")

if token:
    # Jika token ada (di GitHub Actions), gunakan token tersebut
    dagshub.init(repo_owner='rhebyersamonica11', repo_name='msml-loan-scoring', mlflow=True, token=token)
else:
    # Jika token tidak ada (lokal), gunakan cara biasa
    dagshub.init(repo_owner='rhebyersamonica11', repo_name='msml-loan-scoring', mlflow=True)

def run_experiment():
    # Persiapan Data dengan path yang fleksibel
    data_path = os.path.join(base_dir, "namadataset_preprocessing", "loan_clean.csv")
    
    # Validasi file ada sebelum training
    if not os.path.exists(data_path):
        print(f"Error: File dataset tidak ditemukan di {data_path}")
        return

    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    mlflow.set_experiment("Loan_Approval_Classification_Advance")
    
    # Konfigurasi Hyperparameter Tuning
    configs = [
        {"name": "RF_Tuning_1", "n": 50, "depth": 5}, 
        {"name": "RF_Tuning_2", "n": 150, "depth": 12}
    ]
    
    for cfg in configs:
        # 2. Mulai Eksperimen
        with mlflow.start_run(run_name=cfg["name"]):
            rf = RandomForestClassifier(n_estimators=cfg["n"], max_depth=cfg["depth"], random_state=42)
            rf.fit(X_train, y_train)
            
            # 3. Manual Logging Metrik
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            mlflow.log_metric("accuracy", acc)
            
            # 4. Artefak: Laporan Klasifikasi (TXT)
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, preds))
            mlflow.log_artifact(report_path)
            
            # 5. Artefak: Plot Feature Importance (PNG)
            plt.figure(figsize=(10, 6))
            # Menggunakan X_train.columns langsung
            pd.Series(rf.feature_importances_, index=X_train.columns).sort_values().plot(kind='barh')
            plt.title(f"Feature Importance - {cfg['name']}")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close() # Penting: Tutup plot untuk menghemat memori
            
            # 6. Log Model
            mlflow.sklearn.log_model(
                sk_model=rf, 
                artifact_path="model"
            )
            
            print(f"[+] {cfg['name']} Selesai: Akurasi {acc:.4f}")

if __name__ == "__main__":
    run_experiment()