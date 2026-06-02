import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Autologging mencatat parameter, metrik, dan model secara otomatis
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

def main():
    # Gunakan penyimpanan file lokal agar tidak perlu server
    # 'file:./mlruns' adalah format URI lokal yang paling stabil
    mlflow.set_tracking_uri("file:./mlruns")
    
    mlflow.set_experiment("Eksperimen_Loan_Scoring_Model")
    
    # Path dataset yang fleksibel
    data_path = os.path.join(os.path.dirname(__file__), 'namadataset_preprocessing', 'loan_clean.csv')
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        model = train_baseline_model(X_train, y_train)
        
        # Evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # PERBAIKAN: Hapus 'registered_model_name' 
        # Karena di GitHub Actions (file based) kita tidak bisa mendaftarkan model ke registry tanpa database.
        # Ini akan menghilangkan error 404/database.
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model"
        )
        
        print(f"Baseline run selesai. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()