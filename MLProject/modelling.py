import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Autologging mencatat parameter, metrik, dan model secara otomatis
mlflow.sklearn.autolog()

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    # Membagi data training dan testing
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_baseline_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # PERBAIKAN: Hapus set_tracking_uri. 
    # MLflow akan otomatis menggunakan variabel lingkungan (Environment Variables) 
    # yang kita set di GitHub Actions nanti.
    mlflow.set_experiment("Eksperimen_Loan_Scoring_Model")
    
    # Path dataset yang fleksibel
    data_path = os.path.join(os.path.dirname(__file__), 'namadataset_preprocessing', 'loan_clean.csv')
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        model = train_baseline_model(X_train, y_train)
        
        # Evaluasi manual tetap bisa dilakukan, 
        # namun autolog() sudah menangani pencatatan metrik lainnya
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # PENTING: Mendaftarkan model agar bisa di-build ke Docker nantinya
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            registered_model_name="LoanScoringModel"
        )
        
        print(f"Baseline run selesai. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()