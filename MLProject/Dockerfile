# Menggunakan image Python yang ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy file requirements dan install dependencies
COPY conda.yaml .
RUN pip install mlflow scikit-learn pandas

# Copy file modelling ke dalam container
COPY modelling.py .
COPY namadataset_preprocessing/loan_clean.csv ./namadataset_preprocessing/loan_clean.csv

# Jalankan skrip saat container mulai
CMD ["python", "modelling.py"]