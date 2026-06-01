# Menggunakan image Python yang ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy file requirements dan install dependencies
# Kita ambil dari folder Membangun_model sesuai struktur Anda
COPY Membangun_model/requirements.txt .
RUN pip install -r requirements.txt

# Copy file modelling dan dataset
COPY Membangun_model/modelling.py .
COPY Membangun_model/namadataset_preprocessing/loan_clean.csv ./namadataset_preprocessing/loan_clean.csv

# Jalankan skrip saat container mulai
CMD ["python", "modelling.py"]