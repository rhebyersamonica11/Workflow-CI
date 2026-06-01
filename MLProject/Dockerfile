# Menggunakan image Python yang ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy file requirements dan install dependencies
# Kita ambil dari folder Membangun_model sesuai struktur Anda
COPY MLProject/requirements.txt .
RUN pip install -r requirements.txt

# Copy file modelling dan dataset
COPY MLProject/modelling.py .
COPY MLProject/namadataset_preprocessing/loan_clean.csv ./namadataset_preprocessing/loan_clean.csv

# Jalankan skrip saat container mulai
CMD ["python", "modelling.py"]