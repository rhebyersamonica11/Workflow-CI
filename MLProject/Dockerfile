# Menggunakan image Python yang ringan
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements dari direktori MLProject (lokasi build saat ini)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy file modelling dan dataset dari direktori MLProject
COPY modelling.py .
COPY namadataset_preprocessing/loan_clean.csv ./namadataset_preprocessing/loan_clean.csv

CMD ["python", "modelling.py"]