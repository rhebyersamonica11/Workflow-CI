FROM python:3.9-slim

WORKDIR /app

# Copy dari root
COPY requirements.txt .

# Copy dari folder MLProject
COPY MLProject/modelling.py .
COPY MLProject/namadataset_preprocessing/loan_clean.csv ./namadataset_preprocessing/loan_clean.csv

CMD ["python", "modelling.py"]