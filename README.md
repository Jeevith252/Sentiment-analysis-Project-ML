
# Sentiment Analysis using DistilBERT | FastAPI | Docker | AWS | MLflow

An end-to-end MLOps project that performs **Sentiment Analysis on customer reviews** using **DistilBERT**, tracks experiments with **MLflow**, stores models in **AWS S3**, and serves predictions through a **FastAPI REST API** deployed on **AWS EC2 using Docker**.

---

# Project Overview

This project classifies user reviews into:

- 😊 Positive
- 😞 Negative

The model is trained using "Yelp Dataset" whuch was Collected from Kaggle. The dtaaset mainly contains the review given by the customer. 

---

# Architecture

```text
Yelp Dataset
      ↓
Data Preprocessing
      ↓
DistilBERT Fine-Tuning
      ↓
MLflow Experiment Tracking
      ↓
Model Registry
      ↓
AWS S3 Storage
      ↓
FastAPI API
      ↓
Docker Container
      ↓
AWS EC2 Deployment
```

---

# Features

- Fine-tuned DistilBERT model for sentiment classification
- Experiment tracking using MLflow
- Model artifact storage in AWS S3
- Automatic model download from S3 during startup
- REST API using FastAPI
- Dockerized deployment
- Cloud deployment on AWS EC2
- Interactive Swagger Documentation

---

# Project Structure

```text
Sentiment-Analysis/
│
├── app.py
├── train.py
├── download_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
│
├── sentiment_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
│
├── mlruns/
├── results/
├── sentiment-api.tar
└── README.md
```

---

# Tech Stack

## Machine Learning
- Python
- Hugging Face Transformers
- DistilBERT
- PyTorch
- Scikit-Learn

## MLOps
- MLflow
- Docker
- AWS S3
- AWS EC2

## Backend
- FastAPI
- Uvicorn

---

# Dataset

**Dataset Used:** Yelp Reviews Dataset

### Label Encoding

| Rating | Sentiment |
|---------|------------|
| 1 ⭐, 2 ⭐ | Negative |
| 4 ⭐, 5 ⭐ | Positive |
| 3 ⭐ | Removed |

---

# ⚙️ Model Training

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train.py
```

Model artifacts are saved inside:

```text
sentiment_model/
```

---

# MLflow Tracking

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Track:

- Parameters
- Metrics
- Artifacts
- Model Versions

---

# AWS S3 Integration

Upload model:

```bash
aws s3 cp sentiment_model/ s3://your-bucket-name/ --recursive
```

The application automatically downloads model files from S3 during startup.

---

# Running FastAPI Locally

Start server:

```bash
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# Docker Deployment

Build Docker image:

```bash
docker build -t sentiment-api .
```

Run container:

```bash
docker run -p 8000:8000 sentiment-api
```

---

# ☁️ AWS EC2 Deployment

SSH into EC2:

```bash
ssh -i sentiment-key.pem ubuntu@YOUR_PUBLIC_IP
```

Load Docker Image:

```bash
docker load -i sentiment-api.tar
```

Run Container:

```bash
docker run -d \
-p 8000:8000 \
--name sentiment-container \
sentiment-api
```

Enable Auto Restart:

```bash
docker update --restart unless-stopped sentiment-container
```

---

#  Live API

### Swagger Documentation

```text
http://YOUR_PUBLIC_IP:8000/docs
```

### Prediction Endpoint

```text
POST /predict
```

### Example Request

```json
{
    "review": "Amazing food and excellent service!"
}
```

### Example Response

```json
{
    "review": "Amazing food and excellent service!",
    "sentiment": "Positive",
    "confidence": 99.78
}
```

---

#  API Workflow

```text
User Review
      ↓
FastAPI Endpoint
      ↓
DistilBERT Model
      ↓
Prediction
      ↓
JSON Response
```

---


