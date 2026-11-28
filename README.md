# AutoFare: Auto-Rickshaw Fare Prediction (Bhopal)

Group ID: G158
Problem ID: MP103
Course: SCSE3040 MLOps, Bennett University
Faculty: Dr. Ravi Pratap

## Team

| Enrolment No. | Name |
|---|---|
| S24CSEU2528 | Akanksha Yadav |
| S24CSEU2522 | Erica Kapoor |
| S24CSEU2494 | Anupama Singh |
| S24CSEU2502 | Kuldeep Rajpoot |
| S24CSEU2508 | Sanvi Saparia |

## What this project does

Predicts the fare in rupees (fare_rs) of an auto-rickshaw ride in Bhopal. The metric is MAE. The pipeline covers data checks, cleaning, DVC data versioning, MLflow tracking and registry, a FastAPI service, drift monitoring (PSI), Docker, Kubernetes and GitHub Actions CI.

## Project layout

- src/autofare: data checks, cleaning, training, monitoring
- app/main.py: FastAPI service (GET /health, POST /predict)
- tests: pytest tests
- reports/findings.yaml: data issues, tracking and monitoring results
- predictions/holdout_predictions.csv: predictions for the holdout rows
- k8s: Kubernetes Deployment and Service
- Dockerfile: container image for the API

## Run the tests

    pip install -r requirements.txt
    python -m pytest -q

## Run the API

    uvicorn app.main:app --port 8000

Then open http://127.0.0.1:8000/docs

## Docker

    docker build -t mp103-api:1.0 .
    docker run -p 8000:8000 mp103-api:1.0

## Kubernetes

    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
