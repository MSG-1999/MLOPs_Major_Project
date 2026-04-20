#!/bin/bash
# scripts/run_mlflow.sh — Start MLflow tracking server

mkdir -p mlflow-artifacts

echo "Starting MLflow server..."
echo "UI: http://localhost:5012"
echo "Press Ctrl+C to stop."

mlflow server \
    --host 0.0.0.0 \
    --port 5012 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --serve-artifacts
