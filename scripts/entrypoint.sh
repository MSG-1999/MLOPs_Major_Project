#!/bin/bash
set -e

# Port handling for Cloud Run
PORT="${PORT:-8501}"

echo "Starting DreamForge Studio on port ${PORT}..."

# Start Prometheus metrics exporter in background if requested
if [ "$ENABLE_PROMETHEUS" = "true" ]; then
    echo "Prometheus enabled. Metrics will be exported."
fi

# Run the Streamlit application
exec streamlit run app/main.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
