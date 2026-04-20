#!/bin/bash
# scripts/run_prometheus.sh — Start a dedicated Prometheus instance for DreamForge

echo "Starting DreamForge Monitoring (Prometheus)..."
echo "UI will be available at: http://localhost:9092"

# Use Docker to run Prometheus but point it to the host metrics
docker run -d \
    --name dreamforge_prometheus_standalone \
    -p 9092:9090 \
    -v $(pwd)/configs/prometheus.yml:/etc/prometheus/prometheus.yml \
    -v $(pwd)/configs/alert_rules.yml:/etc/prometheus/alert_rules.yml \
    --add-host host.docker.internal:host-gateway \
    prom/prometheus:v2.47.0

echo "Prometheus started. Check status with 'docker logs -f dreamforge_prometheus_standalone'"
