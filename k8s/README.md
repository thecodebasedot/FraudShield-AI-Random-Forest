# Kubernetes deployment

Manifests to run FraudShield AI on Kubernetes with horizontal autoscaling, TLS
ingress, health probes and secret-managed credentials.

## Files

| File | Purpose |
|------|---------|
| `deployment.yaml` | API Deployment (3 replicas, probes, non-root, resource limits) |
| `service.yaml` | ClusterIP Service on port 80 → 8000 |
| `configmap.yaml` | Non-secret config (threshold, rate limit, Redis/Kafka hosts) |
| `secret.example.yaml` | **Template** for secrets (DB URL, encryption key, Slack) |
| `hpa.yaml` | HorizontalPodAutoscaler (3→20 pods on CPU/memory) |
| `ingress.yaml` | TLS ingress (nginx) with HTTPS redirect |

## Quick start

```bash
# 1. Build & push the image (or load it into your cluster)
docker build -t fraudshield-ai:latest .

# 2. Create the real Secret (never commit it)
kubectl create secret generic fraudshield-secrets \
  --from-literal=DATABASE_URL='postgresql+psycopg://user:pass@db:5432/fraudshield' \
  --from-literal=FRAUDSHIELD_ENC_KEY="$(python -m src.security genkey)"

# 3. Apply the rest
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Watch it scale
kubectl get pods -l app=fraudshield-api -w
```

Redis and Kafka can be added with the official Bitnami Helm charts, then point
`REDIS_URL` / `KAFKA_BOOTSTRAP_SERVERS` (in the ConfigMap) at their services.
