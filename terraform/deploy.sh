#!/bin/bash
# ============================================================
# MLOps Full AWS Deployment Script
# Run this ONCE after AWS CLI is configured
# ============================================================
set -e

AWS_REGION="us-east-2"
PROJECT="mlops"
ECR_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="$ECR_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "======================================================"
echo "PHASE 1: Provision AWS Infrastructure with Terraform"
echo "======================================================"
cd "$(dirname "$0")"
terraform init
terraform plan -out=tfplan
terraform apply tfplan

echo ""
echo "======================================================"
echo "PHASE 2: Configure kubectl for EKS"
echo "======================================================"
CLUSTER_NAME=$(terraform output -raw eks_cluster_name)
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME
kubectl get nodes

echo ""
echo "======================================================"
echo "PHASE 3: Build & Push Docker Images to ECR"
echo "======================================================"
cd ..

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# Build and push Flask app
docker build -t $ECR_REGISTRY/$PROJECT-flask-app:latest .
docker push $ECR_REGISTRY/$PROJECT-flask-app:latest

# Build and push drift pipeline
docker build -t $ECR_REGISTRY/$PROJECT-drift-pipeline:latest \
  -f pipelines/Dockerfile.drift .
docker push $ECR_REGISTRY/$PROJECT-drift-pipeline:latest

echo ""
echo "======================================================"
echo "PHASE 4: Deploy Apps to EKS"
echo "======================================================"

# Update image references in K8s manifests
ECR_APP_IMAGE="$ECR_REGISTRY/$PROJECT-flask-app:latest"
ECR_DRIFT_IMAGE="$ECR_REGISTRY/$PROJECT-drift-pipeline:latest"

# Deploy Flask app
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Update Flask deployment to use ECR image
kubectl set image deployment/flask-deployment \
  flask-container=$ECR_APP_IMAGE

# Deploy observability stack
kubectl apply -f k8s/observability/prometheus.yaml
kubectl apply -f k8s/observability/grafana.yaml
kubectl apply -f k8s/observability/alertmanager.yaml

# Deploy drift detection CronJob
kubectl apply -f k8s/pipelines/pipeline-3-drift.yaml

# Update drift CronJob image
kubectl set image cronjob/drift-detection-pipeline \
  drift-detector=$ECR_DRIFT_IMAGE

# Deploy HPA
kubectl apply -f k8s/pipelines/pipeline-scaling.yaml

echo ""
echo "======================================================"
echo "PHASE 5: Wait for pods to be ready"
echo "======================================================"
kubectl rollout status deployment/flask-deployment --timeout=120s
kubectl rollout status deployment/prometheus --timeout=120s
kubectl rollout status deployment/grafana --timeout=120s

echo ""
echo "======================================================"
echo "DEPLOYMENT COMPLETE — Your URLs:"
echo "======================================================"
JENKINS_IP=$(cd terraform && terraform output -raw jenkins_public_ip)
MLFLOW_IP=$(cd terraform && terraform output -raw mlflow_url)

echo ""
echo "  Jenkins    : http://$JENKINS_IP:8080"
echo "  MLflow     : $MLFLOW_IP"
echo "  Flask App  : use 'kubectl port-forward' or NodePort"
echo "  Prometheus : kubectl port-forward svc/prometheus-service 9090:9090"
echo "  Grafana    : kubectl port-forward svc/grafana-service 3000:3000"
echo ""
echo "  SSH to Jenkins:"
cd terraform && terraform output -raw kubectl_config_command
echo ""
echo "  To connect kubectl:"
echo "  aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME"
echo ""
echo "======================================================"
echo "  IMPORTANT: Run 'terraform destroy' when done!"
echo "  Otherwise you will keep getting charged!"
echo "======================================================"
