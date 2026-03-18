#!/bin/bash
# ============================================================
# CLEANUP — Run this when done to avoid charges!
# ============================================================
set -e

echo "WARNING: This will DELETE all AWS resources!"
read -p "Are you sure? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "Cancelled."
  exit 0
fi

cd "$(dirname "$0")"

echo "Deleting Kubernetes resources first..."
kubectl delete -f ../k8s/ --recursive --ignore-not-found=true 2>/dev/null || true

echo "Destroying Terraform infrastructure..."
terraform destroy -auto-approve

echo ""
echo "All AWS resources deleted. No more charges!"
