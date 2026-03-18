output "jenkins_public_ip" {
  description = "Jenkins server public IP — open :8080 in browser"
  value       = module.jenkins.public_ip
}

output "jenkins_url" {
  description = "Jenkins URL"
  value       = "http://${module.jenkins.public_ip}:8080"
}

output "mlflow_url" {
  description = "MLflow experiment tracking URL"
  value       = "http://${module.mlflow.public_ip}:5000"
}

output "ecr_registry_url" {
  description = "ECR registry URL — use this in Jenkinsfile"
  value       = module.ecr.registry_url
}

output "eks_cluster_name" {
  description = "EKS cluster name — use in kubectl commands"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS API endpoint"
  value       = module.eks.cluster_endpoint
}

output "flask_app_url" {
  description = "Flask ML app URL (via ALB DNS — no Route53 needed)"
  value       = "http://${module.eks.alb_dns_name}"
}

output "grafana_url" {
  description = "Grafana dashboard URL (via ALB DNS)"
  value       = "http://${module.eks.alb_dns_name}:3000"
}

output "s3_raw_data_bucket" {
  description = "S3 bucket for raw data uploads"
  value       = module.s3.raw_data_bucket_name
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for model artifacts"
  value       = module.s3.artifacts_bucket_name
}

output "kubectl_config_command" {
  description = "Run this to configure kubectl for EKS"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
