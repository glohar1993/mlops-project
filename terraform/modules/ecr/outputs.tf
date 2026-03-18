data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

output "registry_url"          { value = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com" }
output "mlops_app_repo_url"    { value = aws_ecr_repository.mlops_app.repository_url }
output "drift_pipeline_repo_url" { value = aws_ecr_repository.drift_pipeline.repository_url }
