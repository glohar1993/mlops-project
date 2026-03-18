# ─────────────────────────────────────────────────────────────────
# ECR — Private Docker registry (replaces DockerHub)
# ─────────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "mlops_app" {
  name                 = "${var.project}-flask-app"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true   # auto scan images for CVEs
  }

  tags = { Name = "${var.project}-flask-app-ecr" }
}

resource "aws_ecr_repository" "drift_pipeline" {
  name                 = "${var.project}-drift-pipeline"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration { scan_on_push = true }
}

# Lifecycle policy: keep only last 5 images to save storage cost
resource "aws_ecr_lifecycle_policy" "mlops_app" {
  repository = aws_ecr_repository.mlops_app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}
