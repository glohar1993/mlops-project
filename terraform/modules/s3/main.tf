# ─────────────────────────────────────────────────────────────────
# S3 Buckets
#   - raw-data     : CSV uploads, streaming data
#   - artifacts    : model.pkl, scaler.pkl, mlflow artifacts
#   - drift-reports: JSON drift reports per run
# ─────────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  buckets = {
    raw_data      = "${var.project}-raw-data-${var.environment}-${local.account_id}"
    artifacts     = "${var.project}-artifacts-${var.environment}-${local.account_id}"
    drift_reports = "${var.project}-drift-reports-${var.environment}-${local.account_id}"
  }
}

resource "aws_s3_bucket" "buckets" {
  for_each = local.buckets
  bucket   = each.value
  tags     = { Name = each.value, Purpose = each.key }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "buckets" {
  for_each = aws_s3_bucket.buckets

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning on artifacts bucket (keep model history)
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.buckets["artifacts"].id
  versioning_configuration { status = "Enabled" }
}

# Lifecycle: move old artifacts to Glacier after 30 days
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.buckets["artifacts"].id

  rule {
    id     = "archive-old-models"
    status = "Enabled"
    filter { prefix = "models/" }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }
  }
}
