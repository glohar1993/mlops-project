output "raw_data_bucket_name"  { value = aws_s3_bucket.buckets["raw_data"].bucket }
output "artifacts_bucket_name" { value = aws_s3_bucket.buckets["artifacts"].bucket }
output "drift_bucket_name"     { value = aws_s3_bucket.buckets["drift_reports"].bucket }
