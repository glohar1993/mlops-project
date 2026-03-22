variable "project"       { type = string }
variable "environment"   { type = string }
variable "vpc_id"        { type = string }
variable "subnet_id"     { type = string }
variable "instance_type" { type = string }
variable "s3_bucket"     { type = string }
variable "rds_endpoint" {
  type    = string
  default = ""
}
variable "rds_secret_arn" {
  type    = string
  default = ""
}
