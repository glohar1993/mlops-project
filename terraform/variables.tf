variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-2"
}

variable "project" {
  description = "Project name — used for naming all resources"
  type        = string
  default     = "mlops"
}

variable "environment" {
  description = "Environment: dev / staging / prod"
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS worker nodes"
  type        = string
  default     = "t3.medium"   # change to t3.large for production
}

variable "node_min" {
  description = "Minimum EKS nodes"
  type        = number
  default     = 1
}

variable "node_max" {
  description = "Maximum EKS nodes (HPA scales pods, Karpenter scales nodes)"
  type        = number
  default     = 5
}

variable "jenkins_instance_type" {
  description = "EC2 instance type for Jenkins server"
  type        = string
  default     = "t3.medium"
}
