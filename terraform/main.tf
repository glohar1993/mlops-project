terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }

  # Remote state — S3 + DynamoDB locking (safe multi-user collaboration)
  # Run once: aws s3 mb s3://mlops-terraform-state --region us-east-1
  #           aws dynamodb create-table --table-name terraform-lock \
  #             --attribute-definitions AttributeName=LockID,AttributeType=S \
  #             --key-schema AttributeName=LockID,KeyType=HASH \
  #             --billing-mode PAY_PER_REQUEST --region us-east-1
  backend "s3" {
    bucket         = "mlops-terraform-state-824033490704"
    key            = "mlops/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MLOps-Production"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "mlops-team"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_ca)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_ca)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ─────────────────────────────────────────
#  PHASE 1: INFRASTRUCTURE
# ─────────────────────────────────────────
module "vpc" {
  source      = "./modules/vpc"
  project     = var.project
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

module "ecr" {
  source      = "./modules/ecr"
  project     = var.project
  environment = var.environment
}

module "s3" {
  source      = "./modules/s3"
  project     = var.project
  environment = var.environment
}

module "eks" {
  source             = "./modules/eks"
  project            = var.project
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  node_instance_type = var.node_instance_type
  node_min           = var.node_min
  node_max           = var.node_max
}

# ─────────────────────────────────────────
#  PHASE 2: JENKINS (CI/CD SERVER)
# ─────────────────────────────────────────
module "jenkins" {
  source            = "./modules/jenkins"
  project           = var.project
  environment       = var.environment
  vpc_id            = module.vpc.vpc_id
  subnet_id         = module.vpc.public_subnet_ids[0]
  instance_type     = var.jenkins_instance_type
  ecr_registry_url  = module.ecr.registry_url
  s3_artifacts_bucket = module.s3.artifacts_bucket_name
}

# ─────────────────────────────────────────
#  PHASE 3: MLFLOW (EXPERIMENT TRACKING)
# ─────────────────────────────────────────
module "mlflow" {
  source        = "./modules/mlflow"
  project       = var.project
  environment   = var.environment
  vpc_id        = module.vpc.vpc_id
  subnet_id     = module.vpc.public_subnet_ids[0]
  instance_type = "t3.small"
  s3_bucket     = module.s3.artifacts_bucket_name
  # Pass RDS endpoint so MLflow EC2 startup uses PostgreSQL, not SQLite
  rds_endpoint  = module.rds.db_endpoint
  rds_secret_arn = module.rds.secret_arn
}

# ─────────────────────────────────────────
#  PHASE 3b: RDS PostgreSQL (MLflow Backend)
#  Replaces SQLite — supports concurrent DAG writes
# ─────────────────────────────────────────
# Data sources: look up existing VPC/subnets from running cluster
data "aws_vpc" "existing" {
  id = "vpc-045c88da3a33c68da"
}

data "aws_subnets" "existing" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.existing.id]
  }
}

module "rds" {
  source      = "./modules/rds"
  project     = var.project
  environment = var.environment
  vpc_id      = data.aws_vpc.existing.id

  # Use existing public subnets (demo env — private subnets not provisioned)
  private_subnet_ids         = data.aws_subnets.existing.ids
  allowed_security_group_ids = ["sg-095e130ccbf3f437e"]  # EKS cluster SG

  instance_class = "db.t3.micro"
}
