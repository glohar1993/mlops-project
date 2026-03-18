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

  # Store state in S3 (after first apply, enable this)
  # backend "s3" {
  #   bucket = "mlops-terraform-state"
  #   key    = "mlops/terraform.tfstate"
  #   region = "us-east-1"
  # }
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
}
