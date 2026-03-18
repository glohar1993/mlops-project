# ─────────────────────────────────────────────────────────────────
# MLflow Server on EC2
# Access: http://<public-ip>:5000
# Stores artifacts in S3, runs metadata in SQLite (upgrade to RDS later)
# ─────────────────────────────────────────────────────────────────

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "mlflow" {
  name   = "${var.project}-mlflow-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "MLflow UI"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-mlflow-sg" }
}

resource "aws_iam_role" "mlflow" {
  name = "${var.project}-mlflow-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "mlflow_s3" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  role       = aws_iam_role.mlflow.name
}

resource "aws_iam_instance_profile" "mlflow" {
  name = "${var.project}-mlflow-profile"
  role = aws_iam_role.mlflow.name
}

resource "aws_instance" "mlflow" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.mlflow.id]
  iam_instance_profile   = aws_iam_instance_profile.mlflow.id

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    apt-get update -y
    apt-get install -y python3-pip

    pip3 install mlflow boto3

    # Start MLflow server with S3 as artifact store
    mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root s3://${var.s3_bucket}/mlflow-artifacts/ \
      --host 0.0.0.0 \
      --port 5000 \
      --serve-artifacts &

    echo "MLflow started" > /tmp/mlflow_ready
  EOF

  tags = { Name = "${var.project}-mlflow-server" }
}
