# ─────────────────────────────────────────────────────────────────
# Jenkins EC2 — CI/CD server
# Auto-installs: Docker, Jenkins, AWS CLI, kubectl
# Access: http://<public-ip>:8080
# ─────────────────────────────────────────────────────────────────

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_key_pair" "jenkins" {
  key_name   = "${var.project}-jenkins-key"
  public_key = tls_private_key.jenkins.public_key_openssh
}

resource "tls_private_key" "jenkins" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Save private key locally to SSH into Jenkins
resource "local_file" "jenkins_private_key" {
  content         = tls_private_key.jenkins.private_key_pem
  filename        = "${path.root}/jenkins-key.pem"
  file_permission = "0400"
}

resource "aws_security_group" "jenkins" {
  name   = "${var.project}-jenkins-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Jenkins web UI"
  }

  ingress {
    from_port   = 50000
    to_port     = 50000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Jenkins agent port"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project}-jenkins-sg" }
}

resource "aws_iam_role" "jenkins" {
  name = "${var.project}-jenkins-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "jenkins_ecr" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
  role       = aws_iam_role.jenkins.name
}

resource "aws_iam_role_policy_attachment" "jenkins_eks" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.jenkins.name
}

resource "aws_iam_role_policy_attachment" "jenkins_s3" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  role       = aws_iam_role.jenkins.name
}

resource "aws_iam_instance_profile" "jenkins" {
  name = "${var.project}-jenkins-profile"
  role = aws_iam_role.jenkins.name
}

resource "aws_instance" "jenkins" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  key_name               = aws_key_pair.jenkins.key_name
  vpc_security_group_ids = [aws_security_group.jenkins.id]
  iam_instance_profile   = aws_iam_instance_profile.jenkins.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  # Bootstrap script — installs everything automatically
  user_data = <<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update -y
    apt-get install -y curl wget unzip git

    # Install Docker
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu
    systemctl enable docker
    systemctl start docker

    # Install AWS CLI
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install

    # Install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl && mv kubectl /usr/local/bin/

    # Install Jenkins
    wget -O /usr/share/keyrings/jenkins-keyring.asc https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key
    echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/" > /etc/apt/sources.list.d/jenkins.list
    apt-get update -y
    apt-get install -y fontconfig openjdk-17-jre jenkins

    systemctl enable jenkins
    systemctl start jenkins

    # Add jenkins user to docker group
    usermod -aG docker jenkins
    systemctl restart jenkins

    # Configure AWS region
    mkdir -p /var/lib/jenkins/.aws
    echo "[default]" > /var/lib/jenkins/.aws/config
    echo "region = ${var.ecr_registry_url}" >> /var/lib/jenkins/.aws/config
    chown -R jenkins:jenkins /var/lib/jenkins/.aws

    echo "JENKINS_SETUP_COMPLETE" > /tmp/jenkins_ready
  EOF

  tags = { Name = "${var.project}-jenkins-server" }
}
