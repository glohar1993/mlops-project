@Library('jenkins-shared') _

// ============================================================
//  Enterprise MLOps CI/CD Pipeline
//  Stages: Checkout → Validate → Test → Scan → Build →
//          Deploy Staging → Integration Test → Approval →
//          Deploy Production → MLflow Train → Notify
// ============================================================

def ECR_REGISTRY = "824033490704.dkr.ecr.us-east-2.amazonaws.com"
def AWS_REGION   = "us-east-2"
def CLUSTER_NAME = "mlops-cluster"
def APP_NAME     = "flask-deployment"
def MLFLOW_URL   = "http://3.15.231.90:5000"

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
        timeout(time: 45, unit: 'MINUTES')
        disableConcurrentBuilds()               // No parallel runs on same branch
    }

    environment {
        ECR_REGISTRY       = "${ECR_REGISTRY}"
        AWS_REGION         = "${AWS_REGION}"
        MLFLOW_URL         = "${MLFLOW_URL}"
        AWS_DEFAULT_REGION = "${AWS_REGION}"
    }

    // ── Branch-based deployment strategy ─────────────────────
    //   feature/*  → validate + test only (no deploy)
    //   main       → staging → approval gate → production
    //   release/*  → full pipeline + versioned image tag
    // ─────────────────────────────────────────────────────────

    stages {

        // ── STAGE 1: Checkout ──────────────────────────────────
        stage('Checkout') {
            steps {
                gitCheckout(
                    'https://github.com/glohar1993/mlops-project.git',
                    '*/main',
                    'github-token-git'
                )
            }
        }

        // ── STAGE 2: Validate (parallel) ───────────────────────
        stage('Validate') {
            parallel {
                stage('Data Quality') {
                    steps {
                        sh """
                            pip3 install --break-system-packages setuptools -q
                            pip3 install --break-system-packages -r requirements.txt -q
                            export PATH=\$PATH:/var/jenkins_home/.local/bin
                            python3 -c "
import sys, pandas as pd
df = pd.read_csv('artifacts/raw/data.csv')
required = ['Operation_Mode','Temperature_C','Vibration_Hz','Power_Consumption_kW',
            'Network_Latency_ms','Packet_Loss_%','Quality_Control_Defect_Rate_%',
            'Production_Speed_units_per_hr','Predictive_Maintenance_Score',
            'Error_Rate_%','Efficiency_Status']
missing = [c for c in required if c not in df.columns]
null_pct = df.isnull().mean().max() * 100
if missing:
    print(f'FAIL: Missing columns: {missing}'); sys.exit(1)
if len(df) < 100:
    print(f'FAIL: Too few rows: {len(df)}'); sys.exit(1)
if null_pct > 20:
    print(f'FAIL: High null rate: {null_pct:.1f}%'); sys.exit(1)
print(f'Data validation PASSED — {len(df)} rows, {len(df.columns)} cols, null%={null_pct:.2f}')
"
                        """
                    }
                }
                stage('Code Lint') {
                    steps {
                        sh """
                            pip3 install --break-system-packages flake8 -q
                            export PATH=\$PATH:/var/jenkins_home/.local/bin
                            flake8 src/ application.py \
                                --max-line-length=120 \
                                --exclude=venv,__pycache__,.git \
                                --count || true
                        """
                    }
                }
            }
        }

        // ── STAGE 3: Unit Tests ────────────────────────────────
        stage('Unit Tests') {
            steps {
                sh """
                    pip3 install --break-system-packages pytest pytest-cov -q
                    export PATH=\$PATH:/var/jenkins_home/.local/bin
                    pytest tests/ \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --junitxml=test-results.xml \
                        -v || true
                """
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ── STAGE 4: Security Scan + Build + Push to ECR ───────
        stage('Build & Push to ECR') {
            when {
                anyOf { branch 'main'; branch 'release/*' }
            }
            steps {
                dockerBuildAndPush(ECR_REGISTRY, 'mlops-flask-app', AWS_REGION)
            }
        }

        stage('Build Drift Pipeline Image') {
            when { branch 'main' }
            steps {
                sh """
                    aws ecr get-login-password --region ${AWS_REGION} | \
                        docker login --username AWS --password-stdin ${ECR_REGISTRY}

                    docker buildx build \
                        --platform linux/amd64 \
                        -t ${ECR_REGISTRY}/mlops-drift-pipeline:latest \
                        -t ${ECR_REGISTRY}/mlops-drift-pipeline:\${GIT_COMMIT_SHORT} \
                        -f pipelines/Dockerfile.drift \
                        --push .
                """
            }
        }

        // ── STAGE 5: Deploy to Staging ─────────────────────────
        stage('Deploy → Staging') {
            when { branch 'main' }
            steps {
                installKubectl(CLUSTER_NAME, AWS_REGION)

                sh """
                    kubectl create namespace staging --dry-run=client -o yaml | kubectl apply -f -

                    kubectl apply -f k8s/deployment.yaml -n staging
                    kubectl apply -f k8s/service.yaml    -n staging

                    kubectl set image deployment/${APP_NAME} \
                        flask-container=${ECR_REGISTRY}/mlops-flask-app:\${GIT_COMMIT_SHORT} \
                        -n staging

                    kubectl rollout status deployment/${APP_NAME} \
                        -n staging --timeout=300s
                """
            }
        }

        // ── STAGE 6: Integration Tests (Staging) ───────────────
        stage('Integration Tests') {
            when { branch 'main' }
            steps {
                sh """
                    kubectl wait pod \
                        -l app=flask-app -n staging \
                        --for=condition=Ready --timeout=120s

                    STAGING_POD=\$(kubectl get pod -n staging -l app=flask-app \
                                   -o jsonpath='{.items[0].metadata.name}')

                    # Health check
                    kubectl exec \$STAGING_POD -n staging -- \
                        curl -sf http://localhost:5001/health

                    # Drift endpoint check
                    kubectl exec \$STAGING_POD -n staging -- \
                        curl -sf http://localhost:5001/drift

                    echo "Integration tests PASSED"
                """
            }
        }

        // ── STAGE 7: Manual Approval Gate ─────────────────────
        stage('Approval: Deploy to Production?') {
            when { branch 'main' }
            steps {
                timeout(time: 30, unit: 'MINUTES') {
                    input(
                        message: "Deploy commit ${env.GIT_COMMIT_SHORT} to PRODUCTION?",
                        ok: 'Deploy to Production',
                        submitter: 'admin,mlops-lead'
                    )
                }
            }
        }

        // ── STAGE 8: Deploy to Production ─────────────────────
        stage('Deploy → Production') {
            when { branch 'main' }
            steps {
                k8sDeploy('default', env.GIT_COMMIT_SHORT, ECR_REGISTRY, APP_NAME)

                sh """
                    kubectl apply -f k8s/observability/prometheus.yaml
                    kubectl apply -f k8s/observability/grafana.yaml
                    kubectl apply -f k8s/observability/alertmanager.yaml
                    kubectl apply -f k8s/pipelines/pipeline-3-drift.yaml
                    kubectl apply -f k8s/pipelines/pipeline-scaling.yaml
                """
            }
        }

        // ── STAGE 9: Trigger MLflow Model Training ─────────────
        stage('Trigger Model Training') {
            when { branch 'main' }
            steps {
                sh """
                    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-\${GIT_COMMIT_SHORT}
  labels:
    triggered-by: jenkins
    commit: \${GIT_COMMIT_SHORT}
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: ${ECR_REGISTRY}/mlops-flask-app:\${GIT_COMMIT_SHORT}
        command: ["python3", "src/model_training.py"]
        env:
        - name: MLFLOW_TRACKING_URI
          value: "${MLFLOW_URL}"
        - name: AWS_DEFAULT_REGION
          value: "${AWS_REGION}"
EOF

                    kubectl wait job/model-training-\${GIT_COMMIT_SHORT} \
                        --for=condition=complete --timeout=1200s \
                        && echo "Model training COMPLETE" \
                        || kubectl logs -l job-name=model-training-\${GIT_COMMIT_SHORT}
                """
            }
        }

        // ── STAGE 10: Deploy Airflow DAGs ──────────────────────
        stage('Sync Airflow DAGs to S3') {
            when { branch 'main' }
            steps {
                sh """
                    aws s3 sync dags/ \
                        s3://mlops-artifacts-prod-824033490704/airflow-dags/ \
                        --region ${AWS_REGION} \
                        --delete
                    echo "DAGs synced to S3"
                """
            }
        }

    } // end stages

    post {
        success {
            echo "PIPELINE SUCCESS — ${env.GIT_BRANCH_NAME} | ${env.GIT_COMMIT_SHORT}"
        }
        failure {
            echo "PIPELINE FAILED — ${env.GIT_BRANCH_NAME} | ${env.GIT_COMMIT_SHORT} — check logs"
        }
        always {
            cleanWs()
        }
    }
}
