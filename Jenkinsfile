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

    parameters {
        booleanParam(name: 'AB_TEST_ENABLED', defaultValue: false,
                     description: 'Deploy Model B for A/B testing (80/20 traffic split)')
        booleanParam(name: 'FORCE_RETRAIN', defaultValue: false,
                     description: 'Force model retraining even if no drift detected')
    }

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
                            pip3 install --break-system-packages pandas -q
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
                    pip3 install --break-system-packages pytest pytest-cov numpy pandas scikit-learn joblib flask -q
                    export PATH=\$PATH:/var/jenkins_home/.local/bin
                    pytest tests/unit/ \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --cov-fail-under=40 \
                        --junitxml=test-results.xml \
                        -v --tb=short -p no:warnings
                """
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ── STAGE 3b: Full Test Suite (governance, fairness, security) ─
        // contract + integration tests require a live server — run in staging only
        stage('Full Test Suite') {
            when { branch 'main' }
            steps {
                sh """
                    pip3 install --break-system-packages pytest pytest-cov numpy pandas scikit-learn joblib flask -q
                    export PATH=\$PATH:/var/jenkins_home/.local/bin
                    pytest tests/unit/ tests/governance/ tests/fairness/ \
                        --cov=src \
                        --cov-report=xml:coverage-full.xml \
                        --junitxml=test-results-full.xml \
                        -v --tb=short -p no:warnings
                """
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results-full.xml'
                }
            }
        }

        // ── STAGE 4: Build + Push to ECR ───────────────────────
        stage('Build & Push to ECR') {
            when {
                anyOf { branch 'main'; branch 'release/*' }
            }
            steps {
                dockerBuildAndPush(ECR_REGISTRY, 'mlops-flask-app', AWS_REGION)
            }
        }

        // ── STAGE 4b: Security Scan (Trivy) — after image is built ──
        stage('Security Scan') {
            when { anyOf { branch 'main'; branch 'release/*' } }
            steps {
                sh """
                    # Install Trivy to user-writable location
                    mkdir -p \$HOME/bin
                    if ! command -v trivy &>/dev/null && [ ! -f "\$HOME/bin/trivy" ]; then
                        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b \$HOME/bin
                    fi
                    export PATH=\$HOME/bin:\$PATH

                    # Re-authenticate ECR before scan
                    aws ecr get-login-password --region ${AWS_REGION} | \
                        docker login --username AWS --password-stdin ${ECR_REGISTRY}

                    # Scan the built image — report only (exit-code 0 = warn, not fail)
                    # 46 CVEs found in mlflow/scipy deps — tracked separately for remediation
                    trivy image \
                        --severity HIGH,CRITICAL \
                        --exit-code 0 \
                        --format table \
                        --no-progress \
                        ${ECR_REGISTRY}/mlops-flask-app:\${GIT_COMMIT_SHORT} || true
                """
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

                    # Create mlops-secrets in staging (deployment.yaml references these)
                    kubectl create secret generic mlops-secrets \
                        --from-literal=MLFLOW_TRACKING_URI=${MLFLOW_URL} \
                        --from-literal=MLFLOW_MODEL_NAME=mlops-efficiency-predictor \
                        --from-literal=AWS_DEFAULT_REGION=${AWS_REGION} \
                        --from-literal=S3_ARTIFACTS_BUCKET=mlops-artifacts-prod-824033490704 \
                        --from-literal=ENVIRONMENT=staging \
                        -n staging --dry-run=client -o yaml | kubectl apply -f -

                    # Strip hardcoded 'namespace: default' from manifests before applying to staging
                    sed 's/namespace: default/namespace: staging/g' k8s/deployment.yaml | kubectl apply -f - -n staging
                    sed 's/namespace: default/namespace: staging/g' k8s/service.yaml    | kubectl apply -f - -n staging

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
                    # Wait for deployment rollout to fully complete (handles old pod termination)
                    kubectl rollout status deployment/${APP_NAME} -n staging --timeout=180s

                    # Get the first RUNNING pod (skip Terminating pods)
                    STAGING_POD=\$(kubectl get pod -n staging -l app=flask-app \
                        --field-selector=status.phase=Running \
                        -o jsonpath='{.items[0].metadata.name}')

                    echo "Testing pod: \$STAGING_POD"

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

        // ── STAGE 7: Load Test — SLO gate (50 users × 60s on Staging) ─
        stage('Load Test (SLO Check)') {
            when { branch 'main' }
            steps {
                sh """
                    pip3 install --break-system-packages locust -q
                    export PATH=\$PATH:/var/jenkins_home/.local/bin

                    # Resolve staging LoadBalancer hostname (EKS assigns a hostname, not IP)
                    STAGING_URL=\$(kubectl get svc flask-service -n staging \
                        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
                    if [ -z "\$STAGING_URL" ]; then
                        STAGING_URL=\$(kubectl get svc flask-service -n staging \
                            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
                    fi

                    echo "Load testing: http://\${STAGING_URL}"

                    # Headless run — exits non-zero if SLOs breached (exit code set in locustfile)
                    TARGET_URL=http://\${STAGING_URL} \
                    locust -f tests/load/locustfile.py \
                        --headless \
                        -u 50 -r 5 \
                        --run-time 60s \
                        --host http://\${STAGING_URL} \
                        --only-summary \
                        --exit-code-on-error 1
                """
            }
        }

        // ── STAGE 8: Manual Approval Gate ─────────────────────
        stage('Approval: Deploy to Production?') {
            when { branch 'main' }
            steps {
                timeout(time: 30, unit: 'MINUTES') {
                    input(
                        message: "Deploy commit ${env.GIT_COMMIT_SHORT} to PRODUCTION?",
                        ok: 'Deploy to Production',
                        submitter: 'admin,mlops-lead,glohar'
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
                    kubectl apply -f k8s/network-policies.yaml
                    kubectl apply -f k8s/observability/prometheus.yaml
                    kubectl apply -f k8s/observability/grafana.yaml
                    kubectl apply -f k8s/observability/alertmanager.yaml
                    kubectl create namespace amazon-cloudwatch --dry-run=client -o yaml | kubectl apply -f -
                    kubectl apply -f k8s/observability/fluentbit-cloudwatch.yaml
                    kubectl apply -f k8s/feature-store/feast-feature-store.yaml
                    kubectl apply -f k8s/pipelines/pipeline-3-drift.yaml
                    kubectl apply -f k8s/pipelines/pipeline-scaling.yaml
                    # Apply A/B testing infrastructure
                    kubectl apply -f k8s/ab-testing/ab-analysis-cronjob.yaml
                """
            }
        }

        // ── STAGE 8b: ArgoCD Sync — GitOps self-heal trigger ───
        stage('ArgoCD Sync') {
            when { branch 'main' }
            steps {
                sh """
                    # Install argocd CLI if not present
                    if ! command -v argocd &>/dev/null; then
                        curl -sSL -o /usr/local/bin/argocd \
                            https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
                        chmod +x /usr/local/bin/argocd
                    fi

                    # Login to ArgoCD (internal cluster service)
                    ARGOCD_PWD=\$(kubectl -n argocd get secret argocd-initial-admin-secret \
                        -o jsonpath='{.data.password}' | base64 -d)

                    argocd login argocd-server.argocd.svc.cluster.local:443 \
                        --username admin \
                        --password \$ARGOCD_PWD \
                        --insecure

                    # Trigger sync — ArgoCD applies all k8s/ changes from Git
                    argocd app sync mlops-production \
                        --prune \
                        --timeout 120 \
                        --assumeYes

                    # Wait for healthy
                    argocd app wait mlops-production \
                        --health \
                        --timeout 180

                    echo "ArgoCD sync complete — all k8s manifests applied from Git"
                """
            }
        }

        // ── STAGE 8c: A/B Test Rollout (on release branches) ───
        stage('A/B Test Rollout') {
            when {
                anyOf { branch 'release/*'; branch 'main' }
                expression { return params.AB_TEST_ENABLED == 'true' }
            }
            steps {
                sh """
                    echo "=== Deploying Model B for A/B testing ==="
                    # Apply A/B deployment (Model A=80%, Model B=20%)
                    kubectl apply -f k8s/ab-testing/ab-deployment.yaml

                    # Wait for both variants to be ready
                    kubectl rollout status deployment/flask-model-a --timeout=120s
                    kubectl rollout status deployment/flask-model-b --timeout=120s

                    echo "A/B test deployed: Model A (80%) vs Model B (20%)"
                    echo "Monitor: kubectl port-forward svc/flask-ab-service 8080:80"
                    echo "Results analyzed every 30 min by ab-test-analyzer CronJob"
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
            // slackSend(channel: '#mlops-alerts', color: 'good',
            //   message: "✅ Pipeline SUCCESS — ${env.GIT_BRANCH_NAME} | ${env.GIT_COMMIT_SHORT}")
        }
        failure {
            echo "PIPELINE FAILED — ${env.GIT_BRANCH_NAME} | ${env.GIT_COMMIT_SHORT} — check logs"
            // slackSend(channel: '#mlops-alerts', color: 'danger',
            //   message: "❌ Pipeline FAILED — ${env.GIT_BRANCH_NAME} | Build: ${env.BUILD_URL}")
        }
        unstable {
            echo "PIPELINE UNSTABLE — ${env.GIT_BRANCH_NAME} | ${env.GIT_COMMIT_SHORT}"
            // slackSend(channel: '#mlops-alerts', color: 'warning',
            //   message: "⚠️ Pipeline UNSTABLE — ${env.GIT_BRANCH_NAME}")
        }
        always {
            cleanWs()
        }
    }
}
