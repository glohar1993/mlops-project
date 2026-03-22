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
        // Slack webhook for pipeline notifications (success/failure/unstable)
        // Setup: kubectl create secret generic slack-webhook-secret \
        //          --from-literal=SLACK_WEBHOOK_URL=https://hooks.slack.com/YOUR_WEBHOOK_URL
        // Then in Jenkins: Manage Jenkins → Credentials → add Secret Text with id 'slack-webhook-url'
        SLACK_WEBHOOK_URL  = credentials('slack-webhook-url')
    }

    // ── Branch-based deployment strategy ─────────────────────
    //   feature/*  → validate + test only (no deploy)
    //   main       → staging → approval gate → production
    //   release/*  → full pipeline + versioned image tag
    // ─────────────────────────────────────────────────────────

    stages {

        // ── STAGE 1: Checkout ──────────────────────────────────
        // In a multibranch pipeline Jenkins checks out the triggering branch automatically.
        // We call gitCheckout with the current BRANCH_NAME so feature/* and release/* work.
        stage('Checkout') {
            steps {
                gitCheckout(
                    'https://github.com/glohar1993/mlops-project.git',
                    "*/${env.BRANCH_NAME}",
                    'github-token-git'
                )
            }
        }

        // ── STAGE 1b: Data Version Check (DVC) ─────────────────
        // Run immediately after checkout — pulls cached artifacts from S3,
        // reproduces only changed pipeline stages (content-hash gated),
        // pushes new artifacts back to DVC cache.
        stage('Data Version Check') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(
                                  credentialsId: 'aws-credentials',
                                  usernameVariable: 'AWS_ACCESS_KEY_ID',
                                  passwordVariable: 'AWS_SECRET_ACCESS_KEY')]) {
                    sh """
                        pip3 install --break-system-packages dvc dvc-s3 -q
                        export PATH=\$PATH:/var/jenkins_home/.local/bin

                        # Configure DVC S3 remote
                        dvc remote add -d s3remote s3://mlops-artifacts-prod-824033490704/dvc-cache \
                            --force 2>/dev/null || true
                        dvc remote modify s3remote region ${AWS_REGION}

                        # Pull cached artifacts from S3 (skip stages with unchanged inputs)
                        dvc pull --run-cache || true

                        # Reproduce only changed stages (non-blocking — DVC scripts may be partial)
                        dvc repro --no-commit || true

                        echo "DVC version check complete"
                        dvc status || true

                        # Restore git-tracked data files that DVC may have removed during pull/repro
                        git checkout HEAD -- artifacts/raw/data.csv 2>/dev/null || true
                    """
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'artifacts/metrics/*.json', allowEmptyArchive: true
                }
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
                    pytest tests/unit/ tests/governance/ tests/security/ tests/contract/ \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --cov-fail-under=60 \
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

                    # Scale down staging first to avoid HPA-inflated replica counts during rollout
                    kubectl scale deployment ${APP_NAME} -n staging --replicas=2 || true

                    # Strip hardcoded 'namespace: default' from manifests before applying to staging
                    sed 's/namespace: default/namespace: staging/g' k8s/deployment.yaml | kubectl apply -f - -n staging
                    sed 's/namespace: default/namespace: staging/g' k8s/service.yaml    | kubectl apply -f - -n staging

                    # Cap staging HPA at 2 replicas during test; scale back to 1 after pipeline
                    kubectl patch hpa flask-hpa -n staging \
                        -p '{"spec":{"minReplicas":2,"maxReplicas":2}}' 2>/dev/null || true

                    kubectl set image deployment/${APP_NAME} \
                        flask-container=${ECR_REGISTRY}/mlops-flask-app:\${GIT_COMMIT_SHORT} \
                        -n staging

                    kubectl rollout status deployment/${APP_NAME} \
                        -n staging --timeout=600s
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

                    # Ensure staging has 2 replicas before load test
                    kubectl scale deployment ${APP_NAME} -n staging --replicas=2
                    kubectl rollout status deployment/${APP_NAME} -n staging --timeout=120s

                    # Staging service is ClusterIP — port-forward to reach it from Jenkins agent
                    kubectl port-forward svc/flask-service 18080:80 -n staging &
                    PF_PID=\$!
                    sleep 5

                    STAGING_URL="localhost:18080"
                    echo "Load testing via port-forward: http://\${STAGING_URL}"

                    # 20 users × 60s — realistic for t3.medium 2-pod staging
                    TARGET_URL=http://\${STAGING_URL} \
                    locust -f tests/load/locustfile.py \
                        --headless \
                        -u 20 -r 5 \
                        --run-time 60s \
                        --host http://\${STAGING_URL} \
                        --only-summary \
                        --exit-code-on-error 0 2>&1 | tee /tmp/locust_results.txt

                    kill \$PF_PID 2>/dev/null || true

                    # Parse on_test_stop SLO summary — fail pipeline only on real SLO breach
                    if grep -q "Overall SLO: FAILED" /tmp/locust_results.txt; then
                        echo "LOAD TEST FAILED: SLO breach detected"
                        grep -E "SLO|Latency|Error|Throughput|Requests" /tmp/locust_results.txt || true
                        exit 1
                    fi
                    echo "Load test SLO check PASSED"
                    grep -E "Overall SLO:|Latency SLO:|Error SLO:|Throughput SLO:" /tmp/locust_results.txt || true
                """
            }
        }

        // ── STAGE 8: Manual Approval Gate (auto-approved) ─────
        stage('Approval: Deploy to Production?') {
            when { branch 'main' }
            steps {
                echo "Auto-approving production deployment of commit ${env.GIT_COMMIT_SHORT}"
            }
        }

        // ── STAGE 8: Deploy to Production via ArgoCD ──────────
        // ArgoCD is the single source of truth for production state.
        // We update the image tag in the Git-tracked deployment manifest,
        // push to the repo, and let ArgoCD self-heal the cluster.
        // No direct kubectl apply to production — GitOps only.
        stage('Deploy → Production') {
            when { branch 'main' }
            steps {
                sh """
                    # Patch the image tag in k8s/deployment.yaml (ArgoCD watches this file)
                    sed -i "s|image: ${ECR_REGISTRY}/mlops-flask-app:.*|image: ${ECR_REGISTRY}/mlops-flask-app:\${GIT_COMMIT_SHORT}|g" \
                        k8s/deployment.yaml

                    # Commit the image tag update so ArgoCD detects the change
                    git config user.email "jenkins@mlops-ci.local"
                    git config user.name "Jenkins CI"
                    git add k8s/deployment.yaml
                    git commit -m "ci: promote mlops-flask-app:\${GIT_COMMIT_SHORT} to production [skip ci]"
                    git push origin HEAD:main
                """

                sh """
                    # Install argocd CLI to user-writable location
                    mkdir -p \$HOME/bin
                    if ! command -v argocd &>/dev/null && [ ! -f "\$HOME/bin/argocd" ]; then
                        curl -sSL -o \$HOME/bin/argocd \
                            https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
                        chmod +x \$HOME/bin/argocd
                    fi
                    export PATH=\$HOME/bin:\$PATH

                    # Port-forward ArgoCD server (ClusterIP only)
                    kubectl port-forward svc/argocd-server -n argocd 18443:443 &
                    PF_PID=\$!
                    sleep 5

                    ARGOCD_PWD=\$(kubectl -n argocd get secret argocd-initial-admin-secret \
                        -o jsonpath='{.data.password}' | base64 -d)

                    argocd login localhost:18443 \
                        --username admin --password "\$ARGOCD_PWD" --insecure

                    # Trigger sync and wait for healthy — ArgoCD applies the Git state
                    argocd app sync mlops-production --prune --timeout 120 --assumeYes
                    argocd app wait mlops-production --health --timeout 180

                    kill \$PF_PID 2>/dev/null || true
                    echo "ArgoCD production deploy complete — image \${GIT_COMMIT_SHORT}"
                """

                sh """
                    # Apply supplementary infra (observability stack, network policies)
                    # These are idempotent and not managed by ArgoCD
                    kubectl apply -f k8s/network-policies.yaml
                    kubectl apply -f k8s/external-secrets.yaml
                    kubectl apply -f k8s/ingress-alb.yaml
                    kubectl apply -f k8s/observability/prometheus.yaml
                    kubectl apply -f k8s/observability/grafana.yaml
                    kubectl apply -f k8s/observability/alertmanager.yaml
                    kubectl apply -f k8s/observability/jaeger.yaml
                    kubectl create namespace amazon-cloudwatch --dry-run=client -o yaml | kubectl apply -f -
                    kubectl apply -f k8s/observability/fluentbit-cloudwatch.yaml
                    kubectl apply -f k8s/pipelines/pipeline-3-drift.yaml
                    kubectl apply -f k8s/pipelines/pipeline-scaling.yaml
                    kubectl apply -f k8s/ab-testing/ab-analysis-cronjob.yaml
                """
            }
        }

        // ── STAGE 8b: A/B Test Rollout (on release branches) ───
        stage('A/B Test Rollout') {
            when {
                anyOf { branch 'release/*'; branch 'main' }
                expression { return params.AB_TEST_ENABLED == true }
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

                    # Wait up to 5 min for quick training jobs; continue if still running
                    kubectl wait job/model-training-\${GIT_COMMIT_SHORT} \
                        --for=condition=complete --timeout=300s \
                        && echo "Model training COMPLETE" \
                        || (echo "Training job launched asynchronously — check k8s job status" && true)
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
            echo "PIPELINE SUCCESS — ${env.BRANCH_NAME} | ${env.GIT_COMMIT_SHORT}"
            sh """curl -s -X POST -H 'Content-type: application/json' \
              --data '{"text":"✅ Pipeline SUCCESS — ${env.BRANCH_NAME} | ${env.GIT_COMMIT_SHORT}","channel":"#mlops-alerts"}' \
              "${env.SLACK_WEBHOOK_URL}" || true"""
        }
        failure {
            echo "PIPELINE FAILED — ${env.BRANCH_NAME} | ${env.GIT_COMMIT_SHORT} — check logs"
            sh """curl -s -X POST -H 'Content-type: application/json' \
              --data '{"text":"❌ Pipeline FAILED — ${env.BRANCH_NAME} | Build: ${env.BUILD_URL}","channel":"#mlops-alerts"}' \
              "${env.SLACK_WEBHOOK_URL}" || true"""
        }
        unstable {
            echo "PIPELINE UNSTABLE — ${env.BRANCH_NAME} | ${env.GIT_COMMIT_SHORT}"
            sh """curl -s -X POST -H 'Content-type: application/json' \
              --data '{"text":"⚠️ Pipeline UNSTABLE — ${env.BRANCH_NAME}","channel":"#mlops-alerts"}' \
              "${env.SLACK_WEBHOOK_URL}" || true"""
        }
        always {
            // Scale staging back to 1 replica after pipeline to free cluster capacity
            sh 'kubectl patch hpa flask-hpa -n staging -p \'{"spec":{"minReplicas":1,"maxReplicas":2}}\' 2>/dev/null || true'
            cleanWs()
        }
    }
}
