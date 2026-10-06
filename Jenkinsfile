// CI/CD: GitHub -> Jenkins -> Checkout -> Testing -> Docker Build -> Deployment
//
// Job setup (once):
//   1. Jenkins: New Item -> Pipeline -> Pipeline script from SCM -> Git
//   2. Repository URL: this GitHub repo. Script path: Jenkinsfile. Branch: */main
//   3. GitHub repo -> Settings -> Webhooks -> http://<jenkins-host>/github-webhook/
//      (or enable "GitHub hook trigger for GITScm polling" on the job)
//   4. The Jenkins agent needs Python 3, pip, and Docker.

pipeline {
    agent any

    environment {
        IMAGE_NAME = 'myopia-screening'
        IMAGE_TAG  = "${env.BUILD_NUMBER}"
        DEPLOY_PORT = '8000'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Testing') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt -r requirements-test.txt
                    pytest
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -t "${IMAGE_NAME}:latest" .
                '''
            }
        }

        stage('Deployment') {
            steps {
                sh '''
                    docker rm -f myopia-screening || true
                    docker run -d \
                        --name myopia-screening \
                        --restart unless-stopped \
                        -p "${DEPLOY_PORT}:8000" \
                        -e PORT=8000 \
                        -e PYTHONUNBUFFERED=1 \
                        "${IMAGE_NAME}:${IMAGE_TAG}"
                    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
                        if python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:${DEPLOY_PORT}/api/health').read().decode())"; then
                            echo "Deployed ${IMAGE_NAME}:${IMAGE_TAG} on port ${DEPLOY_PORT}"
                            exit 0
                        fi
                        sleep 5
                    done
                    echo "Health check failed"
                    docker logs myopia-screening || true
                    exit 1
                '''
            }
        }
    }

    post {
        failure {
            sh 'docker logs myopia-screening || true'
        }
    }
}
