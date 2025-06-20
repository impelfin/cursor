pipeline {
    agent any

    environment {
        APP_DIR = 'node/hello'
    }

    tools {
        nodejs 'NodeJS_18' 
    }

    // 파이프라인 단계 정의
    stages {
        // 1. Git 저장소에서 소스 코드 복제 단계
        // stage('Git Clone') {
        //     steps {
        //         echo 'Git 저장소에서 소스 코드 복제 중...'
        //         git branch: 'main', url: 'https://github.com/impelfin/cursor.git'
        //     }
        // }

        // 2. 소스 코드 체크아웃 확인 단계 (복제 후 확인 메시지)
        stage('Checkout') {
            steps {
                script {
                    echo '소스 코드 체크아웃 확인 중...'
                }
            }
        }

        // 3. PM2 설치 단계 (Node.js 프로세스 관리자)
        stage('Install PM2') {
            steps {
                echo "PM2(프로세스 관리자) 설치 중..."
                sh 'npm install -g pm2'
            }
        }

        // 4. 의존성 설치 단계
        stage('Install Dependencies') {
            steps {
                dir(APP_DIR) {
                    echo "Node.js 의존성 설치 중..."
                    sh 'npm install'
                }
            }
        }

        // 5. 테스트 단계 (선택 사항)
        // stage('Test') {
        //     steps {
        //         dir(APP_DIR) {
        //             echo "애플리케이션 테스트 실행 중..."
        //             sh 'echo "테스트 단계는 선택 사항입니다. `npm test`를 실행하려면 주석을 해제하세요."'
        //         }
        //     }
        // }

        // 6. 빌드 단계 
        // stage('Build') {
        //     steps {
        //         dir(APP_DIR) {
        //             echo "애플리케이션 빌드 중..."
        //             sh 'echo "별도의 빌드 단계가 필요 없습니다. 의존성 설치로 충분합니다."'
        //         }
        //     }
        // }

        // 7. 애플리케이션 실행/배포 단계
        stage('Run Application') {
            steps {
                dir(APP_DIR) {
                    echo "애플리케이션 실행 중 (PM2 사용)..."
                    sh 'pm2 start hello.js --name my-nodejs-app || pm2 reload my-nodejs-app'

                    echo "애플리케이션이 PM2를 통해 백그라운드에서 시작/재시작되었습니다."
                    echo "PM2는 Jenkins 빌드와 독립적으로 애플리케이션을 관리합니다."
                }
            }
        }
    }

    post {
        always {
            echo '파이프라인이 완료되었습니다.'
        }
        success {
            echo '빌드가 성공했습니다!'
        }
        failure {
            echo '빌드가 실패했습니다.'
        }
    }
}