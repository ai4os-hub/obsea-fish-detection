@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/2.1.1']) _

def projectConfig

pipeline {
    agent any

    stages {
        stage('Application testing') {
            steps {
                script {
                    projectConfig = pipelineConfig()
                    buildStages(projectConfig)
                }
            }
        }
    }
    post {
        always {
            // Clean after build
            cleanWs()
        }    
    }
}
