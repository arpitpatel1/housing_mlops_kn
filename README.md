# Real Estate Analysis Backend - MLOps Practice

Hey there! 👋 This project serves as the backend for a simple real estate analysis app I built for some machine learning fun. It's like a little playground where I practiced MLOps magic – making models, deploying them, and keeping an eye on things.

The primary goal is to provide a practical demonstration of employing MLOps technologies in a machine learning projects.

If you want to check out the whole shebang, jump over to the [Real Estate Analysis Frontend](https://ahmedabad-housing.streamlit.app/). It'll take you to the frontend app where the fun happens.

This GitHub repo is basically my MLOps practice diary. Feel free to poke around and see what mischief I got up to!

## Technologies and Tools

- [DVC (Data Version Control)](https://dvc.org/)
- [DVC Pipeline](https://dvc.org/doc/pipelines)
- [DVC Live (optional)](https://dvc.org/doc/live)
- [MLflow](https://mlflow.org/)
- [Docker](https://www.docker.com/)
- [GitHub Actions](https://github.com/features/actions)
- [AWS](https://aws.amazon.com/)
  - S3
  - ECR (Elastic Container Registry)
  - EC2 (Elastic Compute Cloud)
- [Kubernetes](https://kubernetes.io/)
- [Seldon](https://www.seldon.io/)
- [Prometheus](https://prometheus.io/)

## Workflow

1. **Data Versioning with DVC:**
   - Data collection and preprocessing are managed using DVC for efficient version control.
   - Dataset changes trigger the DVC pipeline, automating the data processing workflow.

2. **Model Development and Tracking with MLflow:**
   - MLflow is utilized for managing the end-to-end machine learning lifecycle.
   - Experiment tracking allows monitoring of model performance and hyperparameters.

3. **Containerization with Docker:**
   - Models are encapsulated within Docker containers, ensuring consistency across different environments.

4. **Continuous Integration with GitHub Actions:**
   - GitHub Actions are configured to run automated tests on each push, ensuring code quality and reliability.

5. **AWS Cloud Integration:**
   - AWS S3 is employed for storing datasets securely.
   - Docker images are hosted on AWS ECR for easy deployment.
   - EC2 instances facilitate scalable computing resources for model training and serving.

6. **Kubernetes Deployment with Seldon:**
   - Kubernetes is utilized for orchestrating containerized applications, enabling efficient scaling and deployment.
   - Seldon Core is integrated for deploying machine learning models on Kubernetes.

7. **Monitoring with Prometheus:**
   - Prometheus is employed for real-time monitoring of the deployed models, ensuring reliability and performance.

This workflow provides a practical example of how MLOps practices can be applied to any machine learning project, promoting efficient model development, deployment, and ongoing monitoring.
