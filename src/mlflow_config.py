import os
import mlflow

# S3 bucket for storing artifacts
ARTIFACT_STORE = "s3://peeti/mlflow-artifacts"

# MLflow tracking server URL (use localhost for local tracking)
TRACKING_URI = "http://localhost:5001"


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://s3.amazonaws.com'

    if not all(key in os.environ for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']):
        raise EnvironmentError("AWS credentials not found in environment variables")


def start_run(experiment_name):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()
