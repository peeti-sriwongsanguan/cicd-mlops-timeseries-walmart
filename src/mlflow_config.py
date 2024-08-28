from dotenv import load_dotenv
import os
import mlflow
from contextlib import contextmanager

load_dotenv()

# S3 bucket for storing artifacts
ARTIFACT_STORE = "s3://peeti/mlflow-artifacts"

# MLflow tracking server URL (use localhost for local tracking)
TRACKING_URI = "http://127.0.0.1:5001"


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://s3.amazonaws.com'

    if not all(key in os.environ for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']):
        raise EnvironmentError("AWS credentials not found in environment variables")


@contextmanager
def start_run(experiment_name=None, run_name=None):
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run
