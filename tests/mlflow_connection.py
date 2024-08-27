import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.mlflow_config import setup_mlflow, TRACKING_URI
import mlflow

setup_mlflow()
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

try:
    # List all experiments
    experiments = mlflow.search_experiments()
    print("Available experiments:")
    for exp in experiments:
        print(f"- {exp.name} (ID: {exp.experiment_id})")

    # Try to get or create a specific experiment
    experiment_name = "Test-Experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        print(f"Found existing experiment '{experiment_name}' with ID: {experiment.experiment_id}")

    print("Successfully connected to MLflow server")
except Exception as e:
    print(f"Failed to connect to MLflow server: {e}")