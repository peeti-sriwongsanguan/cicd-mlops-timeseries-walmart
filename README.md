# CICD End-to-End MLOPS Workflows

## Walmart Sales Forecasting: Machine Learning vs Deep Learning
### Description
This is a simple end-to-end mlops project which takes data from [Walmart stores](https://www.kaggle.com/datasets/ujjwalchowdhury/walmartcleaned) (special thanks to Ujjwal Chowdhury for the cleaned dataset) and transforms it with machine learning pipelines from training, model tracking and experimenting with Docker. 
This project implements various time series forecasting models using PyTorch to predict the store sales, including ARIMA, SARIMA, XGBoost, and deep learning models (LSTM, CNN, RNN, GRU). It uses [MLflow](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html) for experiment tracking and optionally Docker for environment consistency.

The project runs locally and uses AWS S3 buckets to store model artifacts during model tracking and experimenting with MLflow.

### Dataset
Available in Kaggle [Walmart stores](https://www.kaggle.com/datasets/ujjwalchowdhury/walmartcleaned), It contains economic conditions like the Consumer Price Index (CPI), unemployment rate (Unemployment Index, etc). 

### My approach 
In this project, I want to try a new package called Optuna. Optuna is an open-source hyperparameter optimization framework developed by Preferred Networks, Inc. It provides a flexible and efficient platform for optimizing machine learning model hyperparameters, allowing users to find the best set of hyperparameters for their models automatically.

## Key findings
- XGBoost performs best overall, with the most accurate and consistent predictions
- Time-based features (like rolling means) and department are crucial for predictions.
- Some features at the bottom (e.g., MarkDown2, DayOfWeek) have very little impact on predictions.

## Now let's dive into the analysis

### Environment & Prerequisites
- Machine: MacBook with M3 Pro chip (MPS device used for GPU acceleration)
- Python version: 3.9
- Docker (optional)
- AWS account with S3 bucket (for MLflow artifact storage)

## Project Structure
```
walmart-sales-forecast/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── mlflow_config.py
│   ├── model.py
│   ├── plot_image.py
│   └── utils.py
│
├── image/
├── data/
│
├── tests/
│   ├── mlflow_connection.py
│   └── test_data_preprocessing.py
│
├── .gitignore
├── .env
├── conftest.py
├── main.py
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── setup.cfg
└── README.md
  ```

### Setup
#### Local Setup

#### 1. Clone the repository:
```
git clone https://github.com/peeti-sriwongsanguan/cicd-mlops-timeseries-walmart.git

cd walmart-sales-forecast
```

#### 2. Install Pipenv if you haven't already:
```
pip install pipenv
```

#### 3. Install dependencies and create a virtual environment:
```
pipenv install --dev
```

#### 4. Activate the virtual environment:
```
pipenv shell
```

#### 5. Set up environment variables:
Create a .env file in the project root and add:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_aws_region
```

#### 6. Use a .env file (for local development):

Install the python-dotenv package:
```
pipenv install python-dotenv
```

#### 7. Make sure the Walmart dataset and place it in the data folder


### Docker (optional)
#### To run the project in a Docker container:

1. Build the Docker image:
```
docker build -t walmart-sales-forecast .
```
![2024-08-25_21-44-45 (1).gif](image%2F2024-08-25_21-44-45%20%281%29.gif)

2.1. Run the container that also create and save image in the image folder 

This command mounts the image directory from your current working directory to /app/image in the container.
```
docker run -v $(pwd)/image:/app/image walmart-sales-forecast
```
2.2. Docker Execution:
```
docker run --env-file .env -p 5001:5001 walmart-sales-forecast
```

### Running the Project
#### Local Execution

1. Start the MLflow tracking server:

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://your-bucket-name/mlflow-artifacts --host 0.0.0.0 --port 5001
```

2. Run the main script:
```
python main.py
```

## Viewing Results in MLflow

1. Ensure the MLflow tracking server is running.
   This is how to run the server as a background process 
   ```
   nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001 > mlflow.log 2>&1 &
   ```
   
   *Note:*
   To stop the server later, you'll need to find its process ID and kill it:
   ```
   ps aux | grep mlflow
   kill <process_id>
   ```

2. Open a web browser and navigate to http://localhost:5001.
![MLflow scnsht.gif](image/MLflow%20scnsht.gif)
3. In the MLflow UI:
   - View a list of all runs under the "Walmart Sales Forecast" experiment.
   - Click on individual runs to see detailed metrics, parameters, and artifacts.
   - Compare runs to analyze the performance of different models.
4. To view plots and other artifacts:
   - Click on a specific run.
   - Navigate to the "Artifacts" section.
   - Click on image files to view plots.
5. To load a specific model:
   - Note the run ID of the model you want to load.
   - Use MLflow's Python API to load the model:
   
   ```
   mlflow.set_tracking_uri("http://localhost:5001")
   model = mlflow.pyfunc.load_model(f"runs:/<run_id>/model")
   ```