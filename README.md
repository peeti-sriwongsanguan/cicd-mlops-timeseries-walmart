# CICD End-to-End MLOPS Workflows

## Walmart Sales Forecasting: XGBoost vs CNN
This is a simple end-to-end mlops project which takes data from [Walmart stores](https://www.kaggle.com/datasets/ujjwalchowdhury/walmartcleaned) (special thanks to Ujjwal Chowdhury for the cleaned dataset) and transforms it with machine learning pipelines from training, model tracking and experimenting with Docker. For the models I compare XGBoost and CNN models for time series forecast.

For the dataset, it contains economic conditions like the Consumer Price Index (CPI), unemployment rate (Unemployment Index, etc). In this project, I want to try a new package called Optuna. Optuna is an open-source hyperparameter optimization framework developed by Preferred Networks, Inc. It provides a flexible and efficient platform for optimizing machine learning model hyperparameters, allowing users to find the best set of hyperparameters for their models automatically.

## Key findings
- XGBoost performs best overall, with the most accurate and consistent predictions
- Time-based features (like rolling means) and department are crucial for predictions.
- Some features at the bottom (e.g., MarkDown2, DayOfWeek) have very little impact on predictions.

## Now let's dive into the analysis

### Environment
- Machine: MacBook with M3 Pro chip (MPS device used for GPU acceleration)
- Python version: 3.9

## Project Structure
```
walmart-sales-forecast/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── plot_image.py
│   └── utils.py
│
├── image/
├── data/
│
├── tests/
│   └── test_data_preprocessing.py
│
├── .gitignore
├── conftest.py
├── main.py
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── setup.cfg
└── README.md
  ```

### Setup

#### 1. Clone the repository:
```
git clone https://github.com/peeti-sriwongsanguan/mlops-timeseries-walmart.git

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

#### 5. Make sure the Walmart dataset and place it in the data folder


### Docker
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
2.2. Otherwise, use this command to run the container:
```
docker run walmart-sales-forecast
```

