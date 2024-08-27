from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
import numpy as np

from src.mlflow_config import setup_mlflow, start_run
from src.data_preprocessing import load_and_preprocess_data, split_data
from src.plot_image import plot_residuals, plot_feature_importance, plot_predictions_vs_actual, plot_model_comparison
from src.model import fit_arima, fit_sarima, fit_xgboost, prepare_dl_data, train_model, evaluate_model, LSTMModel, CNNModel, RNNModel, GRUModel, fit_pytorch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import pandas as pd

pd.set_option('display.max_columns', None)

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.utils import timing_decorator


class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1 / len(models)] * len(models)

    def predict(self, X):
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                # For scikit-learn style models (e.g., XGBoost)
                pred = model.predict(X)
            else:
                # For PyTorch models
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(X)).cpu().numpy().squeeze()
            predictions.append(pred)
        return np.average(predictions, axis=0, weights=self.weights)


@timing_decorator
def create_ensemble(X_train, y_train, X_test, y_test, models):
    ensemble = EnsembleModel(models)
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Ensemble MSE: {mse}")
    return ensemble, mse


@timing_decorator
def main():
    # Setup MLflow
    setup_mlflow()

    start_time = time.time()
    logging.info("Starting the prediction process...")
    zip_filepath = r'data/'
    csv_filename = 'walmart_cleaned.csv'

    # Load and preprocess data
    df = load_and_preprocess_data(f'{zip_filepath}{csv_filename}')

    # Split data
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    # Store feature names
    feature_names = df.columns.drop('Weekly_Sales').tolist()

    # Fit ARIMA and SARIMA models
    with start_run("ARIMA"):
        arima_mse, arima_pred = fit_arima(y_train, y_test)
        mlflow.log_metric("mse", arima_mse)

    with start_run("SARIMA"):
        sarima_mse, sarima_pred = fit_sarima(y_train, y_test)
        mlflow.log_metric("mse", sarima_mse)

    # Fit XGBoost model
    with start_run("XGBoost"):
        xgb_mse, xgb_pred, xgb_model = fit_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_metric("mse", xgb_mse)
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        plot_predictions_vs_actual(y_test, xgb_pred, "XGBoost")
        plot_residuals(y_test, xgb_pred, "XGBoost")
        plot_feature_importance(xgb_model, feature_names, "XGBoost")

    # Prepare data for deep learning models
    train_loader, test_loader, X_test_tensor, y_test_tensor = prepare_dl_data(X_train, X_test, y_train, y_test)

    # Define and train deep learning models
    dl_models = {
        'LSTM': LSTMModel,
        'CNN': CNNModel,
        'RNN': RNNModel,
        'GRU': GRUModel
    }

    results = {}
    for name, model_class in dl_models.items():
        with start_run(name):
            mse, predictions, model = fit_pytorch_model(model_class, name, X_train, y_train, X_test, y_test)
            results[name] = {'MSE': mse, 'Predictions': predictions, 'Model': model}
            mlflow.log_metric("mse", mse)
            mlflow.pytorch.log_model(model, f"{name.lower()}_model")
            plot_predictions_vs_actual(y_test, predictions, name)
            plot_residuals(y_test, predictions, name)

    # Create ensemble
    with start_run("Ensemble"):
        ensemble_models = [xgb_model] + [results[name]['Model'] for name in dl_models]
        ensemble, ensemble_mse = create_ensemble(X_train, y_train, X_test, y_test, ensemble_models)
        ensemble_pred = ensemble.predict(X_test)
        mlflow.log_metric("mse", ensemble_mse)
        plot_predictions_vs_actual(y_test, ensemble_pred, "Ensemble")
        plot_residuals(y_test, ensemble_pred, "Ensemble")

    # Comparison plot
    plot_model_comparison(y_test, xgb_pred, results['CNN']['Predictions'], ensemble_pred)

    # Print results
    print(f"ARIMA MSE: {arima_mse}")
    print(f"SARIMA MSE: {sarima_mse}")
    print(f"XGBoost MSE: {xgb_mse}")
    for name, result in results.items():
        print(f"{name} MSE: {result['MSE']}")
    print(f"Ensemble MSE: {ensemble_mse}")

    end_time = time.time()
    logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
