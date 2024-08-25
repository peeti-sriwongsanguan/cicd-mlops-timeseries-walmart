from src.data_preprocessing import load_and_preprocess_data, split_data

from src.plot_image import save_plot,plot_residuals,plot_feature_importance,plot_predictions_vs_actual, plot_model_comparison
from src.model import fit_arima, fit_sarima, fit_xgboost, prepare_dl_data, train_model, evaluate_model, LSTMModel, CNNModel, RNNModel, GRUModel
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import time
import optuna
import logging
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.utils import timing_decorator

@timing_decorator
def main():
    start_time = time.time()
    logging.info("Starting the prediction process...")
    zip_filepath = r'data/'
    csv_filename = 'walmart_cleaned.csv'
    # Load and preprocess data
    df = load_and_preprocess_data(f'{zip_filepath}{csv_filename}')
    # Split data
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    # Fit ARIMA and SARIMA models
    arima_mse, arima_pred = fit_arima(y_train, y_test)
    sarima_mse, sarima_pred = fit_sarima(y_train, y_test)

    # Fit XGBoost model
    xgb_mse, xgb_pred, xgb_model = fit_xgboost(X_train, y_train, X_test, y_test)

    # Prepare data for deep learning models
    train_loader, test_loader, X_test_tensor, y_test_tensor = prepare_dl_data(X_train, X_test, y_train, y_test)

    # Define and train deep learning models
    models = {
        'LSTM': LSTMModel(input_size=X_train.shape[1], hidden_size=50, num_layers=2),
        'CNN': CNNModel(input_size=X_train.shape[1]),
        'RNN': RNNModel(input_size=X_train.shape[1], hidden_size=50, num_layers=2),
        'GRU': GRUModel(input_size=X_train.shape[1], hidden_size=50, num_layers=2)
    }

    results = {}
    for name, model in models.items():
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer)
        mse, predictions = evaluate_model(model, X_test_tensor, y_test)
        results[name] = {'MSE': mse, 'Predictions': predictions}

    # Print results
    print(f"ARIMA MSE: {arima_mse}")
    print(f"SARIMA MSE: {sarima_mse}")
    print(f"XGBoost MSE: {xgb_mse}")
    for name, result in results.items():
        print(f"{name} MSE: {result['MSE']}")

    end_time = time.time()
    logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")


# 1. Fine-tuning XGBoost
@timing_decorator
def fine_tune_xgboost(X_train, y_train, X_test, y_test):
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'n_estimators': [100, 300, 500, 1000],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Best XGBoost parameters: {best_params}")
    print(f"XGBoost MSE: {mse}")

    return best_model, mse

@timing_decorator
def efficient_tune_xgboost(X_train, y_train, X_test, y_test, n_iter=50, cv=5):
    param_distributions = {
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 1000),
        'min_child_weight': randint(1, 6),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5)
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Best XGBoost parameters: {best_params}")
    print(f"XGBoost MSE: {mse}")

    return best_model, mse


# 2. Investigating CNN architecture
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"NaN indices: {torch.isnan(tensor).nonzero()}")
        print(f"Tensor statistics: Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")


class EfficientCNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(2)
        return self.fc(x)


def train_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping_patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_pred = model(X_val)
                val_loss += criterion(val_pred, y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Check for overfitting
        if epoch > 5 and train_loss < 0.5 * val_loss:
            print(f"Possible overfitting detected at epoch {epoch}")
            break

    return best_val_loss, train_losses, val_losses


def optimize_cnn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split training data into train and validation sets
    train_size = int(0.8 * len(X_train_scaled))
    X_train_tensor = torch.FloatTensor(X_train_scaled[:train_size])
    y_train_tensor = torch.FloatTensor(y_train[:train_size]).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_train_scaled[train_size:])
    y_val_tensor = torch.FloatTensor(y_train[train_size:]).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    def objective(trial):
        num_filters = trial.suggest_int('num_filters', 32, 128)  # Increased lower bound
        kernel_size = trial.suggest_int('kernel_size', 3, 5)  # Adjusted range based on best trials
        lr = trial.suggest_loguniform('lr', 1e-4, 5e-3)  # Adjusted upper bound
        batch_size = trial.suggest_categorical('batch_size', [64, 128])  # Removed 32 based on best trials

        model = EfficientCNNModel(X_train.shape[1], num_filters, kernel_size)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss, train_losses, val_losses = train_evaluate(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=30, early_stopping_patience=5
        )

        return best_val_loss

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=3600)  # 1 hour timeout

    best_params = study.best_params
    print(f"Best CNN parameters: {best_params}")

    best_model = EfficientCNNModel(X_train.shape[1], best_params['num_filters'], best_params['kernel_size'])

    # Train the best model
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['lr'])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    _, train_losses, val_losses = train_evaluate(
        best_model, train_loader, val_loader, criterion, optimizer,
        epochs=50, early_stopping_patience=10
    )

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    save_plot('cnn_loss_plot.png')
    plt.close()

    # Evaluate on test set
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(X_test_tensor).cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    return best_model, mse


# 3. Creating an ensemble model
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


# 4. Improving deep learning models (focusing on LSTM)
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(ImprovedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = self.fc(out[:, -1, :])
        return out

@timing_decorator
def train_lstm(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

@timing_decorator
def optimize_lstm(X_train, y_train, X_test, y_test):
    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        model = ImprovedLSTM(X_train.shape[1], hidden_size, num_layers, dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                                  batch_size=batch_size, shuffle=True)

        train_lstm(model, train_loader, criterion, optimizer, num_epochs=20)

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test)).numpy()
        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print(f"Best LSTM parameters: {best_params}")

    best_model = ImprovedLSTM(X_train.shape[1], best_params['hidden_size'],
                              best_params['num_layers'], best_params['dropout'])
    return best_model, study.best_value

def print_model_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print()


if __name__ == "__main__":
    # print(main())
    start_time = time.time()
    logging.info("Starting the prediction process...")
    zip_filepath = r'data/'
    csv_filename = 'walmart_cleaned.csv'
    # Load and preprocess data
    df = load_and_preprocess_data(f'{zip_filepath}{csv_filename}')
    # Split data
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    # # Fine-tune XGBoost
    # # xgb_model, xgb_mse = fine_tune_xgboost(X_train, y_train, X_test, y_test)
    #
    # best_xgb_model, xgb_mse = efficient_tune_xgboost(X_train, y_train, X_test, y_test)
    # # print(f"XGBoost MSE: {xgb_mse}")
    #
    # # Feature importance
    # feature_importance = best_xgb_model.feature_importances_
    # for i, importance in enumerate(feature_importance):
    #     print(f"Feature {i}: {importance}")
    #
    # # Corrected early stopping implementation
    # best_params = best_xgb_model.get_params()
    # best_params.pop('early_stopping_rounds', None)  # Remove if present
    # best_params.pop('eval_metric', None)  # Remove if present
    #
    # xgb_model_early_stopping = xgb.XGBRegressor(
    #     **best_params,
    #     early_stopping_rounds=10,
    #     eval_metric='rmse'
    # )
    # xgb_model_early_stopping.fit(
    #     X_train, y_train,
    #     eval_set=[(X_test, y_test)],
    #     verbose=False
    # )
    # y_pred_early_stopping = xgb_model_early_stopping.predict(X_test)
    # mse_early_stopping = mean_squared_error(y_test, y_pred_early_stopping)
    # print(f"XGBoost MSE with early stopping: {mse_early_stopping}")
    #
    #
    # # Optimize CNN
    # cnn_model, cnn_mse = optimize_cnn(X_train, y_train, X_test, y_test)
    # print(f"CNN MSE: {cnn_mse}")
    #
    # # Optimize LSTM
    # # lstm_model, lstm_mse = optimize_lstm(X_train, y_train, X_test, y_test)
    #
    # # Create ensemble
    # ensemble, ensemble_mse = create_ensemble(X_train, y_train, X_test, y_test,
    #                                          [best_xgb_model, cnn_model])
    #
    # print(f"XGBoost MSE: {xgb_mse}")
    # print(f"CNN MSE: {cnn_mse}")
    # # # print(f"LSTM MSE: {lstm_mse}")
    # print(f"Ensemble MSE: {ensemble_mse}")

    # Store feature names before converting to numpy arrays
    feature_names = df.columns.drop('Weekly_Sales').tolist()

    # XGBoost
    best_xgb_model, xgb_mse = efficient_tune_xgboost(X_train, y_train, X_test, y_test)
    xgb_pred = best_xgb_model.predict(X_test)
    print_model_metrics(y_test, xgb_pred, "XGBoost")
    plot_predictions_vs_actual(y_test, xgb_pred, "XGBoost")
    plot_residuals(y_test, xgb_pred, "XGBoost")
    plot_feature_importance(best_xgb_model, feature_names, "XGBoost")

    # CNN
    cnn_model, cnn_mse = optimize_cnn(X_train, y_train, X_test, y_test)
    cnn_pred = cnn_model(torch.FloatTensor(X_test)).detach().numpy().squeeze()
    print_model_metrics(y_test.values, cnn_pred, "CNN")
    plot_predictions_vs_actual(y_test, cnn_pred, "CNN")
    plot_residuals(y_test, cnn_pred, "CNN")

    # Ensemble
    ensemble, ensemble_mse = create_ensemble(X_train, y_train, X_test, y_test, [best_xgb_model, cnn_model])
    ensemble_pred = ensemble.predict(X_test)
    print_model_metrics(y_test.values, ensemble_pred, "Ensemble")
    plot_predictions_vs_actual(y_test, ensemble_pred, "Ensemble")
    plot_residuals(y_test, ensemble_pred, "Ensemble")

    # Comparison plot
    plot_model_comparison(y_test, xgb_pred, cnn_pred, ensemble_pred)

    print(f"XGBoost MSE: {xgb_mse}")
    print(f"CNN MSE: {cnn_mse}")
    # # print(f"LSTM MSE: {lstm_mse}")
    print(f"Ensemble MSE: {ensemble_mse}")

    end_time = time.time()
    logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")
