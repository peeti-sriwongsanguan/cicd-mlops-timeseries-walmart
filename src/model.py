from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import xgboost as xgb
import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna


# ARIMA Model
def fit_arima(y_train, y_test):
    arima_model = sm.tsa.ARIMA(y_train, order=(5, 1, 0))
    arima_result = arima_model.fit()
    arima_pred = arima_result.forecast(steps=len(y_test))
    arima_mse = mean_squared_error(y_test, arima_pred)
    return arima_mse, arima_pred


# SARIMA Model
def fit_sarima(y_train, y_test):
    sarima_model = sm.tsa.SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit()
    sarima_pred = sarima_result.forecast(steps=len(y_test))
    sarima_mse = mean_squared_error(y_test, sarima_pred)
    return sarima_mse, sarima_pred


# XGBoost Model with Optuna hyperparameter tuning
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse


def fit_xgboost(X_train, y_train, X_test, y_test):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100)

    best_params = study.best_params
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    return xgb_mse, xgb_pred, xgb_model


# Deep Learning Models Setup
def prepare_dl_data(X_train, X_test, y_train, y_test, batch_size=64):
    device = get_device()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test_tensor, y_test_tensor


# Generic training function for PyTorch models
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    device = get_device()
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()


# Generic evaluation function for PyTorch models
def evaluate_model(model, X_test_tensor, y_test):
    device = get_device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        predictions = model(X_test_tensor).cpu().squeeze()
    mse = mean_squared_error(y_test, predictions.numpy())
    return mse, predictions


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=2)
        self.fc = nn.Linear(64 * (input_size - 1), 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.rnn(x.unsqueeze(1), h_0)
        out = self.fc(out[:, -1, :])
        return out


# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.gru(x.unsqueeze(1), h_0)
        out = self.fc(out[:, -1, :])
        return out
