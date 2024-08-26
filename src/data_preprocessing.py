import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path).iloc[:, 1:]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature engineering
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Week'] = data.index.isocalendar().week
    data['DayOfWeek'] = data.index.dayofweek
    data['Quarter'] = data.index.quarter

    # Add lag features
    for lag in [1, 2, 4, 8, 12]:
        data[f'Sales_Lag_{lag}'] = data['Weekly_Sales'].shift(lag)

    # Add rolling mean features
    for window in [4, 8, 12]:
        data[f'Sales_RollingMean_{window}'] = data['Weekly_Sales'].rolling(window=window).mean()

    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data


# Split the data
def split_data(data, target_col='Weekly_Sales', test_size=0.2):
    y = data[target_col]
    X = data.drop(target_col, axis=1)

    # Use TimeSeriesSplit for more appropriate evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
