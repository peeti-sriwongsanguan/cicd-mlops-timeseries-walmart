import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data, scale_features, split_data


def test_preprocess_data():
    # Create a sample dataframe
    data = {
        'Date': ['2012-01-01', '2012-01-08', '2012-01-15'],
        'Store': [1, 1, 2],
        'Dept': [1, 2, 1],
        'Weekly_Sales': [1000, 1200, 980],
        'Temperature': [32, 35, 28],
        'Fuel_Price': [2.5, 2.4, 2.6],
        'CPI': [211.0, 211.5, 212.0],
        'Unemployment': [8.9, 8.7, 8.8]
    }
    df = pd.DataFrame(data)

    processed_df = preprocess_data(df)

    # Check if new time-based features are created
    assert 'Year' in processed_df.columns
    assert 'Month' in processed_df.columns
    assert 'Day' in processed_df.columns
    assert 'DayOfWeek' in processed_df.columns

    # Check if categorical variables are encoded
    assert 'Store_2' in processed_df.columns
    assert 'Dept_2' in processed_df.columns

    # Check if lag features are created
    assert 'Sales_Lag_1' in processed_df.columns


def test_scale_features():
    data = {
        'Temperature': [32, 35, 28],
        'Fuel_Price': [2.5, 2.4, 2.6],
        'CPI': [211.0, 211.5, 212.0],
        'Unemployment': [8.9, 8.7, 8.8],
        'Year': [2012, 2012, 2012],
        'Month': [1, 1, 1],
        'Day': [1, 8, 15],
        'DayOfWeek': [6, 6, 6]
    }
    df = pd.DataFrame(data)

    scaled_df, scaler = scale_features(df)

    # Check if scaling was applied (mean should be close to 0, std close to 1)
    for column in df.columns:
        assert -0.1 < scaled_df[column].mean() < 0.1
        assert 0.9 < scaled_df[column].std() < 1.1


def test_split_data():
    data = {
        'Feature1': range(100),
        'Feature2': range(100, 200),
        'Weekly_Sales': range(200, 300)
    }
    df = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = split_data(df)

    # Check if the split is 80% train, 20% test
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    # Check if 'Weekly_Sales' is not in X_train or X_test
    assert 'Weekly_Sales' not in X_train.columns
    assert 'Weekly_Sales' not in X_test.columns
