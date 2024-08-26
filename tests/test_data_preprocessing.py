import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_preprocess_data, split_data

def test_load_and_preprocess_data():
    # Create a sample CSV file for testing
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'Weekly_Sales': np.random.rand(100) * 1000,
        'Store': np.random.randint(1, 5, 100),
        'Dept': np.random.randint(1, 10, 100)
    })
    sample_file = 'test_data.csv'
    sample_data.to_csv(sample_file)

    # Test the function
    processed_data = load_and_preprocess_data(sample_file)

    # Assertions
    assert isinstance(processed_data, pd.DataFrame)
    assert 'Year' in processed_data.columns
    assert 'Month' in processed_data.columns
    assert 'Week' in processed_data.columns
    assert 'DayOfWeek' in processed_data.columns
    assert 'Quarter' in processed_data.columns
    assert 'Sales_Lag_1' in processed_data.columns
    assert 'Sales_RollingMean_4' in processed_data.columns

    # Clean up
    import os
    os.remove(sample_file)

def test_split_data():
    # Create sample data
    data = pd.DataFrame({
        'Weekly_Sales': np.random.rand(100) * 1000,
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100)
    })

    # Test the function
    X_train, X_test, y_train, y_test, scaler = split_data(data)

    # Assertions
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[1] == X_test.shape[1] == 2  # Number of features
    assert len(y_train) + len(y_test) == len(data)