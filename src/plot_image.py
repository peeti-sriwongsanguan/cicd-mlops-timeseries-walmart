import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def save_plot(filename):
    directory = "image"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

def plot_predictions_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true.values, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Predictions vs Actual')
    save_plot(f'{model_name.lower()}_predictions_vs_actual.png')
    plt.close()


def plot_residuals(y_true, y_pred, model_name):
    residuals = (y_true - y_pred).values.flatten()  # Ensure it's a flat numpy array

    plt.figure(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = plt.hist(residuals, bins=30, density=True, alpha=0.7)

    # Add a kernel density estimate
    kde = stats.gaussian_kde(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    p = kde(x)
    plt.plot(x, p, 'k', linewidth=2)

    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title(f'{model_name}: Residual Distribution')
    save_plot(f'{model_name.lower()}_residuals.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(f"{model_name}: Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        save_plot(f'{model_name.lower()}_feature_importance.png')
        plt.close()
    else:
        print(f"Model {model_name} does not have feature_importances_ attribute")


def plot_model_comparison(y_test, xgb_pred, cnn_pred, ensemble_pred):
    plt.figure(figsize=(12, 6))

    # Convert all inputs to numpy arrays
    y_test = np.array(y_test)
    xgb_pred = np.array(xgb_pred)
    cnn_pred = np.array(cnn_pred)
    ensemble_pred = np.array(ensemble_pred)

    # Check if all arrays have the same length
    if not all(len(arr) == len(y_test) for arr in [xgb_pred, cnn_pred, ensemble_pred]):
        raise ValueError("All input arrays must have the same length")

    # Create a simple index
    index = np.arange(len(y_test))

    plt.plot(index, y_test, label='Actual', alpha=0.7)
    plt.plot(index, xgb_pred, label='XGBoost', alpha=0.7)
    plt.plot(index, cnn_pred, label='CNN', alpha=0.7)
    plt.plot(index, ensemble_pred, label='Ensemble', alpha=0.7)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Model Comparison: Actual vs Predictions')
    plt.legend()
    save_plot('model_comparison.png')
    plt.close()