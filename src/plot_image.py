import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import logging
import os
import matplotlib
matplotlib.use('Agg')


logging.basicConfig(level=logging.INFO)


def save_plot(filename):
    directory = "/app/image"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)
    logging.info(f"Plot saved to {filepath}")


def plot_predictions_vs_actual(y_true, y_pred, model_name):
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true.values, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Predictions vs Actual')
        save_plot(f'{model_name.lower()}_predictions_vs_actual.png')
    except Exception as e:
        logging.error(f"Error in plot_predictions_vs_actual: {str(e)}")
    finally:
        plt.close()


def plot_residuals(y_true, y_pred, model_name):
    try:
        # Convert to numpy arrays if they're not already
        y_true = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        y_pred = y_pred.values if hasattr(y_pred, 'values') else np.array(y_pred)

        residuals = y_true - y_pred

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
    except Exception as e:
        logging.error(f"Error in plot_residuals: {str(e)}")
    finally:
        plt.close()


def plot_feature_importance(model, feature_names, model_name):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name}: Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            save_plot(f'{model_name.lower()}_feature_importance.png')
        else:
            logging.warning(f"Model {model_name} does not have feature_importances_ attribute")
    except Exception as e:
        logging.error(f"Error in plot_feature_importance: {str(e)}")
    finally:
        plt.close()


def plot_model_comparison(y_test, xgb_pred, cnn_pred, ensemble_pred):
    try:
        plt.figure(figsize=(12, 6))
        arrays = [y_test, xgb_pred, cnn_pred, ensemble_pred]
        if not all(len(arr) == len(y_test) for arr in arrays):
            raise ValueError("All input arrays must have the same length")
        index = np.arange(len(y_test))
        for arr, label in zip(arrays, ['Actual', 'XGBoost', 'CNN', 'Ensemble']):
            plt.plot(index, arr, label=label, alpha=0.7)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Model Comparison: Actual vs Predictions')
        plt.legend()
        save_plot('model_comparison.png')
    except Exception as e:
        logging.error(f"Error in plot_model_comparison: {str(e)}")
    finally:
        plt.close()
