from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def performance_metric(y_true, y_pred):
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    corr = round(np.corrcoef(y_pred, y_true)[1, 0], 3)
    return mae, mse, corr
