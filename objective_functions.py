import numpy as np

def MAPE(y_true, y_pred, smooth=1):
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true+smooth)))

def mape_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return 100 - MAPE(np.exp(y) - 1, np.exp(y_pred) - 1)