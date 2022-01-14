import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import json


def load_xgb_parameters(fname):
    with open(fname, mode="r") as f:
        xgb_params = json.load(f)
    return xgb_params

def load_trained_xgb_model(fname):
    model = XGBRegressor()
    model.load_model("model_sklearn.json")
    return model

def train_model(
        X, y, 
        xgb_params=None, 
        use_gpu=False,
        test_split=None, 
        eval_set=None,
        early_stopping_rounds=10):
    
    if xgb_params is None:
        tree_method = "gpu_hist" if use_gpu else "auto"
        xgb_params = dict(tree_method=tree_method)
    elif isinstance(xgb_params, str):
        xgb_params = load_xgb_parameters(xgb_params)
    if eval_set is None and test_split is not None:
        X_train, y_train, X_val, y_val = train_test_split(X, y, test_split)
        eval_set = (X_val, y_val)
    else:
        X_train, y_train = X, y
        early_stopping_rounds = None

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
    return model