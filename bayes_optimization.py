from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.callbacks import CheckpointSaver
from xgboost import XGBRegressor
from objective_functions import MAPE, mape_scorer
from data_processing import load_xgb_data
import numpy as np
import argparse
import pickle
import json
import os

def _param_map(param_args):
    param_type = param_args["type"]
    if param_type == "Integer":
        Type = Integer
    elif param_type == "Real":
        Type = Real
    else:
        raise TypeError(f"Only Integer and Real types are accepted. Got {param_type}.")
    return Type(param_args["min"], param_args["max"], param_args["prior"])

def load_parameter_space(fname="parameter_space.json"):
    with open(fname, mode="r") as f:
        param_dict = json.load(f)
    
    parameter_space = {param_name: _param_map(param_args) for param_name, param_args in param_dict.items()}
    return parameter_space

def bayes_search_xgb(
        X_train, y_train,
        X_val, y_val,
        use_gpu=False,
        n_iter=10,
        cv=3,
        verbose=3):
    
    parameter_space = load_parameter_space()
    tree_method = "gpu_hist" if use_gpu else "auto"
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method=tree_method
    )

    opt = BayesSearchCV(
        model,
        parameter_space,
        scoring=mape_scorer,
        fit_params=dict(eval_set=[(X_val, y_val)], early_stopping_rounds=10),
        n_iter=n_iter,
        cv=cv,
        verbose=verbose
    )

    checkpoint_saver = CheckpointSaver(f"opt_checkpoints/checkpoint.pkl", compress=9)
    opt.fit(X_train, y_train, callback=checkpoint_saver)

    return model, opt

if __name__ == "__main__":
    data_train, data_test = load_xgb_data()
    y_train = np.log(1 + data_train["Ground_truth"])
    X_train = data_train.drop(columns=["Ground_truth", "Id"])
    X_test = data_test.drop(columns=["Id"])
    model, opt = bayes_search_xgb()

    # Save the model
    model_name = "xgb_opt.pkl"
    with open(os.path.join("trained_modls", model_name), "wb") as f:
        pickle.dump(model, f)