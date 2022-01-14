from data_processing import load_xgb_data
from xgboost_model import train_model, load_xgb_parameters

import numpy as np
import pandas as pd
import argparse
import pickle
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "data", help="dataset path")
    parser.add_argument("--output_folder", type=str, default = "output", help="path to the output folder")
    parser.add_argument("--model_save_path", type=str, default = "trained_models", help="path to the folder where the model will be saved")

    args = parser.parse_args()
    data_path = args.data_path
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
    model_save_path = args.model_save_path
    if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    data_train, data_test = load_xgb_data(path=data_path)

    y_train = np.log(1 + data_train["Ground_truth"])
    X_train = data_train.drop(columns=["Ground_truth", "Id"])
    X_test = data_test.drop(columns=["Id"])

    xgb_params = load_xgb_parameters("xgb_parameters.json")
    model = train_model(X_train, y_train, xgb_params)

    # Save the model
    model_name = "xgb_reg.pkl"
    with open(os.path.join(model_save_path, model_name), "wb") as f:
        pickle.dump(model, f)

    # Save test prediction
    prediction = model.predict(X_test)
    result = pd.DataFrame(dict(
        Id = data_test["Id"],
        Prediction = np.exp(prediction)
    ))

    result.to_csv(os.path.join(output_folder, "prediction.csv"), index=False)


if __name__=="__main__":
    main()