from data_processing import load_xgb_data
from xgboost_model import train_model

import numpy as np
import pandas as pd
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "data", help="dataset path")
    parser.add_argument("--output_folder", type=str, default = "output", help="path to the output folder")

    args = parser.parse_args()
    data_path = args.data_path
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    data_train, data_test = load_xgb_data(path=data_path)

    y_train = np.log(1 + data_train["Ground_truth"])
    X_train = data_train.drop(columns=["Ground_truth", "Id"])
    X_test = data_test.drop(columns=["Id"])

    xgb_params=dict(
        n_estimators=100, 
        tree_method = 'gpu_hist', 
        max_depth = 5,
        learning_rate = .1,
        gpu_id = 0,
        verbosity=1
    )
    model = train_model(X_train, y_train, xgb_params)

    prediction = model.predict(X_test)
    result = pd.DataFrame(dict(
        Id = data_test["Id"],
        Prediction = np.exp(prediction)
    ))

    result.to_csv(os.path.join(output_folder, "prediction.csv"), index=False)


if __name__=="__main__":
    main()