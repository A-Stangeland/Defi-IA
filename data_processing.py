import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm

def create_tabluar_dataset(dataset_path="Data", subset="Train"):
    subset = subset.capitalize()
    if subset not in ["Train", "Test"]:
        raise ValueError("subset must be either \"Train\" or \"Test\".")
    X_station = pd.read_csv(os.path.join(dataset_path, f"{subset}/X_station_{subset.lower()}.csv"))
    baseline_forecast = pd.read_csv(os.path.join(dataset_path,f"{subset}/Baselines/Baseline_forecast_{subset.lower()}.csv"))
    baseline_forecast = baseline_forecast.rename(columns=dict(Prediction="baseline_fore"))
    baseline_observation = pd.read_csv(os.path.join(dataset_path,f"{subset}/Baselines/Baseline_observation_{subset.lower()}.csv"))
    baseline_observation = baseline_observation.rename(columns=dict(Prediction="baseline_obs"))
    station_coords = pd.read_csv(os.path.join(dataset_path,"Other/stations_coordinates.csv"))

    # X_station = pd.merge(X_station, station_coords, on="number_sta", how="left")
    X_station["Id_split"] = X_station["Id"].str.split("_")
    X_station["hour"] = X_station["Id_split"].apply(lambda s: s[-1]).astype(int)
    X_station["Id"] = X_station["Id_split"].apply(lambda s: s[:2]).str.join("_")
    X_station= X_station.drop(columns="Id_split")

    if subset == "Train":
        X_station['date'] = pd.to_datetime(X_station['date'])
        X_station["month"] = X_station["date"].dt.month
        Y_train = pd.read_csv(os.path.join(dataset_path,"Train/Y_train.csv"))
        df = Y_train[["Id", "Ground_truth"]].dropna()
        df = pd.merge(df, baseline_observation[["Id", "baseline_obs"]], on="Id", how="left")
        df = pd.merge(df, baseline_forecast[["Id", "baseline_fore"]], on="Id", how="left")
    else:
        df = pd.merge(baseline_observation, baseline_forecast, on="Id", how="left")
    
    df = pd.merge(df, X_station[["Id", "month"]].groupby("Id").first().reset_index(), on="Id", how="left")
    df["number_sta"] = df["Id"].str.split("_").apply(lambda s: s[0]).astype("int64")
    df = pd.merge(df, station_coords, on="number_sta", how="left")
    df = df.drop(columns="number_sta")

    for hour in tqdm(range(24)):
        X_subset = X_station.loc[X_station["hour"]==hour, ["Id", "ff","t","td","hu","dd","precip"]]
        df = pd.merge(df, X_subset, on="Id", how="left", suffixes=["", f"_{hour}"])

    df = df.rename(columns=dict(ff="ff_0", t="t_0", td="td_0", hu="hu_0", dd="dd_0", precip="precip_0"))

    return df

def get_tabular_dataset(dataset_path="Data", save=True):
    data_train = create_tabluar_dataset(dataset_path, subset="Train")
    data_test = create_tabluar_dataset(dataset_path, subset="Test")

    feature_cols = data_train.drop(columns="Ground_truth").columns
    data_test = data_test[feature_cols]

    data_train.to_csv(os.path.join(dataset_path, "Train/tabular_train.csv"), index=False)
    data_test.to_csv(os.path.join(dataset_path, "Test/tabular_test.csv"), index=False)

    return data_train, data_test



if __name__ == "__main__":
    get_tabular_dataset()