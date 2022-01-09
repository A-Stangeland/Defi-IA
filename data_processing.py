import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd

class DatasetGenerator:
    def __init__(self) -> None:
        pass

def create_tabular_dataset(dataset_path="Data", subset="Train"):
    subset = subset.capitalize()
    if subset not in ["Train", "Test"]:
        raise ValueError("subset must be either \"Train\" or \"Test\".")

    print("Loading data...", end="")
    X_station = pd.read_csv(os.path.join(dataset_path, f"{subset}/X_station_{subset.lower()}.csv"))
    baseline_forecast = pd.read_csv(os.path.join(dataset_path,f"{subset}/Baselines/Baseline_forecast_{subset.lower()}.csv"))
    baseline_forecast = baseline_forecast.rename(columns=dict(Prediction="baseline_pred"))
    baseline_observation = pd.read_csv(os.path.join(dataset_path,f"{subset}/Baselines/Baseline_observation_{subset.lower()}.csv"))
    baseline_observation = baseline_observation.rename(columns=dict(Prediction="baseline_obs"))
    station_coords = pd.read_csv(os.path.join(dataset_path,"Other/stations_coordinates.csv"))
    print(" Done.")

    print("Creating features...", end="")
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
        df = pd.merge(df, baseline_forecast[["Id", "baseline_pred"]], on="Id", how="left")
    else:
        df = pd.merge(baseline_observation, baseline_forecast, on="Id", how="left")

    month = X_station[["Id", "month"]].groupby("Id").first()
    df = df.merge(month, on="Id", how="left")
    df["number_sta"] = df["Id"].str.split("_").apply(lambda s: s[0]).astype(int)
    df = pd.merge(df, station_coords, on="number_sta", how="left")
    df = df.drop(columns="number_sta")
    
    feature_names = ["precip", "t", "td", "hu", "ff", "dd"]
    for feature in feature_names:
        print(f" {feature}", end="")
        feature_pivot = X_station[["Id", "hour", feature]].pivot(index="Id", columns="hour", values=feature)
        feature_pivot.columns = [f"{feature}_{h}" for h in range(24)]
        if feature == "precip":
            cumul = feature_pivot.sum(axis=1).to_frame("precip_cumul").reset_index()
            df = pd.merge(df, cumul, on="Id", how="left")
            df = pd.merge(df, feature_pivot.reset_index(), on="Id", how="left")
        else:
            mean = feature_pivot.mean(axis=1).to_frame(f"mean_{feature}").reset_index()
            std = feature_pivot.std(axis=1).to_frame(f"std_{feature}").reset_index()
            df = pd.merge(df, mean, on="Id", how="left")
            df = pd.merge(df, std, on="Id", how="left")
            df = pd.merge(df, feature_pivot.reset_index(), on="Id", how="left")
    print(" Done.")

    return df

def get_tabular_dataset(dataset_path="Data", save=True):
    data_train = create_tabular_dataset(dataset_path, subset="Train")
    data_test = create_tabular_dataset(dataset_path, subset="Test")

    feature_cols = data_train.drop(columns="Ground_truth").columns
    data_test = data_test[feature_cols]

    if save:
        data_train.to_csv(os.path.join(dataset_path, "Train/tabular_train.csv"), index=False)
        data_test.to_csv(os.path.join(dataset_path, "Test/tabular_test.csv"), index=False)

    return data_train, data_test

def add_nbr_precp(df, K=5):
    from sklearn.neighbors import NearestNeighbors

    station_coords = pd.read_csv("Data/Other/stations_coordinates.csv")
    coords = station_coords[["number_sta", "lat", "lon"]]
    nn = NearestNeighbors(n_neighbors=K+1, algorithm="ball_tree", metric="haversine")
    nn.fit(coords[["lat", "lon"]])
    dist, idx = nn.kneighbors(coords[["lat", "lon"]])
    for k in range(1, K+1):
        coords[f"nbr_{k}"] = coords.iloc[idx[:,k], 0].reset_index(drop=True)
    
    df["Id_split"] = df["Id"].str.split("_")
    df["number_sta"] = df["Id_split"].apply(lambda s: s[0]).astype(int)
    df["day"] = df["Id_split"].apply(lambda s: s[1]).astype(int)
    max_day = df["day"].max()
    nbr_cumul = pd.DataFrame()
    for d in tqdm(range(max_day+1)):
        day_filter = df["day"] == d
        station_data = df.loc[day_filter, ["Id", "number_sta", "precip_cumul"]]
        nbr_precip = station_data[["number_sta", "precip_cumul"]]
        for k in range(1, K+1):
            station_data = station_data.merge(coords[["number_sta", f"nbr_{k}"]], on="number_sta", how="left")
            nbr_precip.columns = ["number_nbr", f"nbr{k}_precip"]
            station_data = station_data.merge(nbr_precip, left_on=f"nbr_{k}", right_on="number_nbr", how="left").drop(columns=[f"nbr_{k}", "number_nbr"])
        
        station_data = station_data[["Id"] + [f"nbr{k}_precip" for k in range(1, K+1)]]
        nbr_cumul = pd.concat([nbr_cumul, station_data], axis=0)
    df = df.drop(columns=["Id_split", "number_sta", "day"]) 
    df = df.merge(nbr_cumul, on="Id", how="left")
    df["mean_nbr_precip"] = df[["precip_cumul"] + [f"nbr{k}_precip" for k in range(1, K+1)]].mean(axis=1)
    return df

def download_dataset(destination="data"):
    dest_path=os.path.join(destination, 'data.zip')
    gdd.download_file_from_google_drive(file_id="1ET2cY773F9Ai56YVlHEZm5VBEl719_D8",
                                    dest_path=dest_path,
                                    showsize=True,
                                    unzip=True)

def load_xgb_data(path="data"):
    try:
        data_train = pd.read_csv(os.path.join(path, "data_train.csv"))
        data_test = pd.read_csv(os.path.join(path, "data_test.csv"))
    except FileExistsError:
        print("Dataset not found.")
        response = input("Do you want to download the dataset? ([y]/n)")
        if response.lower() in ["y", "yes", ""]:
            download_dataset(dest_path=path)
        
        data_train = pd.read_csv(os.path.join(path, "data_train.csv"))
        data_test = pd.read_csv(os.path.join(path, "data_test.csv"))

    y_train = data_train["Ground_truth"]
    X_train = data_train.drop(columns="Ground_truth")
    X_test = data_test
    return X_train, y_train, X_test

if __name__ == "__main__":
    data_train, data_test = get_tabular_dataset()
    print(data_train.columns)
    print(data_test.shape)