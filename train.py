import pandas as pd
import json
import numpy as np
import sklearn 

import os
import pathlib
import json

from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import random
import joblib


class_mapping = {"Gehen": 0, "Fahrrad": 1, "Auto": 2, "Bus": 3}
train_proportion = 0.8
model_file = "my_model.joblib"


## set folder for data load
data_path: str = os.path.join("C:\\", "Users", "jsche", "Desktop", "ML4BFinal", "data")
sliding_window_length: int = 10
classes = ("Bus", "Fahrrad", "Auto", "Gehen")

def get_label_from_filename(fname, classes=classes):
    for cls_name in classes:
        if cls_name.lower() in fname.lower():
            return cls_name


def prepare_data_points_from_raw_df(df: pd.DataFrame):
    """Output df only contains measurements from Accelerometer, Gyroscope, Orientation interpolated (i.e. no NaN values)."""
    # From app documentation: Merge different sensor data measurements into common data points
    # Split into relevant sensors
    df_accelerometer = df[df["sensor"] == "Accelerometer"][["time", "seconds_elapsed", "z", "y", "x"]]
    df_gyroscope = df[df["sensor"] == "Gyroscope"][["time", "seconds_elapsed", "z", "y", "x"]]
    df_orientation = df[df["sensor"] == "Orientation"][["time", "seconds_elapsed", "qz", "qy", "qx", "qw", "roll", "pitch", "yaw"]]
    
    # Set index to timestamp
    for d in [df_gyroscope, df_accelerometer, df_orientation]:
        d.index = pd.to_datetime(d["time"], unit="ns")

    df = df_gyroscope
    df = df.join(df_accelerometer, lsuffix="_gyroscope", rsuffix="_accelerometer", how="outer").interpolate()
    df = df.join(df_orientation, how="outer").interpolate()

    nan_rows = df.isna().any(axis=1)
    if nan_rows.any():
        print("Dropping the following rows as they contain NaN entries:")
        print(df.loc[nan_rows].transpose())
        df = df.dropna()  # !! make sure to drop NaN entries !!
    
    return df


## load data
dfs: list = []
for filename, activity in (
    (f, get_label_from_filename(f)) for f in os.listdir(data_path)  # !! make more flexible with auto file discovery !!
):
#         ("gehen-2023-05-26-07-39-48.json", "Gehen"),
#         ("fahrrad-2023-06-18-19-36-15.json", "Fahrrad"),
#         ("autofahren-2023-05-26-07-57-08.json", "Auto"),
    print("Loading from", filename, f"(class {activity})")
    new_df = prepare_data_points_from_raw_df(pd.read_json(os.path.join(data_path, filename)))
    new_df["Activity"] = activity
                        
    dfs.append(new_df)



# Create data items that consist of several consequent timesteps
# ASSUMPTION: approximately equidistant points in time
def to_windows(df):
    windows = []

    for window in df.rolling(window=sliding_window_length, method="table", axis="index"):
        # Exclude first and last too small sliding windows
        if len(window.index) < sliding_window_length:
            continue
        
        # Drop any time-related information (not necessary for ML model)
        activity = window["Activity"].unique()[0]
        window = window.drop(["Activity", "time", "time_gyroscope", "time_accelerometer", "seconds_elapsed", "seconds_elapsed_gyroscope", "seconds_elapsed_accelerometer"], axis=1).reset_index(drop=True)
        concat_values = window.to_numpy().flatten()

        windows.append((concat_values, activity))
    
    return windows


windows = []
for df in dfs:
    windows += to_windows(df)



# Shuffle
random.shuffle(windows)

split_at = int(np.round(train_proportion*len(windows)))

windows_test = windows[split_at:]
windows_train = windows[:split_at]

X_train, y_str_train = zip(*windows_train)
y_train = [class_mapping[s] for s in y_str_train]
X_test, y_str_test = zip(*windows_test)
y_test = [class_mapping[s] for s in y_str_test]

if os.path.exists(model_file):
    print("Loading model from", model_file)
    trained_model = joblib.load(model_file)  # !! Use joblib, not pickle !!
else:
    print("Training model ...")
    #clf = OutputCodeClassifier(LinearSVC(random_state=0),
    #                           code_size=2, random_state=0)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
            hidden_layer_sizes=(5, 2), random_state=1)
    trained_model = clf.fit(X_train, y_train)
    joblib.dump(trained_model, model_file)  # !! Use joblib, not pickle !!

print("Accuracy:", sklearn.metrics.accuracy_score(y_test, trained_model.predict(X_test)))


