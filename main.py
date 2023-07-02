import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier

# Lade das Scikit-Learn-Modell
trained_model = joblib.load('my_model.joblib')  # !! Use joblib, not pickle !!

sliding_window_length: int = 10
CLASS_MAPPING = {0: "Gehen", 1: "Fahrrad", 2: "Auto", 3: "Bus"}
COLOR_MAPPING = ['green', 'blue', 'red', 'brown']

# Streamlit-App konfigurieren
st.title('Bewegungserkennung')

# Benutzerdefinierte Funktion zum Vorhersagen der Bewegungsart
def predict_motion(data, model=trained_model):  # !! wird nicht mehr verwendet -> Funktion kann weg !!
    # Führe Vorhersage mit dem Modell durch
    prediction = model.predict(data.reshape(1, -1))
    
    # Mapping der Vorhersage auf die entsprechenden Klassen
    if prediction == 0:
        return 'Gegangen'
    elif prediction == 1:
        return 'Fahrrad gefahren'
    elif prediction == 2:
        return 'Auto gefahren'
    elif prediction == 3:
        return 'Bus gefahren'

def main():
    # Seite auswählen
    selected = option_menu(
        menu_title = None,
        options=["Home", "Berechnung", "Hintergrund", "About"],
        icons=["house", "calculator", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal")

    # Seiteninhalt anzeigen
    if selected == "Home":
        show_home()
    elif selected == "Berechnung":
        show_berechnung()
    elif selected == "Hintergrund":
        show_hintergrund()
    elif selected == "About":
        show_about()

#Page: Home

def show_home():
    st.title("Home")
    st.write("Diese Streamlit Website kann anhand einer JSON File, die mit dem Sensor Logger aufgezeichnet wurde ausgeben, wie viel Kalorien in dem Zeitraum verbrannt wurden.")
    st.write("Diese Kalorienberechnung basiert auf generellen Durchschnittswerten und kann durch Geschlecht oder Körpermasse variieren.")
    st.write("Hier ist eine Übersicht der weiteren Reiter:")
    st.write("- Berechnung: Kalorienberechnung")
    st.write("- Hintergrund: Backstory zu dem Erstellen der Website")
    st.write("- About: Informationen über den Ersteller bzw. die Herangehensweise an die Thematik")

#Page: Berechnung

def show_berechnung():
    st.title("Berechnung")
    # Streamlit-Anwendung
    uploaded_files = st.file_uploader("Lade eine Datei als .JSON File hoch und es wird der Kalorienverbrauch berechnet!", accept_multiple_files=True) # !! allow more than one file !!

    data_list = []
    if len(uploaded_files) > 0:
        # read data
        for uploaded_file in uploaded_files:
            data = data_prep(pd.read_json(uploaded_file))
            # !! Set all to same day for test purposes !!
            data.index = data.index.to_series().apply(lambda t: pd.to_datetime(t).replace(year=2023, month=6, day=23))
            # Some debugging info
            print(f"Loaded {uploaded_file.name}: {len(data.index)} samples, from {data.index.min()} to {data.index.max()}")
            data_list.append(data) # !! Don't keep large JSON output, only compressed prepared one !!
        st.write("Daten wurden eingelesen!")

    if st.button("Berechnung starten!"):
        if  len(uploaded_files) > 0:
            st.write("Daten werden vorverarbeitet ...")
            # Collect per-file predictions
            all_windows, all_times, all_raw_predictions = [], [], []  # !! again: treatment of more than one file: make sure windows don't overlap !!
            for data in data_list:
                # Unpack: list of windows, list of list of timestamps per (matching window at same index)
                windows, times = zip(*to_windows_with_time_array(data))
                all_windows.append(windows)
                all_times.append(times)
                all_raw_predictions.append(get_raw_predictions(windows))
            st.write("Daten werden mit dem Modell ausgewertet und analysiert ...")
            analysis(all_windows, all_times, all_raw_predictions, plot=True)
        else:
            st.write("Es müssen zuerst Daten hochgeladen werden!")

#Page: Hintergrund

def show_hintergrund():
    st.title("Hintergrund")
    st.write("Im Rahmen des Projekts Machine Learning for Business habe ich (Julian Scheffczyk - 6. Semester Wirtschaftsinformatik Bachelor) ein Machine Learning Model trainiert und in dieser Streamlit Website eingebaut.")
    st.write("Namen des Kurses: Maschine Learning For Business")
    st.write("Betreuer: Annika Schreiner, Markus Schmitz, Dr. Martin Enders")


#Page: About

def show_about():
    st.title("About")
    st.write("Author: Julian Scheffczyk")
    st.image('JulianScheffczyk.jpg', caption='Julian Scheffczyk', width=200 ,output_format="auto")
    st.write("Machine Learning Model: Neuronal Network - Classification (SciKit Learn)")
    st.write("https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron")
    st.write("Data Preperation: Windowing in 10 Sekunden Parts")
    st.write("Data Collection: Sensor Logger App")
    st.write("Sensors: Accelerometer, Gyroscope, Orientation")
    st.write("Sample Rate: 1 Hz")

def data_prep(df: pd.DataFrame):
    df_accelerometer = df[df["sensor"] == "Accelerometer"][["time", "seconds_elapsed", "z", "y", "x"]]
    df_gyroscope = df[df["sensor"] == "Gyroscope"][["time", "seconds_elapsed", "z", "y", "x"]]
    df_orientation = df[df["sensor"] == "Orientation"][["time", "seconds_elapsed", "qz", "qy", "qx", "qw", "roll", "pitch", "yaw"]]

    # Set index to timestamp
    for d in [df_gyroscope, df_accelerometer, df_orientation]:
        d.index = pd.to_datetime(d["time"], unit="ns")

    df = df_gyroscope
    df = df.join(df_accelerometer, lsuffix="_gyroscope", rsuffix="_accelerometer", how="outer").interpolate()
    df = df.join(df_orientation, how="outer").interpolate()

    df = df.dropna()
    return df


def to_windows_with_time_array(df: pd.DataFrame):
    """Returns list of pairs (window, times)"""
    windows = []
    
    for window in df.rolling(window=sliding_window_length, method="table", axis="index"):
        # Exclude first and last too small sliding windows
        if len(window.index) < sliding_window_length:
            continue
        
        # Drop any time-related information (not necessary for ML model)
        times = window.index.values
        window = window.drop(["time", "time_gyroscope", "time_accelerometer", "seconds_elapsed", "seconds_elapsed_gyroscope", "seconds_elapsed_accelerometer"], axis=1).reset_index(drop=True)
        concat_values = window.to_numpy().flatten()
        
        windows.append((concat_values, times))  # !! don't forget to append !!

    return windows

def calc_total_durations(acts_by_time_df, class_mapping=CLASS_MAPPING):
    """Calculates the total duration per activity in minutes from predictions.
    Returns dataframe with index the activity common names from class_mapping, and columns:
    'tot_time' (float): total duration in minutes"""

    results = {k: {"tot_time": pd.Timedelta(0), "kalorien": 0} for k in class_mapping.keys()}
    
    # Split into sections of same activity
    acts_by_time_df['group'] = acts_by_time_df['activity'].ne(acts_by_time_df['activity'].shift()).cumsum()
    lines = acts_by_time_df.groupby('group')
    for _, line in lines:
        curr_pred_class = line["activity"].unique()[0]
        curr_duration = line["time"].max() - line["time"].min()
        results[curr_pred_class]["tot_time"] += curr_duration

        # Hier kommt die Berechnung des Kalorienverbrauchs basierend auf der Aktivität hin
        # Annahme: Kalorienverbrauch pro Minute für jede Aktivität
        # Gehen: 5 kcal/min, Fahrrad: 10 kcal/min, Auto: 2 kcal/min, Bus: 3 kcal/min
        kalorienverbrauch = 0
        if curr_pred_class == 0:  # Gehen
            kalorienverbrauch = 5 * curr_duration.total_seconds() / 60
        elif curr_pred_class == 1:  # Fahrrad
            kalorienverbrauch = 10 * curr_duration.total_seconds() / 60
        elif curr_pred_class == 2:  # Auto
            kalorienverbrauch = 2 * curr_duration.total_seconds() / 60
        elif curr_pred_class == 3:  # Bus
            kalorienverbrauch = 3 * curr_duration.total_seconds() / 60

        results[curr_pred_class]["kalorien"] += kalorienverbrauch

    results_df = pd.DataFrame(results).transpose().astype({"tot_time": "timedelta64[s]", "kalorien": float})

    # Total time in (float) minutes
    results_df.loc[:, "tot_time"] = (results_df["tot_time"] / np.timedelta64(1, "m"))

    return results_df



def plot_total_durations(results_df, class_mapping=CLASS_MAPPING):
    # For control: Print results of summation
    #print(results_df)
    st.write(results_df.rename({"tot_time": "Minuten gesamt"}, axis=1).rename(class_mapping, axis=0))
    print(results_df.dtypes)
    print(results_df["tot_time"].isna().any())


    # Plot total duration (normalized to minutes) by category
    fig, ax = plt.subplots()
    results_df["tot_time"].plot.bar(ax=ax)
    ax.set_ylabel("Minuten gesamt")
    st.pyplot(fig)

    return results_df


def activities_by_timestamps(windows, times, raw_predictions,
                             time_window=30 # time window for averaging in seconds; skip smoothing/filling of gaps if None
                             ):
    """Create and plot dataframe mapping timestamps to activities.
    Returns dataframe with columns:
    'time' (datetime64[ns]): timestamp
    'activity' (str): the predicted activity key
    """

    # Get dataframe assigning all timestamps an activity class
    acts_by_time = []
    for i, timestamps in enumerate(times):
        curr_class = raw_predictions[i]
        acts_by_time += [{"time": t, "activity": curr_class} for t in timestamps]
    acts_by_time_df = (
        pd.DataFrame(acts_by_time)
        .astype({"time": "datetime64[ns]"})
        .groupby(["time"]).aggregate({"activity": lambda a: a.mode().values[0]})  # make unique timestamp indices
        .reset_index() # make "time" a column again, not an index (used in plotting)
    )

    # Smoothen by averaging to one entry every time_window seconds
    if time_window is not None:
        acts_by_time_df = (
            acts_by_time_df
            # set most occurring value (if any observations available in time slot)
            .resample(f"{time_window}S", on="time").aggregate({"activity": lambda slot: None if not len((f := slot.mode()))>0 else f.values[0]})
            .dropna(axis=0, how="all")
            .reset_index()
        )
        # acts_by_time_df = (
        #     acts_by_time_df
        #     .rolling(window=pd.Timedelta(seconds=time_window), min_periods=20, center=True, on="time")
        #     .aggregate({"activity": lambda a: None if not len((f := x.mode()))>0 else f.values[0]})
        #     .reset_index()
        # )

    return acts_by_time_df

    
def plot_activities_by_timestamp(acts_by_time_df, class_mapping=CLASS_MAPPING, color_mapping=COLOR_MAPPING,):
    # Plot activity by time
    fig, ax = plt.subplots()
    ax.set_yticks(ticks=list(class_mapping.keys()),
                    labels=list(class_mapping.values()))  # label y-axis ticks with activity categories
    ax.set_ylim([-0.5, max(class_mapping.keys()) + 0.5]) # Always show all activity categories
    
    # Draw gray line connecting activity classes
    ax.step(x=acts_by_time_df["time"], y=acts_by_time_df["activity"], label='_all', color="gray", linestyle="dotted")
    
    # Colored line per activity class
    # Split into sections of same activity
    acts_by_time_df['group'] = acts_by_time_df['activity'].ne(acts_by_time_df['activity'].shift()).cumsum()
    lines = acts_by_time_df.groupby('group')
    for _, line in lines:
        curr_activity = line["activity"].unique()[0]
        line.plot(
            x="time", y="activity", ax=ax, color=color_mapping[int(curr_activity)],
            label=f"{curr_activity}", # start with _ to no show label
            marker="o",  # draw dots as marker points on the line
            #markevery=0.02  # only draw some dots (equally distributed, every 2% of plot width a point)
        )
    ax.set_xlabel("Uhrzeit")
    ax.get_legend().remove() # no legend
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Put legend center right of plot
    st.pyplot(fig)


def get_raw_predictions(windows, model=trained_model):
    # Actual ML model prediction
    return model.predict(np.array(windows))


def analysis(all_windows, all_times, all_raw_predictions, plot=True):

    # Concat results from several files
    windows = [i for t in all_windows for i in t]
    times = [i for t in all_times for i in t]
    raw_predictions = [i for t in all_raw_predictions for i in t]

    # Get smoothed mapping of (regular interval) timestamps to predicted activity
    acts_by_time_df = activities_by_timestamps(windows, times, raw_predictions)
    
    # Calc and plot total duration in minutes per activity class
    tot_durations_df = calc_total_durations(acts_by_time_df)
    
    plot_total_durations(tot_durations_df)
    plot_activities_by_timestamp(acts_by_time_df)
    
    return tot_durations_df

if __name__ == "__main__":
    main() 
