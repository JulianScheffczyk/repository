import streamlit as st
#from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pickle as pk
from joblib import load

# Lade das Scikit-Learn-Modell
trained_model = load('my_model.joblib')

sliding_window_length: int = 10

# Streamlit-App konfigurieren
st.title('Bewegungserkennung')

# Benutzerdefinierte Funktion zum Vorhersagen der Bewegungsart
def predict_motion(data):
    # Führe Vorhersage mit dem Modell durch
    prediction = model.predict(data.reshape(1, -1))
    
    # Mapping der Vorhersage auf die entsprechenden Klassen
    if prediction == 0:
        return 'Gegangen'
    elif prediction == 1:
        return 'Fahrrad gefahren'
    elif prediction == 2:
        return 'Auto gefahren'

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
    st.write("Willkommen auf der Startseite!")

#Page: Berechnung

def show_berechnung():
    st.title("Berechnung")
    # Streamlit-Anwendung
    uploaded_file = st.file_uploader("Lade eine Datei als .JSON File hoch und es wird der Kalorienverbrauch berechnet!")

    if uploaded_file is not None:
        # read data
        data = pd.read_json(uploaded_file)
        st.write("Daten wurden eingelesen!")

    if st.button("Berechnung starten!"):
        if  uploaded_file is not None:
            st.write("Daten werden mit dem Model ausgewertet...")
            load_in_model(data_analysis(data_prep(data)))
        else:
            st.write("Es müssen zuerst Daten hochgeladen werden!")

#Page: Hintergrund

def show_hintergrund():
    st.title("Hintergrund")
    st.write("Hier findest du weitere Informationen.")

    # Beispielhintergrundinformationen
    st.write("Dies ist eine Beispiel-Streamlit-App mit mehreren Seiten.")
    st.write("Sie wurde mit Python und Streamlit erstellt.")

#Page: About

def show_about():
    st.title("About")
    st.write("Informationen über diese App.")

    # Beispiel-App-Informationen
    st.write("Diese Streamlit-App wurde erstellt, um mehrere Seiten zu demonstrieren.")
    st.write("Sie wurde von [dein Name] entwickelt.")

#TODO umschreiben:

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
    return df

def data_analysis(df: pd.DataFrame):
    windows = []
    i = 0
    for window in df.rolling(window=sliding_window_length, method="table", axis="index"):
        # Exclude first and last too small sliding windows
        if len(window.index) < sliding_window_length:
            continue
        
        # Drop any time-related information (not necessary for ML model)
        window = window.drop(["time", "time_gyroscope", "time_accelerometer", "seconds_elapsed", "seconds_elapsed_gyroscope", "seconds_elapsed_accelerometer"], axis=1).reset_index(drop=True)
    load_in_model(windows)

def load_in_model(windows):
    gehen_count = 0
    fahrrad_count = 0
    auto_count = 0
    print('yes sir')
    trained_model.predict(windows)
    
    #TODO Predict Ausgabe -> counts für Kalorienberechnung

if __name__ == "__main__":
    main()
