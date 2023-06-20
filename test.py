import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Lade das Scikit-Learn-Modell
model = joblib.load('my_model.joblib')

# Streamlit-App konfigurieren
st.title('Bewegungserkennung')
st.write('Verwende die Bewegungssensordaten des Handys, um zu erkennen, ob du gegangen bist, Fahrrad gefahren bist oder Auto gefahren bist.')

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

# Streamlit-Anwendung
if __name__ == '__main__':
    # Benutzerdefinierte Eingabe der Sensordaten
    accelerometer_x = st.number_input('Beschleunigungsmesser (x-Achse)', value=0.0)
    accelerometer_y = st.number_input('Beschleunigungsmesser (y-Achse)', value=0.0)
    accelerometer_z = st.number_input('Beschleunigungsmesser (z-Achse)', value=0.0)
    gyroscope_x = st.number_input('Gyroskop (x-Achse)', value=0.0)
    gyroscope_y = st.number_input('Gyroskop (y-Achse)', value=0.0)
    gyroscope_z = st.number_input('Gyroskop (z-Achse)', value=0.0)
    orientation = st.number_input('Orientierung', value=0.0)
    
    # Vorhersage nur, wenn alle Sensordaten verfügbar sind
    if st.button('Erkenne Bewegungsart'):
        data = np.array([accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z, orientation])
        motion = predict_motion(data)
        st.write(f'Du hast {motion}!')
