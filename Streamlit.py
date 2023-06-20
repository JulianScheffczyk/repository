import streamlit as st
import joblib
import os

data_path: str = os.path.join("C:\\", "Users", "jsche", "Downloads")

# Laden des ML-Modells
model = joblib.load("c:\Users\jsche\Desktop\ML4B\my_model.joblib")

# Funktion zur Berechnung des Kalorienverbrauchs
def berechne_kalorienverbrauch(gewicht, groesse, aktivitaet):
    if aktivitaet == "Gehen":
        kalorienverbrauch = 0.5 * gewicht + 0.1 * groesse
    elif aktivitaet == "Fahrrad fahren":
        kalorienverbrauch = 0.3 * gewicht + 0.2 * groesse
    elif aktivitaet == "Auto fahren":
        kalorienverbrauch = 0.1 * gewicht + 0.05 * groesse
    else:
        kalorienverbrauch = 0.0
    return kalorienverbrauch

# Streamlit-Anwendung
def main():
    st.title("Kalorienverbrauch-Rechner")
    
    gewicht = st.number_input("Gewicht (kg)")
    groesse = st.number_input("Körpergröße (cm)")
    aktivitaet = st.selectbox("Aktivität", ["Gehen", "Fahrrad fahren", "Auto fahren"])
    
    kalorienverbrauch = berechne_kalorienverbrauch(gewicht, groesse, aktivitaet)
    st.write("Kalorienverbrauch:", kalorienverbrauch, "kcal")
    
    st.write("Aktivitätserkennung:")
    features = [gewicht, groesse]
    prediction = model.predict([features])[0]
    if prediction == 0:
        st.write("Die Aktivität wird als Gehen erkannt.")
    elif prediction == 1:
        st.write("Die Aktivität wird als Fahrrad fahren erkannt.")
    elif prediction == 2:
        st.write("Die Aktivität wird als Auto fahren erkannt.")
    else:
        st.write("Aktivität nicht erkannt.")

if __name__ == "__main__":
    main()

