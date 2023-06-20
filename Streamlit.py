import streamlit as st
import joblib

# Laden des ML-Modells
model = joblib.load('my_model.joblib')

# Funktion zum Berechnen des Kalorienverbrauchs
def berechne_kalorienverbrauch(gewicht, koerpergroesse, aktivitaet):
    input_data = [[gewicht, koerpergroesse, aktivitaet]]
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit-Anwendung
def main():
    st.title("Kalorienverbrauch Rechner")
    st.write("Gib dein Gewicht, deine Körpergröße und die Art der Aktivität ein:")

    gewicht = st.number_input("Gewicht (in kg)")
    koerpergroesse = st.number_input("Körpergröße (in cm)")
    aktivitaet = st.selectbox("Aktivität", ["Gehen", "Fahrradfahren", "Autofahren"])

    if st.button("Berechnen"):
        kalorienverbrauch = berechne_kalorienverbrauch(gewicht, koerpergroesse, aktivitaet)
        st.write(f"Der geschätzte Kalorienverbrauch beträgt: {kalorienverbrauch} kcal")

    st.write("Bitte beachte, dass dies eine grobe Schätzung ist.")

if __name__ == '__main__':
    main()
