import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Lade die Trainingsdaten
with open("gehen.json", "r") as f:
    training_data = json.load(f)

# Aufteilen der Trainingsdaten in Eingabe-X und Ziel-Y
X = np.array(training_data["data"])
Y = np.array(training_data["labels"])

# Normalisieren der Eingabe-X, falls erforderlich
# ...

# Aufteilen der Daten in Trainings- und Testdaten
split_index = int(len(X) * 0.8)  # 80% für das Training verwenden, 20% für die Validierung
X_train, X_val = X[:split_index], X[split_index:]
Y_train, Y_val = Y[:split_index], Y[split_index:]

# Erstelle das neuronale Netzwerk
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Ausgabe: 0 oder 1 (gehen oder stehen)

# Kompilieren des Modells
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Trainiere das Modell
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)

# Speichere das trainierte Modell
model.save("gehen_stehen_model.h5")