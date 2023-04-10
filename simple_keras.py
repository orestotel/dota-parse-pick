import json
import numpy as np
import os
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Ensure GPU is available
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError("GPU not available for training")

# Load data
print("Loading data...")
with open("parse-data/bubblegum1.json", "r") as f:
    data = json.load(f)
print(f"Number of matches in dataset: {len(data)}")

# Load hero names from opendota
print("Fetching heroes from OpenDota...")
heroes_data = requests.get("https://api.opendota.com/api/heroes").json()
hero_names = [hero["localized_name"] for hero in heroes_data]

# Preprocess data
def encode_teams(team):
    encoded = [0] * len(hero_names)
    for hero in team:
        encoded[hero_names.index(hero)] = 1
    return encoded

X = []
y = []

for match in data:
    radiant_team_encoded = encode_teams(match["radiant_team"])
    dire_team_encoded = encode_teams(match["dire_team"])
    X.append(radiant_team_encoded + dire_team_encoded)
    y.append(match["radiant_win"])

X = np.array(X)
y = np.array(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create or load model
model_name = "trainmodel2"
if os.path.exists(model_name):
    model = load_model(model_name)
else:
    model = Sequential([
        Dense(128, activation="relu", input_dim=X_train.shape[1]),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),  # Added layers
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
batch_size = 4096  # Increased batch size
epochs = int(input("Enter the number of epochs to train: "))
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Save model
model.save(model_name)

# Console interface for predicting matches
print("Enter the hero names starting from the Radiant team and ending with the Dire team.")
print("If a hero name is spelled wrong or has already been chosen, input again until the correct name is given.")

while True:
    try:
        hero_input = []
        for i in range(10):
            hero = input(f"Enter hero {i + 1}: ").strip()
            while hero not in hero_names or hero in hero_input:
                print("Invalid hero name or hero already chosen.")
                hero = input(f"Enter hero {i + 1}: ").strip()
            hero_input.append(hero)

        encoded_input = np.array([encode_teams(hero_input[:5]) + encode_teams(hero_input[5:])])
        prediction = model.predict(encoded_input)[0][0]
        print(f"Radiant win probability: {prediction * 100:.2f}%")
    except KeyboardInterrupt:
        break
