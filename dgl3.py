import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
import json
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Load data
with open("D:\\sofardata\\chewtree_process_0_copy.json", "r") as file:
    data = json.load(file)

# Fetch hero names from OpenDota API
response = requests.get("https://api.opendota.com/api/heroes", headers={"Authorization": "Bearer YOUR_API_KEY"})
heroes = response.json()
hero_names = [hero['localized_name'] for hero in heroes]

# Encode hero names to integers
encoder = LabelEncoder()
encoder.fit(hero_names)


def process_match_data(match_data, encoder):
    dataset = []
    labels = []
    num_heroes = len(encoder.classes_)

    for match in match_data:
        radiant_team = match["radiant_team"]
        dire_team = match["dire_team"]
        radiant_win = match["radiant_win"]

        match_heroes = [0] * num_heroes * 2

        for hero in radiant_team:
            hero_index = encoder.transform([hero])[0]
            match_heroes[hero_index] = 1

        for hero in dire_team:
            hero_index = encoder.transform([hero])[0]
            match_heroes[hero_index + num_heroes] = 1

        dataset.append(match_heroes)
        labels.append(1 if radiant_win else 0)

    return np.array(dataset), np.array(labels)


matches, labels = process_match_data(data, encoder)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(matches, labels, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(256, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
auc_score = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"AUC-ROC: {auc_score}")

# Save the model
model.save("D:\\d-disk-repo\\dota-parse-pick\\modelix.h5")


# Test the model
def test_model(model, matches, labels, encoder):
    num_heroes = len(encoder.classes_)

    for _ in range(10):
        index = random.randint(0, len(matches) - 1)
        match = matches[index]
        label = labels[index]
        radiant_win = label == 1
        radiant_team = []
        dire_team = []

        for i in range(num_heroes):
            if match[i] == 1:
                radiant_team.append(encoder.inverse_transform([i])[0])
            if match[i + num_heroes] == 1:
                dire_team.append(encoder.inverse_transform([i])[0])

        radiant_team = ", ".join(radiant_team)
        dire_team = ", ".join(dire_team)

        match = np.array([match])
        prediction = model.predict(match)[0][0]
        prediction = round(prediction, 2)
        print(f"Predicted: {prediction}, Actual: {label}, Difference: {abs(prediction - label)}")
        print(f"Radiant Team: {radiant_team}")
        print(f"Dire Team: {dire_team}")
        print(f"Probability of Radiant Win: {prediction}, Actual Probability of Radiant Win: {label}")
        print()

    test_model(model, X_test, y_test, encoder)


