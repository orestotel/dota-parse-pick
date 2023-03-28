import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load data from the JSON file
with open("parse-data/bigflow.json", "r") as file:
    data = json.load(file)

# Encode hero names as integers
heroes = set()
for match in data:
    heroes.update(match["radiant_team"] + match["dire_team"])

encoder = LabelEncoder()
encoder.fit(list(heroes))

# Prepare input and output data
X = []
y = []

for match in data:
    radiant_team = encoder.transform(match["radiant_team"])
    dire_team = encoder.transform(match["dire_team"])
    match_input = np.concatenate((radiant_team, dire_team))
    X.append(match_input)
    y.append(1 if match["radiant_win"] else 0)

X = np.array(X)
y = to_categorical(y, num_classes=2)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data to files
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
