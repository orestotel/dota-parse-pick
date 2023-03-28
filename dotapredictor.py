import json
import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from opendota_fetcher import *
OPENDOTA_API_KEY = "2a5b8577-d7ee-4ef2-85ca-f15e5c8bdf75"

import opendota_fetcher

# Fetch match data from the OpenDota API
new_matches = fetch_and_save_matches("2a5b8577-d7ee-4ef2-85ca-f15e5c8bdf75", 300000, 3000,
                                     "parse-data/newmatches5.json")



def load_data_from_files(filenames):
    data = []
    for filename in filenames:
        with open(filename, "r") as file:
            file_data = json.load(file)
            data.extend(file_data)
    return data

# Load data from multiple JSON files
filenames = ["bigflow.json", "newmatches1.json", "newmatches2.json", "newmatches3.json", "newmatches4.json","newmatches5.json"]
data = load_data_from_files(filenames)

def get_model_accuracy():
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100

# Preprocess and encode data
heroes = set()
for match in data:
    heroes |= set(match["radiant_team"]) | set(match["dire_team"])
heroes = list(heroes)
hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)}

X = []
y = []
for match in data:
    radiant_team = [hero_to_idx[hero] for hero in match["radiant_team"]]
    dire_team = [hero_to_idx[hero] for hero in match["dire_team"]]
    match_vector = [0] * len(heroes)
    for idx in radiant_team:
        match_vector[idx] = 1
    for idx in dire_team:
        match_vector[idx] = -1
    X.append(match_vector)
    y.append(match["radiant_win"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)


def predict_win_probability(radiant_team, dire_team):
    match_vector = [0] * len(heroes)
    for hero in radiant_team:
        match_vector[hero_to_idx[hero]] = 1
    for hero in dire_team:
        match_vector[hero_to_idx[hero]] = -1
    win_probability = model.predict_proba([match_vector])[0][1]
    return win_probability * 100


# GUI
def on_submit():
    radiant_team = [radiant_heroes[idx].get() for idx in range(5)]
    dire_team = [dire_heroes[idx].get() for idx in range(5)]
    win_probability = predict_win_probability(radiant_team, dire_team)
    model_accuracy = get_model_accuracy()
    result_text.set(
        f"Radiant win probability: {win_probability:.2f}% | Model accuracy: {model_accuracy:.2f}%"
    )


root = tk.Tk()
root.title("Dota 2 Match Predictor")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

radiant_heroes = [tk.StringVar() for _ in range(5)]
dire_heroes = [tk.StringVar() for _ in range(5)]

for idx, var in enumerate(radiant_heroes):
    combobox = ttk.Combobox(mainframe, textvariable=var, values=heroes)
    combobox.grid(column=0, row=idx, sticky=(tk.W, tk.E))

for idx, var in enumerate(dire_heroes):
    combobox = ttk.Combobox(mainframe, textvariable=var, values=heroes)
    combobox.grid(column=1, row=idx, sticky=(tk.W, tk.E))

submit_button = ttk.Button(mainframe, text="Predict", command=on_submit)
submit_button.grid(column=0, row=5, columnspan=2, pady=10)

result_text = tk.StringVar()
result_label = ttk.Label(mainframe, textvariable=result_text)
result_label.grid(column=0, row=6, columnspan=2, pady=10)

root.mainloop()

