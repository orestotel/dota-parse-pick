import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk

# Load all datasets
dataset_filenames = ["parse-data/bigflow.json", "parse-data/newmatches1.json",
                     "parse-data/newmatches2.json", "parse-data/newmatches3.json",
                     "parse-data/newmatches4.json",
                     "parse-data/dataset1.json"]

data = []

for filename in dataset_filenames:
    with open(filename, "r") as file:
        data.extend(json.load(file))

# Preprocess and encode data
heroes = set()
for match in data:
    heroes |= set(match["radiant_team"]) | set(match["dire_team"])
heroes = list(heroes)
hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)}

def preprocess_data(data):
    X = []
    y = []
    for match in data:
        radiant_team = [hero_to_idx[hero] for hero in match["radiant_team"]]
        dire_team = [hero_to_idx[hero] for hero in match["dire_team"]]
        X.append((radiant_team, dire_team))
        y.append(match["radiant_win"])
    return X, y

X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define FNN model
class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Convert data to tensors
def data_to_tensor(data):
    tensor_data = []
    for (radiant_team, dire_team) in data:
        vector = np.zeros(len(heroes) * 2)
        for idx, hero_idx in enumerate(radiant_team):
            vector[hero_idx] = 1
        for idx, hero_idx in enumerate(dire_team):
            vector[hero_idx + len(heroes)] = 1
        tensor_data.append(vector)
    return torch.tensor(tensor_data, dtype=torch.float32)

X_train_tensor = data_to_tensor(X_train)
X_test_tensor = data_to_tensor(X_test)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Training
def train(X_train, y_train, num_epochs=100, hidden_size=128):
    input_size = len(heroes) * 2
    num_classes = 2
    model = FNNModel(input_size, hidden_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

model = train(X_train_tensor, y_train_tensor)

# Evaluation
def evaluate(X_test, y_test, model):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted)
        print(f"Accuracy: {accuracy * 100:.2f}%")

evaluate(X_test_tensor, y_test_tensor, model)

# Create a GUI for prediction
def predict(model, radiant_team, dire_team):
    radiant_team_idx = [hero_to_idx[hero] for hero in radiant_team]
    dire_team_idx = [hero_to_idx[hero] for hero in dire_team]
    input_data = [(radiant_team_idx, dire_team_idx)]
    input_tensor = data_to_tensor(input_data)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()[0]



def predict_callback():
    radiant_team = [rt_hero1.get(), rt_hero2.get(), rt_hero3.get(), rt_hero4.get(), rt_hero5.get()]
    dire_team = [dt_hero1.get(), dt_hero2.get(), dt_hero3.get(), dt_hero4.get(), dt_hero5.get()]
    probabilities = predict(model, radiant_team, dire_team)
    radiant_win_probability = probabilities[1] * 100
    result_text.set(f"(higaben)Radiant Win Chance: {radiant_win_probability:.2f}%")


root = tk.Tk()
root.title("Dota 2 Match Predictor")

# Create the dropdown menus
heroes_var = tk.StringVar(value=heroes)
rt_hero1 = ttk.Combobox(root, values=heroes, state="readonly")
rt_hero2 = ttk.Combobox(root, values=heroes, state="readonly")
rt_hero3 = ttk.Combobox(root, values=heroes, state="readonly")
rt_hero4 = ttk.Combobox(root, values=heroes, state="readonly")
rt_hero5 = ttk.Combobox(root, values=heroes, state="readonly")
dt_hero1 = ttk.Combobox(root, values=heroes, state="readonly")
dt_hero2 = ttk.Combobox(root, values=heroes, state="readonly")
dt_hero3 = ttk.Combobox(root, values=heroes, state="readonly")
dt_hero4 = ttk.Combobox(root, values=heroes, state="readonly")
dt_hero5 = ttk.Combobox(root, values=heroes, state="readonly")

# Place the dropdown menus
for i, h in enumerate([rt_hero1, rt_hero2, rt_hero3, rt_hero4, rt_hero5]):
    h.grid(row=0, column=i, padx=5, pady=5)
for i, h in enumerate([dt_hero1, dt_hero2, dt_hero3, dt_hero4, dt_hero5]):
    h.grid(row=1, column=i, padx=5, pady=5)

# Add a predict button
predict_button = tk.Button(root, text="Predict", command=predict_callback)
predict_button.grid(row=2, column=0, columnspan=5, padx=5, pady=5)

# Add a label to display the result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.grid(row=3, column=0, columnspan=5, padx=5, pady=5)

root.mainloop()
