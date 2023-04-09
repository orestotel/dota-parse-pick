import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk
import os
import pickle
from tqdm import tqdm
import sys
import threading
import time
import glob
import combine_files
import training_module
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def loading_animation(stop_event):
    animation_chars = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write("\rLoading... " + animation_chars[idx % len(animation_chars)])
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

# Start loading animation
stop_loading_animation = threading.Event()
loading_thread = threading.Thread(target=loading_animation, args=(stop_loading_animation,))
loading_thread.start()

# Load all datasets
dataset_filenames = ["parse-data/batches/bubblegum1.json"]

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")
sys.stdout.flush()

data = []

# Start loading animation
stop_loading_animation = threading.Event()
loading_thread = threading.Thread(target=loading_animation, args=(stop_loading_animation,))
loading_thread.start()

for filename in tqdm(dataset_filenames, desc="Loading datasets"):
    with open(filename, "r") as file:
        data.extend(json.load(file))

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")
sys.stdout.flush()

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

# Start loading animation
stop_loading_animation = threading.Event()
loading_thread = threading.Thread(target=loading_animation, args=(stop_loading_animation,))
loading_thread.start()

X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")
sys.stdout.flush()

# Helper function to create graph data for a team
def team_to_graph_data(team):
    # Define a graph for the team with heroes as nodes
    num_nodes = len(team)
    x = torch.tensor([hero_to_idx[h] for h in team], dtype=torch.long).view(-1, 1)
    edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long)
    edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim = 1)
    return x, edge_index

    # Convert
    # the
    # data
    # into
    #  PyTorch
    #   Geometric
    #    format

    def to_pyg_data(X, y):
        pyg_data = []

    for (radiant_team, dire_team), radiant_win in zip(X, y):
        radiant_x, radiant_edge_index = team_to_graph_data(radiant_team)
    dire_x, dire_edge_index = team_to_graph_data(dire_team)
    data = Data(radiant_x=radiant_x, radiant_edge_index=radiant_edge_index,
                dire_x=dire_x, dire_edge_index=dire_edge_index,
                y=torch.tensor([radiant_win], dtype=torch.float))
    pyg_data.append(data)
    return pyg_data

    #  Create
    #   PyTorch
    #  Geometric
    #   datasets
    train_data = to_pyg_data(X_train, y_train)
    test_data = to_pyg_data(X_test, y_test)

    #   Create
    #    data
    #    loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    #    Define
    #  the
    #    model

    class Dota2GCN(nn.Module):
        def init(self, in_channels, hidden_channels):

            super(Dota2GCN, self).init()

    self.conv1 = GCNConv(in_channels, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.fc = nn.Linear(2 * hidden_channels, 1)

    def forward(self, data):
        radiant_x, radiant_edge_index = data.radiant_x, data.radiant_edge_index
        dire_x, dire_edge_index = data.dire_x, data.dire_edge_index

        # Pass the radiant team's graph through the GCN layers
        radiant_x = self.conv1(radiant_x, radiant_edge_index)
        radiant_x = torch.relu(radiant_x)
        radiant_x = self.conv2(radiant_x, radiant_edge_index)
        radiant_x = torch.relu(radiant_x)
        radiant_x = global_mean_pool(radiant_x, torch.zeros(len(radiant_x), dtype=torch.long))

        # Pass the dire team's graph through the GCN layers
        dire_x = self.conv1(dire_x, dire_edge_index)
        dire_x = torch.relu(dire_x)
        dire_x = self.conv2(dire_x, dire_edge_index)
        dire_x = torch.relu(dire_x)
        dire_x = global_mean_pool(dire_x, torch.zeros(len(dire_x), dtype=torch.long))

        # Concatenate the team embeddings and pass through a final linear layer
        out = torch.cat([radiant_x, dire_x], dim=1)
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out.squeeze(1)
# Instantiate the model and optimizer
model = Dota2GCN(1, 64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Train the model
training_module.train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=100)

# Save the model
torch.save(model.state_dict(), "model.pt")

# Load the model
model = Dota2GCN(1, 64).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def predict(model, radiant_team, dire_team):
    radiant_x, radiant_edge_index = team_to_graph_data(radiant_team)
    dire_x, dire_edge_index = team_to_graph_data(dire_team)
    data = Data(radiant_x=radiant_x, radiant_edge_index=radiant_edge_index,
                dire_x=dire_x, dire_edge_index=dire_edge_index)
    data = data.to(device)
    with torch.no_grad():
        prediction = model(data)
    return prediction.item()

# Example usage
radiant_team = ["hero1", "hero2", "hero3", "hero4", "hero5"]
dire_team = ["hero6", "hero7", "hero8", "hero9", "hero10"]

win_probability = predict(model, radiant_team, dire_team)
print(f"Radiant win probability: {win_probability * 100:.2f}%")

# Tkinter GUI
def get_heroes_list():
    # Replace this with your actual list of heroes
    return sorted(heroes)

def on_submit():
    radiant_team = [rt_var1.get(), rt_var2.get(), rt_var3.get(), rt_var4.get(), rt_var5.get()]
    dire_team = [dt_var1.get(), dt_var2.get(), dt_var3.get(), dt_var4.get(), dt_var5.get()]

    win_probability = predict(model, radiant_team, dire_team)
    result_label.config(text=f"Radiant win probability: {win_probability * 100:.2f}%")

# Create the main window
root = tk.Tk()
root.title("Dota 2 Match Predictor")

# Radiant team
rt_label = tk.Label(root, text="Radiant Team")
rt_label.grid(row=0, column=0)

rt_var1, rt_var2, rt_var3, rt_var4, rt_var5 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()

rt_cb1 = ttk.Combobox(root, textvariable=rt_var1, values=get_heroes_list(), state="readonly", width=20)
rt_cb1.grid(row=1, column=0)
rt_cb2 = ttk.Combobox(root, textvariable=rt_var2, values=get_heroes_list(), state="readonly", width=20)
rt_cb2.grid(row=2, column=0)
rt_cb3 = ttk.Combobox(root, textvariable=rt_var3, values=get_heroes_list(), state="readonly", width=20)
rt_cb3.grid(row=3, column=0)
rt_cb4 = ttk.Combobox(root, textvariable=rt_var4, values=get_heroes_list(), state="readonly", width=20)
rt_cb4.grid(row=4, column=0)
rt_cb5 = ttk.Combobox(root, textvariable=rt_var5, values=get_heroes_list(), state="readonly", width=20)
rt_cb5.grid(row=5, column=0)

# Dire team
dt_label = tk.Label(root, text="Dire Team")
dt_label.grid(row=0, column=1)

dt_var1, dt_var2, dt_var3, dt_var4, dt_var5 = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()

dt_cb1 = ttk.Combobox(root, textvariable=dt_var1, values=get_heroes_list(), state="readonly", width=20)
dt_cb1.grid(row=1, column=1)
dt_cb2 = ttk.Combobox(root, textvariable=dt_var2, values=get_heroes_list(), state="readonly", width=20)
dt_cb2.grid(row=2, column=1)
dt_cb3 = ttk.Combobox(root, textvariable=dt_var3, values=get_heroes_list(), state="readonly", width=20)
dt_cb3.grid(row=3, column=1)
dt_cb4 = ttk.Combobox(root, textvariable=dt_var4, values=get_heroes_list(), state="readonly", width=20)
dt_cb4.grid(row=4, column=1)
dt_cb5 = ttk.Combobox(root, textvariable=dt_var5, values=get_heroes_list(), state="readonly", width=20)
dt_cb5.grid(row=5, column=1)

# Submit button
submit_button = tk.Button(root, text="Predict", command=on_submit)
submit_button.grid(row=6, column=0, columnspan=2)

# Result label
result_label = tk.Label(root, text="")
result_label.grid(row=7, column=0, columnspan=2)

# Run the main loop
root.mainloop()

