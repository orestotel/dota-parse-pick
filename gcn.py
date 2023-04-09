import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
from gcn_parse import parse_data
from datetime import datetime
from tqdm import tqdm
import sys
import threading
import time
import glob
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx  # For visualization
import matplotlib.pyplot as plt  # For visualization
from reportlab.lib.pagesizes import letter   # For PDF generation
from reportlab.pdfgen import canvas  # For PDF generation
from reportlab.lib import colors # For PDF generation
from reportlab.lib.units import inch    # For PDF generation
import ujson as json


sys.setrecursionlimit(1000000)

#conda install pytorch torchvision torchaudio -c pytorch -c conda-forge -c nvidia

# Set device  (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Start loading animation
def loading_animation(stop_event):
    animation_chars = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write("\rLoading... " + animation_chars[idx % len(animation_chars)])
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

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
json_file = 'parse-data/batches/bubblegum1.json' # Replace with your own path
pickle_file = 'preprocessed_data.pkl'

if os.path.exists(pickle_file):
    print("Loading preprocessed data from pickle file...")
    # Load preprocessed data from pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        print(f"Loaded {len(data)} matches")
else:
    print("Preprocessing data...")
    # Load and preprocess the data
    with open(json_file, 'rb') as f:
        print("Loading data from JSON file...")
        matches = json.load(f)
        print(f"Loaded {len(matches)} matches")
        for match in matches:
            data.append(match)
    # Save the preprocessed data to a pickle file
    with open(pickle_file, 'wb') as f:
        print("Saving preprocessed data to pickle file...")
        pickle.dump(data, f)
        print(f"Saved {len(data)} matches in a pickle file")

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")
sys.stdout.flush()

# Preprocess and encode data
# Collect all unique hero names
heroes = set()
for match in data:
    heroes |= set(match["radiant_team"]) | set(match["dire_team"])
heroes = list(heroes)
# Create a dictionary to map hero names to indices
hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)}

def preprocess_data(data):
    X = []
    y = []
    for match in data:
        radiant_team = match["radiant_team"]
        dire_team = match["dire_team"]
        X.append((radiant_team, dire_team))
        y.append(match["radiant_win"])
    return X, y

# Start loading animation
stop_loading_animation = threading.Event()
loading_thread = threading.Thread(target=loading_animation, args=(stop_loading_animation,))
loading_thread.start()

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")

def team_to_graph_data(team):
    # Define a graph for the team with heroes as nodes
    num_nodes = len(team)
    x = torch.tensor([hero_to_idx[hero] for hero in team], dtype=torch.float) # Change dtype to torch.float
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i)], dtype=torch.long).t().contiguous()
    edge_index = edge_index.view(2, -1)
    data = Data(x=x, edge_index=edge_index)

    return data

def build_graph_data(X):
    graph_data = []
    for radiant_team, dire_team in tqdm(X, desc="Building graph data"):
        # Get the graph data for each team
        radiant_data = team_to_graph_data(radiant_team)
        dire_data = team_to_graph_data(dire_team)

        # Concatenate the two graphs
        x = torch.cat([radiant_data.x, dire_data.x])
        y = torch.tensor([1, 0], dtype=torch.long)  # 1 for Radiant, 0 for Dire
        y = y.view(-1, 1).repeat(1, len(radiant_team) + len(dire_team)).view(-1)
        edge_index = torch.cat([radiant_data.edge_index, dire_data.edge_index + len(radiant_team)], dim=1)
        data = Data(x=x, y=y, edge_index=edge_index)

        graph_data.append(data)
    return graph_data


# Start loading animation
stop_loading_animation = threading.Event()
loading_thread = threading.Thread(target=loading_animation, args=(stop_loading_animation,))
loading_thread.start()

# Preprocess data and encode it into PyTorch Geometric graph data
X, y = preprocess_data(data)
graph_data = build_graph_data(X)

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")

# Train-validation-test split
train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)
test_data = val_data[int(len(val_data)/2):]
val_data = val_data[:int(len(val_data)/2)]

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Define model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, node_dim=0)  # Change node_dim to 0
        self.conv2 = GCNConv(hidden_dim, output_dim, node_dim=0)  # Change node_dim to 0

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, torch.tensor([0, 1], dtype=torch.long).to(device))
        return x


model = GCN(len(heroes), 64, 2).to(device)
print(model)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

# Train model
best_val_acc = 0
for epoch in range(100):
    # Training loop
    model.train()
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(out, batch.y.to(device))
        loss.backward()
        optimizer.step()

    # Validation loop
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_total = 0
        val_correct = 0
        for batch in tqdm(val_loader, desc=f"Validation epoch {epoch}"):
            batch = batch.to(device)
            out = model(batch.x.to(device), batch.edge_index.to(device))
            loss = criterion(out, batch.y.to(device))
            val_loss += loss.item() * batch.num_graphs
            _, predicted = torch.max(out.data, 1)
            val_total += batch.num_graphs
            val_correct += (predicted == batch.y.to(device)).sum().item()

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch}: Validation loss = {val_loss:.4f}, Validation accuracy = {val_acc:.4f}")

        if val_acc > best_val_acc:
            print(f"New best validation accuracy = {val_acc:.4f}, saving model...")
            torch.save(model.state_dict(), "gcn_model.pt")
            best_val_acc = val_acc

# Test loop
model.eval()
with torch.no_grad():
    test_loss = 0
    test_total = 0
    test_correct = 0
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(device)
        out = model(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(out, batch.y.to(device))
        test_loss += loss.item() * batch.num_graphs
        _, predicted = torch.max(out.data, 1)
        test_total += batch.num_graphs
        test_correct += (predicted == batch.y.to(device)).sum().item()

    test_acc = test_correct / test_total
    test_loss /= test_total

    print(f"Test loss = {test_loss:.4f}, Test accuracy = {test_acc:.4f}")