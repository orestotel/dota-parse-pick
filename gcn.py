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

# Set device
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

json_file = 'parse-data/batches/bubblegum1.json' # Replace with your own path
data = parse_data(json_file)

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
sys.stdout.flush()

# Helper function to create graph data for a team
def team_to_graph_data(team):
    # Define a graph for the team with heroes as nodes
    num_nodes = len(team)
    x = torch.tensor([hero_to_idx[h] for h in team], dtype=torch.long).view(-1, 1)
    edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long)
    edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)
    return x, edge_index


# Convert
def convert_to_graph_data(X, y):
    graph_data_list = []
    for match, label in zip(X, y):
        radiant_team, dire_team = match
        radiant_x, radiant_edge_index = team_to_graph_data(radiant_team)
        dire_x, dire_edge_index = team_to_graph_data(dire_team)
        graph_data = Data(radiant_x=radiant_x, radiant_edge_index=radiant_edge_index,
                          dire_x=dire_x, dire_edge_index=dire_edge_index, y=label)
        graph_data_list.append(graph_data)
    return graph_data_list


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Define the Dota2GCN model
class Dota2GCN(nn.Module):
    def __init__(self, num_heroes, hidden_channels):
        super(Dota2GCN, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels * 2, 1)

    def forward(self, data):
        radiant_x, radiant_edge_index = data.radiant_x, data.radiant_edge_index
        dire_x, dire_edge_index = data.dire_x, data.dire_edge_index

        radiant_x = self.conv1(radiant_x, radiant_edge_index)
        radiant_x = torch.relu(radiant_x)
        radiant_x = self.conv2(radiant_x, radiant_edge_index)
        radiant_x = torch.relu(radiant_x)
        radiant_x = global_mean_pool(radiant_x, torch.zeros(radiant_x.size(0), dtype=torch.long))

        dire_x = self.conv1(dire_x, dire_edge_index)
        dire_x = torch.relu(dire_x)
        dire_x = self.conv2(dire_x, dire_edge_index)
        dire_x = torch.relu(dire_x)
        dire_x = global_mean_pool(dire_x, torch.zeros(dire_x.size(0), dtype=torch.long))

        x = torch.cat([radiant_x, dire_x], dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.view(-1)

model = Dota2GCN(len(heroes), 64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device, dtype=torch.float)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            label = data.y.to(device, dtype=torch.float)
            loss = criterion(output, label)
            total_loss += loss.item()
            pred = (output > 0.5).float()
            correct += pred.eq(label.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Set the number of training epochs
num_epochs = 50

# Perform training and evaluation
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
