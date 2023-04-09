import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
from gcn_parse import parse_data,fetch_heroes
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
from reportlab.lib.pagesizes import letter  # For PDF generation
from reportlab.pdfgen import canvas  # For PDF generation
from reportlab.lib import colors  # For PDF generation
from reportlab.lib.units import inch  # For PDF generation
import ujson as json

sys.setrecursionlimit(1000000)

# conda install pytorch torchvision torchaudio -c pytorch -c conda-forge -c nvidia

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
json_file = 'parse-data/batches/bubblegum1.json'  # Replace with your own path
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
    data = parse_data(json_file)
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

# Train-validation-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data , batch_size=128, shuffle=False)



# Define GCN model


class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Instantiate model and optimizer
model = GCN(num_features=22, hidden_size=64, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
num_epochs = 100

train_losses = []  # Losses for each epoch
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_acc = 0
    for batch in train_loader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.to(device)
        y = batch.y.to(device)
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy_score(y.cpu(), torch.argmax(out, dim=1).cpu())
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc) # Accuracy for each epoch

    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch.x.float().to(device)
            edge_index = batch.edge_index.to(device)
            y = batch.y.to(device)
            out = model(x, edge_index)
            loss = nn.CrossEntropyLoss()(out, y)
            val_loss += loss.item()
            val_acc += accuracy_score(y.cpu(), torch.argmax(out, dim=1).cpu())
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate model on test set
model.eval()
test_acc = 0
with torch.no_grad():
    for batch in test_loader:
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.to(device)
        y = batch.y.to(device)
        out = model(x, edge_index)
        test_acc += accuracy_score(y.cpu(), torch.argmax(out, dim=1).cpu())
test_acc /= len(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model_path = "gcn_model.pt"
torch.save(model.state_dict(), model_path)

# Visualize a match graph
match = data[0]
G = to_networkx(match).to_undirected()
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=100, cmap="coolwarm",
                       node_color=match.y.numpy())
nx.draw_networkx_edges(G, pos, width=1)
plt.axis("off")
plt.show() # Display the graph

# Generate PDF report
report_filename = "gcn_report.pdf"

with canvas.Canvas(report_filename, pagesize=letter) as report:
    # Set up report
    report.setTitle("GCN Report")
    report.setAuthor("Your Name")
    report.setFont("Helvetica", 12)

    # Add title
    report.drawCentredString(inch * 4.25, inch * 10.5, "GCN Report")

    # Add date
    today = datetime.today().strftime('%Y-%m-%d')
    report.drawRightString(inch * 7.5, inch * 10.5, f"Date: {today}")

    # Add introduction
    report.drawString(inch, inch * 10, "Introduction:")
    report.drawString(inch, inch * 9.5,
                      "Graph convolutional networks (GCNs) are a type of neural network that can operate on graphs")
    report.drawString(inch, inch * 9,
                      "and have achieved state-of-the-art performance on many graph-based tasks, such as node classification,")
    report.drawString(inch, inch * 8.5,
                      "link prediction, and graph classification. In this report, we use a GCN to predict the outcome")
    report.drawString(inch, inch * 8, "of matches in the game Dota 2.")

    # Add dataset information
    report.drawString(inch, inch * 7, "Dataset Information:")
    report.drawString(inch, inch * 6.5, f"Number of matches: {len(data)}")
    report.drawString(inch, inch * 6, f"Number of features per node: 22")
    report.drawString(inch, inch * 5.5, f"Number of classes: 2 (radiant win or dire win)")

    # Add model architecture
    report.drawString(inch, inch * 4.5, "Model Architecture:")
    report.drawInlineImage("gcn_architecture.png", inch, inch * 3, width=400, height=200)

    # Add results
    report.drawString(inch, inch * 2, "Results:")
    report.drawString(inch, inch * 1.5, f"Test accuracy: {test_acc:.4f}")

    # Save report
    report.save()

# Load saved model
model_path = "gcn_model.pt"
model = GCN(num_features=22, hidden_size=64, num_classes=2).to(device)
model.load_state_dict(torch.load(model_path))

# Make predictions on a new match
new_match = parse_data("parse-data/new_match.json", fetch_heroes())[0]
G = to_networkx(new_match).to_undirected()
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=100, cmap="coolwarm",
                       node_color=new_match.y.numpy())
nx.draw_networkx_edges(G, pos, width=1)
plt.axis("off")
plt.show()

# Predict outcome of new match
model.eval()
x = torch.tensor(new_match.x, dtype=torch.float).unsqueeze(0).to(device)
edge_index = torch.tensor(new_match.edge_index, dtype=torch.long).unsqueeze(0).to(device)
out = model(x, edge_index)
predicted_class = torch.argmax(out, dim=1)
print("Predicted class:", predicted_class.item())

# Generate PDF report with prediction
with canvas.Canvas(report_filename, pagesize=letter) as report:
    # Set up report
    report.setTitle("GCN Report")
    report.setAuthor("Your Name")
    report.setFont("Helvetica", 12)

    # Add title and date from previous report
    report.drawCentredString(inch * 4.25, inch * 10.5, "GCN Report")

    # Add date
    today = datetime.today().strftime('%Y-%m-%d')
    report.drawRightString(inch * 7.5, inch * 10.5, f"Date: {today}")

    # Add introduction and dataset information from previous report
    report.drawString(inch, inch * 10, "Introduction:")
    report.drawString(inch, inch * 9.5,
                      "Graph convolutional networks (GCNs) are a type of neural network that can operate on graphs")
    report.drawString(inch, inch * 9,
                      "and have achieved state-of-the-art performance on many graph-based tasks, such as node classification,")
    report.drawString(inch, inch * 8.5,
                      "link prediction, and graph classification. In this report, we use a GCN to predict the outcome")
    report.drawString(inch, inch * 8, "of matches in the game Dota 2.")

    # Add dataset information
    report.drawString(inch, inch * 7, "Dataset Information:")
    report.drawString(inch, inch * 6.5, f"Number of matches: {len(data)}")
    report.drawString(inch, inch * 6, f"Number of features per node: 22")
    report.drawString(inch, inch * 5.5, f"Number of classes: 2 (radiant win or dire win)")

    # Add model architecture
    report.drawString(inch, inch * 4.5, "Model Architecture:")
    report.drawInlineImage("gcn_architecture.png", inch, inch * 3, width=400, height=200)

    # Add results (new)
    report.drawString(inch, inch * 2, "Results:")
    report.drawString(inch, inch * 1.5, f"Test accuracy: {test_acc:.4f}")

    # Add prediction information to report (new)
    report.drawString(inch, inch * 1, "Prediction:")
    report.drawString(inch, inch * 0.5, f"Predicted class: {predicted_class.item()}")

    # Save
report.save() # Save report with prediction included in the same file as the original report (gcn_report.pdf)