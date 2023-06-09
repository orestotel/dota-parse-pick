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



# Preprocess and encode data
heroes = set()
for match in data:
    heroes |= set(match["radiant_team"]) | set(match["dire_team"])
heroes = list(heroes)
hero_to_idx = {hero: idx for idx, hero in enumerate(heroes)}

# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")
sys.stdout.flush()


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

# Define FNN model
class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
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
    return torch.tensor(np.array(tensor_data), dtype=torch.float32)


# Start loading animation
stop_loading_animation = threading.Event()
loading_thread = threading.Thread(target=loading_animation, args=(stop_loading_animation,))
loading_thread.start()

X_train_tensor = data_to_tensor(X_train).to(device)
X_test_tensor = data_to_tensor(X_test).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# Stop loading animation
stop_loading_animation.set()
loading_thread.join()
sys.stdout.write("\r")
sys.stdout.flush()


# Training
from torch.utils.tensorboard import SummaryWriter

def train(X_train, y_train, num_epochs=10000, hidden_size=321, starting_epoch=0, pretrained_model=None, log_dir="runs"):
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    input_size = len(heroes) * 2
    num_classes = 2
    model = pretrained_model.to(device) if pretrained_model else FNNModel(input_size, hidden_size, num_classes).to(
        device)
    model = model.to(device)  # Add this line

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Ask the user if they want to use TensorBoard
    use_tensorboard = input("Do you want to use TensorBoard for visualization? (y/n): ").lower() == 'y'

    if use_tensorboard:
        # Create TensorBoard writer
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(starting_epoch, num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        if use_tensorboard:
            # Log loss and model graph to TensorBoard
            writer.add_scalar("Loss/train", loss, epoch)
            if epoch == starting_epoch:
                writer.add_graph(model, X_train)

        # Save the model after every 100 epochs
        if (epoch + 1) % 100 == 0:
            model_save_path = f"models/intermediate_model_epoch_{epoch + 1}.pkl"
            #model_save_path = f"models/intermediate_model_epoch.pkl"
            with open(model_save_path, 'wb') as file:
                pickle.dump(model, file)

    if use_tensorboard:
        writer.close()

    return model


# Evaluation
def predict(model, radiant_team, dire_team):
    radiant_team_idx = [hero_to_idx[hero] for hero in radiant_team]
    dire_team_idx = [hero_to_idx[hero] for hero in dire_team]
    input_data = [(radiant_team_idx, dire_team_idx)]
    input_tensor = data_to_tensor(input_data).to(device)  # Make sure the input tensor is on the same device as the model
    with torch.no_grad():
        outputs = model(input_tensor)  # The input tensor is now on the correct device (GPU)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()[0]

# Load the model with a loading bar
def load_model_with_loading_bar(model_filename):
    try:
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
    except EOFError:
        print("Error: The model file is empty or corrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    return model



#if specified same name and trained new model models will overwrite
#to train new model change name beforehand
#careful!
print("#if specified same name and trained new model models will overwrite \n to train new model change name beforehand \n careful!")
model_filename = "models/intermediate_model_epoch_1500.pkl"

user_choice = input("\nEnter '1' to use the hardcoded trained model, '2' to create and use a new model, '3' to continue training hardcoded model: ")

if user_choice == '1':
    model = load_model_with_loading_bar(model_filename)
    model = model.to(device)  # Add this line
elif user_choice == '2':
    model = training_module.train(X_train_tensor, y_train_tensor)
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    model = model.to(device)  # Add this line
elif user_choice == '3':
    # Load the saved model and starting_epoch
    starting_epoch = int(input("Enter the starting epoch from 10000: "))
    model = load_model_with_loading_bar(model_filename)
    model = model.to(device)  # Add this line

    # Resume training
    model = training_module.train(X_train_tensor, y_train_tensor, num_epochs=10000, hidden_size=321,
                                  starting_epoch=starting_epoch, pretrained_model=model)
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
else:
    print("Invalid input. Exiting.")
    exit()

# Create a GUI for prediction
def predict(model, radiant_team, dire_team):
    radiant_team_idx = [hero_to_idx[hero] for hero in radiant_team]
    dire_team_idx = [hero_to_idx[hero] for hero in dire_team]
    input_data = [(radiant_team_idx, dire_team_idx)]
    input_tensor = data_to_tensor(input_data).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()[0]

def predict_callback():
    radiant_team = [rt_hero1.get(), rt_hero2.get(), rt_hero3.get(), rt_hero4.get(), rt_hero5.get()]
    dire_team = [dt_hero1.get(), dt_hero2.get(), dt_hero3.get(), dt_hero4.get(), dt_hero5.get()]
    probabilities = predict(model, radiant_team, dire_team)
    radiant_win_probability = probabilities[1] * 100
    result_text.set(f"Radiant Win Chance: {radiant_win_probability:.2f}%")
    print(f"Radiant Win Chance: {radiant_win_probability:.2f}%")



root = tk.Tk()
root.title("Dota 2 Match Predictor")

# Create the dropdown menus
heroes.sort()
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

# Add Radiant Team and Dire Team labels
radiant_label = tk.Label(root, text="Radiant Team", font=("Helvetica", 14))
dire_label = tk.Label(root, text="Dire Team", font=("Helvetica", 14))

# Add Predict button
predict_button = tk.Button(root, text="Predict", command=predict_callback)

# Add result label
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 14))

# Grid layout
radiant_label.grid(row=0, column=0, padx=10, pady=10)
rt_hero1.grid(row=1, column=0, padx=5, pady=5)
rt_hero2.grid(row=2, column=0, padx=5, pady=5)
rt_hero3.grid(row=3, column=0, padx=5, pady=5)
rt_hero4.grid(row=4, column=0, padx=5, pady=5)
rt_hero5.grid(row=5, column=0, padx=5, pady=5)

dire_label.grid(row=0, column=1, padx=10, pady=10)
dt_hero1.grid(row=1, column=1, padx=5, pady=5)
dt_hero2.grid(row=2, column=1, padx=5, pady=5)
dt_hero3.grid(row=3, column=1, padx=5, pady=5)
dt_hero4.grid(row=4, column=1, padx=5, pady=5)
dt_hero5.grid(row=5, column=1, padx=5, pady=5)

predict_button.grid(row=6, column=0, padx=10, pady=10)
result_label.grid(row=6, column=1, padx=10, pady=10)

# Start the GUI
root.mainloop()