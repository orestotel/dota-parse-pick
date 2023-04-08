import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyModel(nn.Module):
    def __init__(self, num_heroes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(num_heroes * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  # Added a new fully connected layer
        self.fc4 = nn.Linear(32, 2)  # Updated the final layer to match the output of the previous layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))  # Added the new layer to the forward pass
        x = self.fc4(x)
        return x


def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    hero_set = set()
    for match in data:
        hero_set |= set(match["radiant_team"] + match["dire_team"])
    hero2index = {hero: idx for idx, hero in enumerate(hero_set)}
    num_heroes = len(hero2index)

    X = []
    y = []
    for match in data:
        radiant_team = [0] * num_heroes
        dire_team = [0] * num_heroes
        for hero in match["radiant_team"]:
            radiant_team[hero2index[hero]] = 1
        for hero in match["dire_team"]:
            dire_team[hero2index[hero]] = 1
        X.append(radiant_team + dire_team)
        y.append(1 if match["radiant_win"] else 0)
    return X, y, num_heroes


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_predictions

    return model, epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_predictions

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model_name', type=str, default='newmodel',
                        help='The name of the model to save (default: newmodel)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='The path to the pretrained model to continue training (default: None)')
    parser.add_argument('--test_data', type=str, default='parse-data/bubblegum1.json',
                        help='The path to the test dataset (default: parse-data/bubblegum1.json)')
    args = parser.parse_args()
    model_name = args.model_name
    pretrained_model = args.pretrained_model
    test_data = args.test_data

    # Load data
    X_train, y_train, num_heroes = load_data("parse-data/bubblegum1.json")
    X_test, y_test, _ = load_data(test_data)

    # Split data into training and validation sets
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(X_train_tensor, dtype=torch.float32),
                                  torch.tensor(y_train_tensor, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_tensor, dtype=torch.float32),
                                torch.tensor(y_val_tensor, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))

    train_dataloader = DataLoader(train_dataset, batch_size=100000, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=100000, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=100000, shuffle=False)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    if pretrained_model:
        model = MyModel(num_heroes).to(device)
        model.load_state_dict(torch.load(pretrained_model))
    else:
        model = MyModel(num_heroes).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train model
    num_epochs = 100
    for epoch in range(num_epochs):
        start_time = time.time()

        model, train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%")

    # Evaluate model on test set
    test_loss, test_acc = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"models/{model_name}.pth")

if __name__ == "__main__":
    main()
