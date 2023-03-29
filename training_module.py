import torch
import torch.nn as nn
import torch.optim as optim
import pickle

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

def train(X_train, y_train, num_epochs=1000, hidden_size=321, starting_epoch=0, pretrained_model=None):
    input_size = X_train.shape[1]
    num_classes = 2
    model = pretrained_model if pretrained_model else FNNModel(input_size, hidden_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(starting_epoch, num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Save the model after every 100 epochs
        if (epoch + 1) % 100 == 0:
            model_save_path = f"models/intermediate_model_epoch_{epoch + 1}.pkl"
            with open(model_save_path, 'wb') as file:
                pickle.dump(model, file)

    return model
