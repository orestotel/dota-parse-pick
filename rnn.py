import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load preprocessed data from files
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

# Hyperparameters
heroes = np.unique(X_train)
vocab_size = len(heroes)
embedding_dim = 32
sequence_length = X_train.shape[1]

# Define the RNN model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    SimpleRNN(64),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
