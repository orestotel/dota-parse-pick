import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
import json

# Process match data for this kind of dataset:
#[
#    {
#        "match_id": 7099695906,
#        "radiant_win": false,
#       "radiant_team": [
#           "Dark Willow",
#            "Sniper",
#            "Oracle",
#            "Spectre",
#            "Lone Druid"
#        ],
#        "dire_team": [
#            "Bristleback",
#            "Juggernaut",
#            "Dazzle",
#            "Lion",
#            "Weaver"
#        ]
#    },...
# ]
# to this kind of dataset:
#[
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...], # 0 means hero is not in the match, 1 means hero is in the match
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
#    ...
#]
# which is basically a matrix of 0s and 1s where each row represents a match and each column represents a hero
# meaning that if a hero is in a match, the corresponding value in the matrix is 1, otherwise it is 0
# and this is the input data for the neural network
# and the labels are the match results (radiant_win)
# so the neural network will learn to predict the match results based on the heroes in the match
def process_match_data(match_data): # match_data is a list of matches
    dataset = [] # input data for the neural network
    labels = [] # labels for the input data

    for match in match_data: # match is a dictionary
        radiant_team = match["radiant_team"] # radiant_team is a list of heroes
        dire_team = match["dire_team"] # dire_team is a list of heroes
        radiant_win = match["radiant_win"] # radiant_win is a boolean

        match_heroes = [0] * 10  # 10 represents the total number of heroes

        for hero in radiant_team: # hero is a string
            hero_index = get_hero_index(hero) # get the index of the hero in the hero_names list
            match_heroes[hero_index] = 1 # set the corresponding value in the match_heroes list to 1

        for hero in dire_team: # hero is a string
            hero_index = get_hero_index(hero) # get the index of the hero in the hero_names list
            match_heroes[hero_index] = 1 # set the corresponding value in the match_heroes list to 1

        dataset.append(match_heroes) # append the match_heroes list to the dataset list
        labels.append(1 if radiant_win else 0) # append 1 if radiant_win is True, otherwise append 0

    return np.array(dataset), np.array(labels) # return the dataset and labels as numpy arrays


# Load data
with open("D:\\sofardata\\chewtree_process_0_copy.json", "r") as file:
    data = json.load(file)

# Fetch hero names from OpenDota API and store them in a list called hero_names
response = requests.get("https://api.opendota.com/api/heroes", headers={"Authorization": "Bearer 2a5b8577-d7ee-4ef2-85ca-f15e5c8bdf75"})
heroes = response.json() # heroes is a list of dictionaries
hero_names = [hero['localized_name'] for hero in heroes] # hero_names is a list of strings

# Encode hero names to integers using LabelEncoder
encoder = LabelEncoder() # create a LabelEncoder object
encoder.fit(hero_names) # fit the encoder to the hero_names list

# Process match data into dataset and labels
matches, labels = process_match_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(matches, labels, test_size=0.2, random_state=42)

# Define the neural network model architecture using Keras Sequential API and Dense layers with ReLU activation function
# and Dropout layers with dropout rate of 0.3 to prevent overfitting and a final Dense layer with sigmoid activation function
# to output a probability between 0 and 1 for each match (0 means radiant loss, 1 means radiant win)
# and print the model summary to see the model architecture
model = Sequential([
    Dense(256, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.summary() # print the model summary to see the model architecture and the number of trainable parameters

# Compile the model with RMSprop optimizer, categorical cross-entropy loss, and AUC-ROC metric
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

# Train the model on the training data for 100 epochs with batch size of 32 and print the AUC-ROC value
# on the training data after training is complete to see how well the model is performing on the training data
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the trained model on the testing data and print the AUC-ROC value on the testing data
# to see how well the model is performing on the testing data and compare it to the AUC-ROC value on the training data
auc_score = model.evaluate(X_test, y_test, verbose=0)[1] # get the AUC-ROC value on the testing data
print(f"AUC-ROC: {auc_score}") # print the AUC-ROC value on the testing data

# Save the trained model to a file located in d-disk-repo/dota-parse-pick/, which is the current directory
model.save("D:\\d-disk-repo\\dota-parse-pick\\modelix.h5") # save the trained model to a file

# A simple test to see if the model is working properly that picks random matches from the dataset and predicts the match results
# and prints the predicted result and the actual result, as well as the heroes for each team in the match
# amd the probability of radiant win predicted by the model for each match and the actual probability of radiant win
# and the difference between the predicted probability and the actual probability. Here is the complete function
# that does this test:
def test_model(model, matches, labels, hero_names): # model is the trained model, matches is the dataset, labels is the labels, hero_names is the list of hero names
    for i in range(10): # pick 10 random matches from the dataset and test the model on them
        index = random.randint(0, len(matches) - 1) # pick a random index from 0 to the length of the dataset - 1
        match = matches[index] # get the match at the random index
        label = labels[index] # get the label at the random index
        radiant_win = label == 1 # radiant_win is True if label is 1, otherwise it is False (radiant_win is a boolean)
        radiant_team = [] # list of radiant team heroes
        dire_team = [] # list of dire team heroes

        for i in range(len(match)): # iterate over the match list
            if match[i] == 1: # if the value at the current index is 1, then the hero is in the match
                if i < 5: # if the index is less than 5, then the hero is in the radiant team
                    radiant_team.append(hero_names[i]) # append the hero name to the radiant_team list
                else: # otherwise the hero is in the dire team
                    dire_team.append(hero_names[i]) # append the hero name to the dire_team list

        radiant_team = ", ".join(radiant_team) # join the radiant_team list into a string
        dire_team = ", ".join(dire_team) # join the dire_team list into a string

        match = np.array([match]) # convert the match list into a numpy array
        prediction = model.predict(match)[0][0] # get the prediction for the match
        prediction = round(prediction, 2) # round the prediction to 2 decimal places
        print(f"Predicted: {prediction}, Actual: {label}, Difference: {abs(prediction - label)}") # print the predicted result, the actual result, and the difference between them
        print(f"Radiant Team: {radiant_team}") # print the radiant team heroes
        print(f"Dire Team: {dire_team}") # print the dire team heroes
        print(f"Probability of Radiant Win: {prediction}, Actual Probability of Radiant Win: {label}") # print the predicted probability of radiant win and the actual probability of radiant win
        print() # print an empty line





