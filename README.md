Dota 2 Match Predictor
This project is a Dota 2 match prediction tool that uses a Feedforward Neural Network (FNN) to predict the outcome of a Dota 2 match based on hero selections. The model is trained on a dataset of past matches and their outcomes.

Table of Contents
Installation
Usage
Training the Model
Evaluation
Prediction
License
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/dota2-match-predictor.git
Install the required dependencies:
Copy code
pip install -r requirements.txt
Download the dataset and place it in the appropriate folder:
bash
Copy code
parse-data/batches/
Usage
Run the main script:
Copy code
python dota2_match_predictor.py
Choose one of the following options:
Use the hardcoded trained model.
Create and use a new model.
Continue training the hardcoded model.
If you choose to continue training the hardcoded model, enter the starting epoch.

Use the GUI to enter the hero selections for both the Radiant and Dire teams.

Click on "Predict" to get the probability of the Radiant team winning the match.

Training the Model
To train the model, run the main script and choose the option to create and use a new model. The model will be trained for the specified number of epochs. You can adjust the number of epochs, learning rate, and hidden size in the train function call.

After every 100 epochs, the model will be saved as an intermediate model. You can resume training from these intermediate models by selecting the option to continue training the hardcoded model.

Evaluation
To evaluate the model, run the evaluate function with the test dataset. It will calculate the accuracy of the model in predicting the outcomes of the test matches.

Prediction
The model can predict the probability of the Radiant team winning based on the hero selections for both teams. Use the GUI to input the hero selections and click on "Predict" to get the win probability.

License
This project is licensed under the MIT License. See the LICENSE file for more information.


﻿# dota-parse-pick
hello hello
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 0 0"/>![image](https://user-images.githubusercontent.com/28295297/228538621-a26c5b20-afcf-40a1-9150-348af95fe121.png)
![image](https://user-images.githubusercontent.com/28295297/228538784-d30f6ab7-28c0-4f04-a46d-0c5a8e8627a2.png)
