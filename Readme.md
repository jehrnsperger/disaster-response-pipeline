
### Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Instructions](#instructions)
4. [Main Components](#files)
5. [Files](#filetree)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Introduction <a name="introduction"></a>
This Repository classifies text messages into one or more of 36 predefined categories. During emergencies such as natural 
disasters it is important to classify incoming messages as quickly as possible and allocate them to the correct organisations. 
For instance, the text message "We need food now!" should be allocated to categories such as 'Food' and 'Aid Related'.

## Prerequisites<a name="prerequisites"></a>

To install the requirements in requirements.txt, run
```
pip install -r requirements.txt
```

## Instructions<a name="instructions"></a>
1. Run the Processing Pipeline\
Provide the filepaths of the messages and categories datasets as the first and second argument respectively, as
well as the filepath of the database to save the cleaned data to as the third argument. Example:
```
python process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv .data/DisasterResponse.db
```
2. Run the ML-Pipeline
provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to
save the model to as the second argument. Example: 
```
python train_classifier.py ./data/DisasterResponse.db ./models/classifier.pkl 
```
3. Start the Flask-Web-App by running
```
python run.py
```
Visit the Website at http://0.0.0.0:3001/

## Main Components <a name="files"></a>
#### process_data.py
Reads in two .csv files, 'messages' and 'categories'. The files are merged on an ID, cleaned, and stored as SQL-table in the database DisasterResponse.db.

#### train_classifier.py
Reads the table from DisasterResponse.db and preprocesses the text messages for further use in the NLP-Pipeline. 
Since each message can be classified for multiple labels, the MultiOutputClassifier serves as a wrapper for the RandomForestClassifier.
ATF-IDF vectorizer is used for feature extraction of the text data and the model is trained using GridSearchCV. F1-Score, Precision, and Recall are reported on unseen test data 
and the trained model is saved as classifier.pkl. 

#### run.py
Generates the Flask-Web-App. The user can input a text message, which is then classified by the trained model. 
The predicted categories are highlighted. The App also shows some basic plots about the dataset that was used for training.

## Files<a name="filetree"></a>
```
.
|-- LICENSE
|-- Readme.md
|-- data
|   |-- disaster_categories.csv
|   `-- disaster_messages.csv
|-- models
|-- process_data.py
|-- requirements.txt
|-- run.py
|-- templates
|   |-- go.html
|   `-- master.html
`-- train_classifier.py
```

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Licensed under the MIT license and provided by [Udacity](https://www.udacity.com). 

