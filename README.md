# Disaster-Response-Pipeline
Udacity Data Science Project 2

## Overview
The ETL Pipeline reads in two data files containing social media data about natural disasters and prepares them for the ML Pipeline.
The ML Pipeline reads in the SQL database created by the ETL Pipeline and runs the message text through a ML model and builds/evaluates a classification model to predict categories based on message text.
Details on both pipelines are separated below for clarity.

The ETL Pipline contains functions that:
* **_load_data_**: Loads in & merges the data files
* **_clean_data_**: Parses out a compound "categories column" into 36 binary columns, that are then joined back onto the dataframe, with some non-binary records, un-helpful columns and duplicated records dropped.
* **_save_data_**: Outputs the dataframe to a SQL database at a given filepath

The ML Pipeline contains functions that:
* **_load_data_**: Loads in the SQL database and defines X and Y values
* **_tokenize_**: Uses Natural Language Toolkit (NLTK) to standardise text and prepare it for the ML model through tokenization, part-of-speech tagging, lemmatization and stopword removal.
* **_build_model_**: A pipeline that counts how often tokens occur, weights them and classifies accross multiple categories
* **_evaluate_model_**: Trains, tests, adjusts and re-fits the pipline model.
* **_save_model_**: Saves the tuned model to a pickle file

## ETL Pipeline
### Components
* Python file - 'process_data.py'
* CSV files - 'messages.csv', 'categories.csv' - **_input_**
* SQL database - 'DisasterResponse.db' - **_output_**
* Jupyter Notebook - 'ETL Pipeline Preparation.ipynb' - included to show data exploration process

### Requirements
To run the process_data.py code you will need to ensure that you have the following installed in your environment:
* sqlite3
* sqlalchemy
* pandas

### Known Bugs
Running the python file locally causes a key error that does not occur when running the modular code in Jupyter Notebook.  Running the python file in an IDE environment also works as intended.  It's not currently clear to me what is causing this discrepancy.

## ML Pipeline
### Components
* Jupyter Notebook?
* Python file - 'train_classifier.py'
* SQL database - 'DisasterResponse.db' - **_input_**
* Pickle file - 'classifier.pkl' - **_output_**

### Requirements
To run the train_classifier.py code you will need to ensure that you have the following installed in your environment:
* re
* nltk
* sqlite3
* sqlalchemy
* pandas
* sklearn
* pickle
