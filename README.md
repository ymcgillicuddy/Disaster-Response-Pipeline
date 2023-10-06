# Disaster-Response-Pipeline
Udacity Data Science Project 2

## Overview
The ETL Pipeline reads in two data files containing social media data about natural disasters and prepares them for the ML Pipeline.

The ML Pipeline reads in the SQL database created by the ETL Pipeline and runs the message text through a ML model and builds/evaluates a classification model to predict aid categories based on message text.

The web app allows the user to enter message text and see how the model classifies that against the different aid categories in the dataset

Details on all elements are separated below for clarity.

The ETL Pipline contains functions that:
* **_load_data_**: Loads in & merges the data files
* **_clean_data_**: Parses out a compound "categories column" into 36 binary columns, that are then joined back onto the dataframe, with some non-binary records, un-helpful columns and duplicated records dropped.
* **_save_data_**: Outputs the dataframe to a SQL database at a given filepath

The ML Pipeline contains functions that:
* **_load_data_**: Loads in the SQL database and defines X, Y and category name values
* **_tokenize_**: Uses Natural Language Toolkit (NLTK) to standardise text and prepare it for the ML model through tokenization, part-of-speech tagging, lemmatization and stopword removal.
* **_build_model_**: A pipeline that counts how often tokens occur, weights them and classifies accross multiple categories; and tests parameters that can be adjusted.
* **_evaluate_model_**: Returns an evaluation of the model's performance
* **_save_model_**: Saves the tuned model to a pickle file

The web app...
* imports the database from the ETL Pipeline and the model from the ML pipeline
* ...

## Universal Requirements
The following are required to run all python elements of this pipeline.  To view which are required for each code, view the import statements at the start of the code.
* sqlite3
* sqlalchemy
* pandas
* ntlk
* sklearn
* re - **_ML Pipeline only_**
* pickle - **_ML Pipeline only_**
* json - **_web app only_**
* plotly - **_web app only_**
* flask - **_web app only_**
* joblib - **_web app only_**

## ETL Pipeline
### Components
* Python file - 'process_data.py'
* CSV files - 'messages.csv', 'categories.csv' - **_input_**
* SQL database - 'DisasterResponse.db' - **_output_**
* Jupyter Notebook - 'ETL Pipeline Preparation.ipynb' - included to show data exploration process

### Known Bugs
Running the python file locally causes a key error that does not occur when running the modular code in Jupyter Notebook.  Running the python file in an IDE environment also works as intended.  It's not currently clear to me what is causing this discrepancy.

## ML Pipeline
### Components
* Python file - 'train_classifier.py'
* SQL database - 'DisasterResponse.db' - **_input_**
* Pickle file - 'classifier.pkl' - **_output_**

## Web App
### Components
* Python file - 'python run.py'
* SQL database - 'DisasterResponse.db' - **_input_**
* Pickle file - 'classifier.pkl' - **_input_**
