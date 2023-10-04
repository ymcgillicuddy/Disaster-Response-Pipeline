# Disaster-Response-Pipeline
Udacity Data Science Project 2

## Overview
The ETL Pipeline reads reads in two data files containing social media data about natural disasters and prepares them for the ML Pipeline.
The ML Pipeline...
Details on both pipelines are separated below for clarity.

The ETL Pipline contains functions that:
* **_load_data_**: Load in & merge the data files
* **_clean_data_**: Parse out a compound "categories column" into 36 binary columns, that are then joined back onto the dataframe, with some non-binary records, un-helpful columns and duplicated records dropped.
* **_save_data_**: Outputs the dataframe to a SQL database at a given filepath

The ML Pipeline contains functions that:
*
*

## ETL Pipeline
### Components
* Jupyter Notebook - 'ETL Pipeline Preparation.ipynb' - included to show data exploration process
* Python file - 'process_data.py'
* CSV files - 'messages.csv', 'categories.csv'
* SQL database - 'DisasterResponse.db' - **_output_**

### Requirements
To utilise the notebook you will need to ensure that you have the following installed in your environment:
* sqlite3
* sqlalchemy
* pandas

## ML Pipeline
### Components
* Jupyter Notebook?
* Python file - 'train_classifier.py'

### Requirements
