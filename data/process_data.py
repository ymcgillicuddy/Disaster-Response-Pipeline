import sys
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    '''
    Reads in two data files and merges them into a single dataframe
    '''
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories = pd.read_csv(categories_filepath, encoding='latin-1')
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Reads in dataframe
    Parses compound categories column into 36 binary columns
    Joins the binary columns back onto the dataframe; drops the original "categories" column
    Drops non-binary records and duplicates
    Returns cleansed dataframe
    '''
    categories = df['categories'].str.split(';',n=36, expand = True) 
    #split string on semi-colon# create a dataframe of the 36 individual category columns

    row = categories.iloc[0] # select the first row of the categories dataframe

    row["category_colnames"] = categories.iloc[0].str.slice(0,-2)
    categories.columns = row["category_colnames"]
    #slice off last 2 characters from first row to generate column headers

    for column in categories:
        categories[column] = categories[column].str.slice(-1,)
        categories[column] = pd.to_numeric(categories[column])
        # take the last character of the string as the value and ensure values are numerical

    df = df.drop(['categories'], axis=1) # drop the original categories column from `df`
    df = pd.concat([df,categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe
    df = df.drop(df[df['related'] == 2].index) #drop some rows with "2" in related
    df = df.drop(['child_alone'], axis=1) #drop this column as it had zero 1 values so can't be used for prediction model
    df = df.drop_duplicates() # drop duplicates   
    return df


def save_data(df, database_filename):
    '''
    Reads in dataframe and a database_filename in the form "filename.db"
    Outputs the dataframe to a SQL database at the given filepath
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists = 'replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
