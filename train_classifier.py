import sys
import re
import nltk
nltk.download(['wordnet','stopwords','punkt','averaged_perceptron_tagger','universal_tagset'])

import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    '''
    Reads in a SQL database and defines X, Y and category_names values ahead of building a data model
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, 'sqlite:///' + database_filepath)
    X = df['message'].values #define X values
    Y = df[df.columns[4:]].astype('int') #define Y values
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Uses natural language toolkit (NLTK) to...
    Convert a body of text to lower case
    Create a list of words (tokenisation)
    Tag the parts of speech (POS) e.g. nouns, verbs
    Remove verb stems for consistency (lemmatization)
    Remove stopwords that aren't valuable for analysis
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower()) 
    tokens = word_tokenize(text)
    tokens = pos_tag(tokens, tagset='universal')
    lemmed = [WordNetLemmatizer().lemmatize(tok[0], pos='v') for tok in tokens]
    lemmed = [l for l in lemmed if l not in stopwords.words ("english")]
    
    return lemmed


def build_model():
    '''
    Utilises the tokenize function output as a custom tokenizer within a pipeline that...
    Counts each time an instance occurs using CountVectorizer
    Weights those counts using TfidfTransformer
    Uses the RandomForestClassifier to classify each instance across all available categories in the dataset

    Utilises GridSearch CV to test alternative parameters
    '''
    pipeline = Pipeline([
        ('t_count', CountVectorizer(tokenizer=tokenize)),
        ('weighted', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(),n_jobs = -1)),
    ])
    parameters = {'clf__estimator__n_estimators': [50,100,200]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)
    #uses GridSearch CV to try different parameters against the RandomForestClassifier in the model
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Returns the model's test scores across all categories data using classification_report
    Returns the best parameters
    '''
    Y_pred = model.predict(X_test) 
    cr = classification_report(Y_test, pd.DataFrame(Y_pred, columns=category_names), target_names=category_names)
    #stores f1 score, precision and recall by iterating through all category columns
    print(cr)
    print(model.best_params_)


def save_model(model, model_filepath):
    '''
    Saves the model to the given filepath as a pickle file
    '''
    with open(model_filepath, 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
    
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
