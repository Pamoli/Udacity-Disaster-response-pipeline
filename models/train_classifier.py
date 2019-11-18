# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    db='sqlite:///'+database_filepath
    engine = create_engine(db)
    df = pd.read_sql_table('df_final', con=engine)
    df=df.drop(["original","genre"],axis=1)
    df=df.dropna(how='any')
    X = df["message"]
    y = df[df.columns[3:]]
    category_names = y.columns
    return X,y,category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens    


def build_model():
    multiclass = MultiOutputClassifier(RandomForestClassifier())
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', multiclass)
    ])
    
    parameters = {'clf__estimator__max_depth': [8, 40, None],
              'clf__estimator__min_samples_leaf':[4, 6, 14]}
    cv = GridSearchCV(pipeline, parameters) 
    return cv

def display_results(y_test, y_pred):
    d = []
    for col in y_test.columns:
        category=col
        precision, recall, f_score, support = precision_recall_fscore_support(np.array(y_test[col]),np.array(y_pred[col]),average='weighted')
        d.append({'Category': category, 'Precision': precision, 'Recall': recall, 'f-Score': f_score})
    print(pd.DataFrame(d))

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_tuned = model.predict(X_test)
    y_pred_tuned = pd.DataFrame(y_pred_tuned)
    y_pred_tuned.columns=Y_test.columns
    display_results(Y_test, y_pred_tuned)
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training model...')
        cv.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()