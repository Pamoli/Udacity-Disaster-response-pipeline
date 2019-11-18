# Udacity-Disaster-response-pipeline
In this project, we'll to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

We'll use two datasets data ses in this project, one containing real messages that were sent during disaster events, other having the classification of the messages across 36 categories (e.g., aid_relaed, missing_people, storm, fire etc.). The objective is to create a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

This project will also include a web app where an emergency worker can input a new message and get classification results in several categories along with some basic visualizations of the data. <br/>

# Description of the files in the repository: 
**1. data folder:** <br/>
In this folder, there are four files: <br/>
**- disaster_messages:** This csv file contains real messages that were sent during disaster events <br/>
**- disaster_categories:** This csv file contains categories of the messages <br/>
**- process_data.py:**  A data cleaning pipeline that loads the messages and categories datasets, merges the two datasets, cleans the data and stores it in a SQLite database <br/>
**- DisasterResponse:** The SQLite database that contains the final data <br/>
<br/>
**2. models folder**: <br/>
In this folder, there are two files: <br/>
**- train_classifier.py:**  A machine learning pipeline that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set and exports the final model as a pickle file <br/>
**- classifier.pkl:** The pickled model <br/>
<br/>
**3. app folder**: <br/>
In this folder, there are two files: <br/>
**- templates:**  Templates for the web app <br/>
**- run.py:** Python script to run the web app <br/>
<br/>
**4. Notebooks folder**: <br/>
This folder contains the Jupyter notebook versions of process_data.py and train_classifier.py <br/>
<br/>

# How to run: <br/>
1. Run process_data.py with message dataset name, categories dataset name and database name as additional arguments.
Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
2. Run train_classifier.py with database name and pickled model name as additional arguments.
Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl
3. Run run.py that'll run the web app
4. Go to http://0.0.0.0:3001/
