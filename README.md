# Udacity-Disaster-response-pipeline
In this project, we'll to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

We'll use two datasets data ses in this project, one containing real messages that were sent during disaster events, other having the classification of the messages across 36 categories (e.g., aid_relaed, missing_people, storm, fire etc.). The objective is to create a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

This project will also include a web app where an emergency worker can input a new message and get classification results in several categories along with some basic visualizations of the data.

# Description of the files in the repository: <br/>
**1. data folder**: <br/>
In this folder, there are four files: <br/>
**- disaster_messages:** This csv file contains real messages that were sent during disaster events <br/>
**- disaster_categories:** This csv file contains categories of the messages <br/>
**- process_data.py:**  
a data cleaning pipeline that:
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
