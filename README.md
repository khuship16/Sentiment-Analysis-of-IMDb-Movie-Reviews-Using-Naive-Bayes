CS 410 Final Project Software - Group 29

Project: Sentiment Analysis with Naive Bayes

Project Overview: This project implements sentiment analysis on a dataset of IMDb movie reviews using various Naive Bayes models. We evaluate the performance of unigram and bigram models, both with and without TF-IDF weighting.

Group Members:
Noah Antisseril
Khushi Patel
Praneetha Bhogi
Madhav Parthasarathy

Included Files:
naive_bayes.py - Implementation of unigram Naive Bayes classifier
naive_bayes_tfidf.py - Implementation of unigram Naive Bayes classifier with inverse document frequency weighting for words
naive_bayes_bigram.py - Implementation of bigram Naive Bayes classifier
naive_bayes_bigram_tfidf.py - Implementation of bigram Naive Bayes classifier with inverse document frequency weighting for words
main.ipynb - Jupyter notebook that runs and compares the performance of the various Naive Bayes implementations
dataset.csv - Kaggle dataset containing roughly 50,000 movie reviews taken from the website IMDb

Instructions:
Independent Model Training and Testing -> The .py files can all be run independently to train/test the specific implementation
Model Comparison -> Run main.ipynb in a Jupyter Notebook environment to train, test, and compare all models’ performance.