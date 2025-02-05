# Sentiment Analysis of IMDb Movie Reviews
This project applies Naive Bayes for sentiment analysis of IMDb movie reviews. 

📄 **[Full Report (PDF)](./Sentiment_Review_Full_Report.pdf)**

## Overview
- This project implements sentiment analysis on a dataset of IMDb movie reviews using various Naive Bayes models. We evaluate the performance of unigram and bigram models, both with and without TF-IDF weighting.
  - Implemented text preprocessing with stopword removal, TF-IDF, and n-gram vectorization.
  - Developed multiple Naive Bayes models (Unigram, Bigram, IDF, Mixture).
  - Evaluated model performance using accuracy, precision, recall, and confusion matrices.

## Technologies Used
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
- **Techniques:** Naive Bayes, TF-IDF, N-grams, Laplace Smoothing, Confusion Matrices

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
