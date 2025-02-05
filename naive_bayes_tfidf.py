# Import necessary libraries for data manipulation, text preprocessing, and evaluation
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

'''
Only using IDF from TF-IDF
Not using TF since it depends on document, and we do not need to rank documents
Using IDF to more heavily weight rare words
'''

# Filepath to the dataset containing reviews and sentiments
filepath_dataset = "dataset.csv"

class NaiveBayes_IDF:
    def __init__(self):
        # Helper function to convert sentiment labels ("positive"/"negative") into integers
        def text_to_int(text):
            if text == "positive":
                return 1
            elif text == "negative":
                return 0

        # Download necessary NLTK resources
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        # Initialize stopwords and lemmatizer for preprocessing
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Load dataset and preprocess text data
        self.df = pd.read_csv(filepath_dataset)
        self.df['processed_review'] = self.df['review'].apply(self.preprocess)  # Clean text
        self.df['sentiment'] = self.df['sentiment'].apply(text_to_int)  # Convert sentiments to integers

        # Split dataset into training and testing sets
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        # Initialize word counts and document frequencies
        self.pos_word_counts = defaultdict(int)  # Positive class word frequencies
        self.neg_word_counts = defaultdict(int)  # Negative class word frequencies
        self.num_words = 0  # Total number of unique words
        self.num_neg = 0  # Total words in negative reviews
        self.num_pos = 0  # Total words in positive reviews

        # Document frequency tracking and total document count
        self.doc_freq = defaultdict(int)  # Number of documents containing each word
        self.M = len(self.df)  # Total number of documents

    # Clean and preprocess text (lowercasing, removing special characters, lemmatizing, etc.)
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r"<.*?>", '', text)  # Remove HTML tags
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Remove special characters
        text = re.sub(r"\d+", '', text)  # Remove digits
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Remove stopwords
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        return ' '.join(words)  # Return processed text as a single string

    # Train the Naive Bayes model with IDF weighting
    def train_naive_bayes(self):
        unique_words = set()  # Set to track unique words in training data
        for _, row in self.train_df.iterrows():
            _, label, doc = row  # Extract sentiment label and processed review
            doc_words = doc.split()
            used = set()  # To track words already counted for document frequency
            for word in doc_words:
                # Increment word counts based on sentiment label
                if label == 1:
                    self.pos_word_counts[word] += 1
                    self.num_pos += 1
                elif label == 0:
                    self.neg_word_counts[word] += 1
                    self.num_neg += 1
                unique_words.add(word)  # Add to unique words
                if word not in used:  # Update document frequency only once per document
                    self.doc_freq[word] += 1
                    used.add(word)
        self.num_words = len(unique_words)  # Total number of unique words

    # Naive Bayes classifier with IDF weighting
    def naive_bayes(self, hold_text, laplace):
        text = hold_text.split()  # Tokenize input text
        log_pos_prob = 0  # Log probability for positive class
        log_neg_prob = 0  # Log probability for negative class
        for word in text:
            # Calculate smoothed probabilities for the word in each class
            curr_pos = (self.pos_word_counts[word] + laplace) / (self.num_pos + laplace * self.num_words)
            curr_neg = (self.neg_word_counts[word] + laplace) / (self.num_neg + laplace * self.num_words)
            # Apply IDF weighting if the word exists in the corpus
            if self.doc_freq[word] == 0:
                curr_pos = 1
                curr_neg = 1
            else:
                curr_pos *= math.log(1 + (self.M + 1) / self.doc_freq[word])
                curr_neg *= math.log(1 + (self.M + 1) / self.doc_freq[word])
            # Update log probabilities
            log_pos_prob += math.log(curr_pos)
            log_neg_prob += math.log(curr_neg)
        # Return the class with the higher probability
        return 1 if log_pos_prob > log_neg_prob else 0

    # Apply the Naive Bayes classifier to the test dataset
    def naive_bayes_test_df(self, laplace=1.0):
        self.test_df['predictions'] = self.test_df['processed_review'].apply(
            lambda text: self.naive_bayes(text, laplace))

    # Print evaluation metrics (accuracy, precision, recall, confusion matrix)
    def print_statistics(self):
        predicted, true = self.test_df["predictions"], self.test_df["sentiment"]
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, average='binary')
        recall = recall_score(true, predicted, average='binary')
        conf_matrix = confusion_matrix(true, predicted)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Confusion Matrix:\n", conf_matrix)
        return accuracy, precision, recall  # Return metrics

# Main script
if __name__ == "__main__":
    test = NaiveBayes_IDF()  # Instantiate the model
    test.train_naive_bayes()  # Train the model
    test.naive_bayes_test_df()  # Test the model
    test.print_statistics()  # Print and return evaluation metrics