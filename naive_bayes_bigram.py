# Import required libraries for data processing, text manipulation, and evaluation
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

# Path to the dataset containing reviews and sentiments
filepath_dataset = "dataset.csv"

class BigramNaiveBayes:
    def __init__(self):
        # Convert sentiment labels ("positive"/"negative") into integers
        def text_to_int(text):
            return 1 if text == "positive" else 0

        # Download necessary NLTK resources for stopwords and lemmatization
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        # Initialize stopwords and lemmatizer for preprocessing
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Load the dataset and preprocess the reviews
        self.df = pd.read_csv(filepath_dataset)
        self.df['processed_review'] = self.df['review'].apply(self.preprocess)  # Clean text
        self.df['sentiment'] = self.df['sentiment'].apply(text_to_int)  # Convert sentiment labels to integers

        # Split the dataset into training and testing sets
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        # Initialize unigram and bigram counts
        self.bi_pos_counts, self.bi_neg_counts = defaultdict(int), defaultdict(int)  # Bigram counts
        self.uni_pos_counts, self.uni_neg_counts = defaultdict(int), defaultdict(int)  # Unigram counts

        # Initialize total unigram and bigram counts for each class
        self.uni_num_pos, self.uni_num_neg = 0, 0  # Total unigrams in positive and negative classes
        self.bi_num_pos, self.bi_num_neg = 0, 0  # Total bigrams in positive and negative classes

        # Initialize unique unigram and bigram counts
        self.num_uni = 0  # Total unique unigrams
        self.num_bi = 0  # Total unique bigrams

    # Clean and preprocess text by removing special characters, lowercasing, and lemmatizing
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", ' ', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r"<.*?>", ' ', text)  # Remove HTML tags
        text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text)  # Remove special characters
        text = re.sub(r"\d+", ' ', text)  # Remove digits
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Remove stopwords
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        return ' '.join(words)  # Return cleaned text

    # Train the Naive Bayes model by calculating unigram and bigram probabilities
    def train_naive_bayes(self):
        unique_uni = set()  # Track unique unigrams
        unique_bi = set()  # Track unique bigrams
        for _, row in self.train_df.iterrows():
            _, label, doc = row  # Extract sentiment label and processed review
            doc_words = doc.split()
            for i, word in enumerate(doc_words):
                # Update unigram counts based on class label
                if label == 1:
                    self.uni_pos_counts[word] += 1
                    self.uni_num_pos += 1
                elif label == 0:
                    self.uni_neg_counts[word] += 1
                    self.uni_num_neg += 1
                unique_uni.add(word)

                # Update bigram counts based on class label
                if i < len(doc_words) - 1:
                    bigram = (doc_words[i], doc_words[i + 1])
                    if label == 1:
                        self.bi_pos_counts[bigram] += 1
                        self.bi_num_pos += 1
                    elif label == 0:
                        self.bi_neg_counts[bigram] += 1
                        self.bi_num_neg += 1
                    unique_bi.add(bigram)
        self.num_uni = len(unique_uni)  # Total unique unigrams
        self.num_bi = len(unique_bi)  # Total unique bigrams

    # Naive Bayes classification using unigrams and bigrams
    def naive_bayes(self, hold_text, u_laplace, b_laplace, mixture_lambda):
        text = hold_text.split()
        pos_prob = 0  # Log probability for the positive class
        neg_prob = 0  # Log probability for the negative class
        for i, word in enumerate(text):
            # Calculate unigram probabilities with Laplace smoothing
            curr_uni_pos = (self.uni_pos_counts[word] + u_laplace) / (self.uni_num_pos + u_laplace * self.num_uni)
            curr_uni_neg = (self.uni_neg_counts[word] + u_laplace) / (self.uni_num_neg + u_laplace * self.num_uni)

            # Calculate bigram probabilities if applicable
            curr_bi_pos = 1
            curr_bi_neg = 1
            if i < len(text) - 1:
                bigram = (text[i], text[i + 1])
                curr_bi_pos = (self.bi_pos_counts[bigram] + b_laplace) / (self.bi_num_pos + b_laplace * self.num_bi)
                curr_bi_neg = (self.bi_neg_counts[bigram] + b_laplace) / (self.bi_num_neg + b_laplace * self.num_bi)

            # Combine unigram and bigram probabilities using a mixture weight
            pos_prob += (1 - mixture_lambda) * math.log(curr_uni_pos) + mixture_lambda * math.log(curr_bi_pos)
            neg_prob += (1 - mixture_lambda) * math.log(curr_uni_neg) + mixture_lambda * math.log(curr_bi_neg)

        # Return the class with the higher probability
        return 1 if pos_prob > neg_prob else 0

    # Apply Naive Bayes classification to the test dataset
    def naive_bayes_test_df(self, u_laplace=1.0, b_laplace=1.0, mixture_lambda=0.4):
        self.test_df['predictions'] = self.test_df['processed_review'].apply(
            lambda text: self.naive_bayes(text, u_laplace, b_laplace, mixture_lambda))

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
    test = BigramNaiveBayes()  # Instantiate the model
    test.train_naive_bayes()  # Train the model
    test.naive_bayes_test_df(1, 0.5, 0.6)  # Test the model with specific parameters
    test.print_statistics()  # Print evaluation metrics