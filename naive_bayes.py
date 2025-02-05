# Import necessary libraries
import pandas as pd  # For data manipulation
import re  # For text preprocessing using regex
import nltk  # For NLP tools like stopwords and lemmatization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict  # For default dictionaries to count word frequencies
from sklearn.model_selection import train_test_split  # To split dataset into training and testing
import math  # For mathematical operations like logarithms
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score  # For model evaluation

# Path to the dataset file
filepath_dataset = "dataset.csv"

class NaiveBayes:
    def __init__(self):
        # Helper function to convert sentiment labels to integers
        def text_to_int(text):
            return 1 if text == "positive" else 0

        # Download required resources for stopwords and lemmatization
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        # Initialize preprocessing tools
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Load dataset and preprocess reviews
        self.df = pd.read_csv(filepath_dataset)
        self.df['processed_review'] = self.df['review'].apply(self.preprocess)  # Clean text
        self.df['sentiment'] = self.df['sentiment'].apply(text_to_int)  # Convert sentiments to integers

        # Split data into training and testing sets
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        # Initialize word counts and totals for Naive Bayes
        self.pos_word_counts = defaultdict(int)  # Word frequencies for positive reviews
        self.neg_word_counts = defaultdict(int)  # Word frequencies for negative reviews
        self.num_words = 0  # Total unique words
        self.num_neg = 0  # Total words in negative reviews
        self.num_pos = 0  # Total words in positive reviews

    # Clean and preprocess text (lowercase, remove unwanted characters, lemmatize, etc.)
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
        text = re.sub(r"<.*?>", '', text)  # Remove HTML tags
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Remove special characters
        text = re.sub(r"\d+", '', text)  # Remove digits
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Remove stopwords
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        return ' '.join(words)  # Return processed text

    # Train the Naive Bayes model by calculating word frequencies for each sentiment class
    def train_naive_bayes(self):
        unique_words = set()  # To track unique words in the training data
        for _, row in self.train_df.iterrows():
            _, label, doc = row  # Extract label and processed review
            for word in doc.split():
                if label == 1:
                    self.pos_word_counts[word] += 1
                    self.num_pos += 1
                elif label == 0:
                    self.neg_word_counts[word] += 1
                    self.num_neg += 1
                unique_words.add(word)
        self.num_words = len(unique_words)  # Total number of unique words

    # Perform Naive Bayes classification on a single text
    def naive_bayes(self, text, laplace=1.0):
        text = text.split()  # Tokenize the text
        log_pos_prob = 0  # Log probability for positive class
        log_neg_prob = 0  # Log probability for negative class
        for word in text:
            # Calculate probabilities with Laplace smoothing
            curr_pos = (self.pos_word_counts[word] + laplace) / (self.num_pos + laplace * self.num_words)
            curr_neg = (self.neg_word_counts[word] + laplace) / (self.num_neg + laplace * self.num_words)
            log_pos_prob += math.log(curr_pos)
            log_neg_prob += math.log(curr_neg)
        # Return the class with higher probability
        return 1 if log_pos_prob > log_neg_prob else 0

    # Apply Naive Bayes to the test dataset
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
        return accuracy, precision, recall  # Return the metrics

# Main script
if __name__ == "__main__":
    test = NaiveBayes()  # Create Naive Bayes model instance
    test.train_naive_bayes()  # Train the model on the training set
    test.naive_bayes_test_df()  # Test the model on the test set
    test.print_statistics()  # Print and return evaluation metrics