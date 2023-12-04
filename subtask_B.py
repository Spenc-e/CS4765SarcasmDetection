import csv
import math
from collections import defaultdict
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Regular expression that matches words that might start with @ or #
# followed by a character, that might end in ! or ?
custom_token_pattern = r'(?u)[@#]?\b\w+\b[!?]?'

custom_target_names = ['Sarcasm','Irony','Satire','Understatement','Overstatement','Rhetorical Question']

# Reads data from a csv file
def prepareData(file_name):
    text = []
    klasses = []

    with open(file=file_name, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # Skipping header text
        header = next(csv_reader)

        for row in csv_reader:
            text.append(row[0].lower())
            klasses.append([int(label) for label in row[1:7]])
    return text, klasses



# Counts and prints the occurrences of classes in a dataset
def countKlasses(file_name, klasses):
    # Store the counts for each class
        class_counts = {class_name: 0 for class_name in custom_target_names}

        for labels in klasses:
            # Increment the count for each class that is labeled (has a value of 1)
            for i, label in enumerate(labels):
                if label == 1:
                    class_counts[custom_target_names[i]] += 1

        # Print the counts for each class
        print("\nThe number of occurances for each class in \'"+file_name+"\'")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} occurrences")

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True,
                          key=lambda x : klass_freqs[x])[0]
        print(klass_freqs)

    def classify(self, test_instance):
        return self.mfc

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'nb', or 'nbse'
    method = sys.argv[1]
    train_data = sys.argv[2]
    test_data = sys.argv[3]

    # Initializing data
    train_text, train_klasses = prepareData(train_data)
    test_text, test_klasses = prepareData(test_data)

    # Splitting the train data into dev data. Specifying random state to allow reproducibility
    train_docs, dev_docs,train_labels,dev_labels = train_test_split(train_text, train_klasses, test_size=0.2, random_state=4)


    if method == "nb":
        
        count_vectorizer = CountVectorizer(token_pattern=custom_token_pattern) #ngram_range=(2,4) seems to be best
        
        train_counts = count_vectorizer.fit_transform(train_text)
        test_counts = count_vectorizer.transform(test_text)

        # Initialize a dictionary to store the counts for each class
        class_counts = {class_name: 0 for class_name in custom_target_names}

        countKlasses(train_data,train_klasses)
        countKlasses(test_data,test_klasses)

        # Using the binary relevance method, so creating a NB model for each class
        nb_classifier = MultinomialNB()
        classifier = MultiOutputClassifier(nb_classifier)

        classifier.fit(train_counts, train_klasses)

        predicted_klasses = classifier.predict(test_counts)

        evaluation_report = metrics.classification_report(test_klasses, predicted_klasses, target_names=custom_target_names,zero_division=0)

        print('Classification Report')
        print(evaluation_report)

        # results = NB_model.predict(test_counts)

    elif method == "baseline":
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_text]

    elif method == "lr":

        count_vectorizer = CountVectorizer(token_pattern=custom_token_pattern) #ngram_range=(2,4) seems to be best
        
        train_counts = count_vectorizer.fit_transform(train_text)
        test_counts = count_vectorizer.transform(test_text)

        # Create a Logistic Regression model
        logreg_classifier = LogisticRegression()

        # Use MultiOutputClassifier for multi-label classification
        classifier = MultiOutputClassifier(logreg_classifier)

        # Fit the model to the training data
        classifier.fit(train_counts, train_klasses)

        # Make predictions on the test data
        predicted_labels = classifier.predict(test_counts)

        # zero_division='warn' will issue a warning when there is a class with no predictions
        print(metrics.classification_report(test_klasses, predicted_labels,target_names=custom_target_names,zero_division=0))
