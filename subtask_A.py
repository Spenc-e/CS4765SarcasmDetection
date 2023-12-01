import math
import sklearn
import csv
from collections import defaultdict
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

# The regular expression looks for words that possibly start with '@' 
# or '#' and possibly end with '!' or '?'. The '@' or '#' and the '!' 
# or '?' are optional. It captures words, mentions, or hashtags in a text.
# this information may prove useful when analyzing tweets.
custom_token_pattern = r'(?u)[@#]?\b\w+\b[!?]?'

def print_safe(text):
    try:
        print(text)
    except UnicodeEncodeError:
        # For encoding error, try explicitly encoding to utf-8
        print(text.encode('utf-8'))

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

    # Method will be one of 'baseline', 'lr', 'nb'
    method = sys.argv[1]
    train_data = sys.argv[2]
    test_data = sys.argv[3]

    # Reading the training data
    with open(file=train_data, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # Skipping header text
        header = next(csv_reader)
        train_text = []
        train_klasses = []

        for row in csv_reader:
            train_text.append(row[1].lower())
            train_klasses.append(row[2])

    # Reading the test data
    with open(file=test_data, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # Skipping header text
        header = next(csv_reader)
        test_text = []
        test_klasses = []

        for row in csv_reader:
            test_text.append(row[0].lower())
            test_klasses.append(row[1])

    # Splitting the train data into dev data. Specifying random state to allow reproducibility
    train_docs, dev_docs,train_labels,dev_labels = train_test_split(train_text, train_klasses, test_size=0.2, random_state=4)

    if method == "nb":

        # Allows you to set the ngram size. (3,3) is trigram
        # (1,3) is unigram to trigram. If you want to individually test
        # different ngram sizes, you must set the range to be the same
        # number on either side of the comma. e.g. (2,2) = bigram 
        ngram_custom_range = (2,4)
        
        # tokenizing text, using a custom regular expression defined above using ngram classification.
        count_vectorizer = CountVectorizer(token_pattern=custom_token_pattern,ngram_range=ngram_custom_range)

        #Each cell in the matrix indicates the frequency of a type in the documents.
        train_counts = count_vectorizer.fit_transform(train_text)
        test_counts = count_vectorizer.transform(test_text)


        NB_model = MultinomialNB()
        NB_model.fit(train_counts, train_klasses)
        results = NB_model.predict(test_counts)

    elif method == "baseline":
        classifier = Baseline(train_labels)
        results = [classifier.classify(x) for x in test_text]

    elif method == "lr":
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # tokenizing text, using a custom regular expression defined above
        count_vectorizer = CountVectorizer(token_pattern=custom_token_pattern)

        #Each cell in the matrix indicates the frequency of a type in the documents.
        train_counts = count_vectorizer.fit_transform(train_text)
        test_counts = count_vectorizer.transform(test_text)


        # Training a logistic regression classifier on the train data.
        # The random state is set to an arbitrary number to allow
        # reproducibility in the results

        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)

        # Predict the class for each test document
        results = clf.predict(test_counts)


# Calculate precision, recall, and f1-score using scikit-learn. 
precision, recall, f1, _ = precision_recall_fscore_support(test_klasses, results, zero_division=1,
                                                           average='weighted', labels=['1'])

# Print the results in a comma-delimited format. This allows you to easily read
# the outputted data and then use it for data visualization in another program
print_safe(f'Precision,Recall,F1-Score')
print_safe(f'{precision:.3f},{recall:.3f},{f1:.3f}')

# Printing the precision, recall, f1-score, macro avg, weight avg, and accuracy
# for a more user friendly viewing in the terminal
print(metrics.classification_report(test_klasses, results,zero_division=1))

