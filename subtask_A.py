import math
import sklearn
import csv
from collections import defaultdict
import sys
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

def tokenize(text, version=''):
    tokens =[]
    if version == 'SE':
        tokens = ['<'+token+'>' for token in re.findall(r'[@#]?\b\w+\b[!?]?', text.casefold(), flags=re.UNICODE)] #gives us start and end info
    else:
        tokens = re.findall(r'[@#]?\b\w+\b[!?]?', text.casefold(), flags=re.UNICODE) #keeps words, @ and # at the start of word, and ?! at end
    print(tokens)
    return tokens

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
        
        ngram_size = 3

        # count_vectorizer = CountVectorizer(analyzer=tokenize,ngram_range=(1,ngram_size))
        count_vectorizer = CountVectorizer(ngram_range=(2,4)) #ngram_range=(2,4) seems to be best


        train_counts = count_vectorizer.fit_transform(train_text)
        test_counts = count_vectorizer.transform(test_text)



        # NB_model = make_pipeline(CountVectorizer(ngram_range=(1,ngram_size)), MultinomialNB())

        NB_model = MultinomialNB()

        NB_model.fit(train_counts, train_klasses)



        results = NB_model.predict(test_counts)

    elif method == "baseline":
        classifier = Baseline(train_labels)
        results = [classifier.classify(x) for x in test_text]

    elif method == "lr":
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        train_counts = count_vectorizer.fit_transform(train_docs)

        print(train_counts.shape)
        print(len(train_labels))


        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_labels)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        test_counts = count_vectorizer.transform(test_text)
        # Predict the class for each test document
        results = clf.predict(test_counts)


# for x in results:
#         print(x)

print(metrics.classification_report(test_klasses, results,zero_division=1))