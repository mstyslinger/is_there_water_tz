import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn import preprocessing, metrics
import sklearn.linear_model as lm
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score, classification_report, confusion_matrix
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import BernoulliNB
import main


def naive_bayes(X_train, y_train, X_test, y_true):
    """
    This does cross validation scoring (calling the main.kfold_train_score()
    function) with the train data on a Bernoulli Naive Bayes classification
    model, prints the training scores, fits and predicts with test data, and
    then prints final model scores.

    Parameters:
    X_train, y_train, X_holdout, y_true, (Numpy arrays): used to train the
    model and score it on the train and test data.

    Returns:
    The fit model.
    """
    model = BernoulliNB()
    f1, recall, precision = main.kfold_train_score(model, X_train, y_train)
    print("* Naive Bayes training scores: ")
    print("* F1 - ", round(f1, 3))
    print("* Recall - ", round(recall, 3))
    print("* Precision - ", round(precision, 3))
    print(" ")
    bernoulli_nb = model.fit(X_train, y_train)
    y_predict = bernoulli_nb.predict(X_test)
    recall = recall_score(y_true, y_predict, average='micro')
    print("Final scores: ")
    print(" ")
    print("The test recall score is ", recall)
    print(" ")
    print(classification_report(y_true, y_predict))
    print(" ")
    print("####################################################")
    print(" ")
    return bernoulli_nb
