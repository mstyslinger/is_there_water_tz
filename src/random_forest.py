import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing, linear_model, metrics
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix
from skater.model import InMemoryModel
import main
import visualization as viz


def select_rand_forest(X_train, y_train):
    """
    This function is built to run through as many estimators (Random Forest
    hyperparamter) as are included in the list 'estimators,' do cross validation
    and scoring with each, and send back the estimator that the model performs
    best with. The Random Forest Classifier model is instantiated with
    'balanced' class weights.

    Parameters:
    X_train, y_train (Numpy arrays): used to train the model. The objects are
    sent to a KFold cross validation function in the "main.py" script, where
    each fold is scored to get averaged training scores (F1, recall, and
    precision).

    Returns:
    A best estimator object (int) and an instantiated Random Forest Classifier
    model object.

    """
    estimators = [50]
    best_score = 0.0001
    for i in estimators:
        model = RandomForestClassifier(n_estimators=i, class_weight='balanced')
        f1, recall, precision = main.kfold_train_score(model, X_train, y_train)
        print("Random Forest training scores with", i, "estimators: ")
        print("* F1 - ", round(f1, 3))
        print("* Recall - ", round(recall, 3))
        print("* Precision - ", round(precision, 3))
        print(" ")
        if precision > best_score:
            best_score = precision
            best_estimator = i
    return best_estimator, model


def rand_forest(X_train, y_train, X_holdout, y_true, model, estimator, classes):
    """
    After the best estimator is determined, this function fits the Random Forest
    model and scores its prediction on holdout test data. Scores are printed.

    Parameters:
    X_train, y_train, X_holdout, y_true, (Numpy arrays): used to train the
    model and score it on the test data.
    model:  Random Forest classifier instantiated in the select_rand_forest function.

    Returns:
    The fit model.
    """
    forest = model.fit(X_train, y_train)
    y_predict = forest.predict(X_holdout)
    cm = confusion_matrix(y_true, y_predict)
    viz.plot_confusion_matrix(y_true, y_predict, cm)
    recall = recall_score(y_true, y_predict, average='micro')
    print("Final scores: ")
    print(" ")
    print("The test recall score is ", recall)
    print(" ")
    print(classification_report(y_true, y_predict))
    print(" ")
    print("####################################################")
    print(" ")
    return forest
