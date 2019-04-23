import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", size=14)
sns.set(style="ticks", color_codes=True)
sns.set(style="white")
import matplotlib.pyplot as plt
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from quilt.data.ResidentMario import missingno_data
import missingno as msno
import pandas_profiling
import itertools


def plot_confusion_matrix(y_true, y_predict, cm, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. Normalization can be
    applied by setting `normalize=True`. Code taken from example in  SKLearn
    documentation.

    Parameters:
    y_true, y_predict (1-d Numpy arrays): Target data from the test dataset and
    predicted y values from a model.
    cm (confusion matrix as 2-d Numpy array, computed in model function)

    Returns:
    No objects returned, but the image is saved to a PNG file.
    """
    plt.figure()
    classes = ['functional', 'needs repairs', 'broken']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('../images/confusion_matrix.png')


def feat_imps_barplot(df_sorted_feats, cols):
    """
    This function plots the feature importances calculated from a Random Forest
    model as a bar plot (bars on left axis), ordered by feature importance.

    Parameters:
    df_sorted_feats (Pandas dataframe): contains the features and their
    associated "importances" for the models predictive power. The dataframe is
    ordered from most to least important.

    Returns:
    Does not return any objects but the image is saved to a PNG file.
    """
    y = df_sorted_feats['importances']
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)
    width = 0.75 # the width of the bars
    idx = np.arange(len(y)) # the x locations for the groups
    ax.barh(idx, y, width, color='k')
    ax.set_yticks(idx+width/10)
    ax.set_xticks(np.arange(0, 1, step=0.1))
    ax.set_yticklabels(df_sorted_feats['cols'].values, minor=False, fontsize=15)
    plt.title('RandomForest Classifier Feature Importance', fontsize=25)
    plt.xlabel('Relative importance', fontsize=20)
    plt.ylabel('Feature', fontsize=20)
    fig.tight_layout()
    plt.show()
    plt.savefig('../images/feature_importances.png')
