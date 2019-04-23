import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing, linear_model, metrics
import sklearn.linear_model as lm
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
import featurize as cleaned
import visualization as viz
import random_forest as rf
import log_regression as mnlr
import naive_bayes as nb


def split(df):
    """
    This function splits the cleaned and dummied dataframe into a stratified
    training and test set.

    Parameters:
    Pandas dataframe.

    Returns:
    Pandas dataframes of X and y for both training and test sets.
    """
    dataframe = df.copy()
    y = dataframe.pop('status_group')
    X = dataframe
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=21)

def upsample(X, y):
    """
    This function upsamples training data (X and y) using the SMOTE method.

    Parameters:
    Pandas dataframes - training X and y.

    Returns:
    Numpy arrays of X and y, with upsampled data.
    """
    sm = SMOTE(sampling_strategy='minority', random_state=32, k_neighbors=4)
    X_smote, y_smote = sm.fit_sample(X, y)
    return X_smote, y_smote

def kfold_train_score(model, X, y):
    """
    This function does the cross valdiation and scoring on training data for
    all models.

    Parameters:
    model: fit model from one of the three model scripts.
    X, y: SMOTE upsampled Numpy arrays for training.

    Returns:
    Numpy arrays of the average F1, recall, and precision scores over the 4
    cross validation splits for the model sent to this function.
    """
    k_fold = KFold(n_splits=4, shuffle=True, random_state=21)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for train_idx, val_idx in k_fold.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_predict = model.predict(X[val_idx])
        '''
        # The F1 score function was not working properly. Although average is
        # set to 'micro', it throws an error saying average cannot be 'binary':
        # f1_scores.append(f1_score(y[val_idx], y_predict), labels= ['functional', 'needs repairs', 'broken'], average='micro')
        '''
        recall_scores.append(recall_score(y[val_idx], y_predict, average='micro'))
        precision_scores.append(precision_score(y[val_idx], y_predict, average='micro'))
    return np.average(f1_scores), np.average(recall_scores), np.average(precision_scores)

def feat_imps(model, X_train, y_train):
    """
    This function calculates the feature importances of the model sent to it
    and sorts them from highest to lowest importances in a dataframe.

    Parameters:
    model: fit model from one of the three model scripts.
    X_train, y_train: Training X and y in Pandas Dataframe objects.

    Returns:
    A 2-column Pandas dataframe with the dataset features and their importances
    (numeric values), sorted by importance.
    """
    model.fit(X_train, y_train)
    feat_importances = model.feature_importances_
    df_sorted_feats = pd.DataFrame()
    df_sorted_feats['cols'] = X_train.columns
    df_sorted_feats['importances'] = feat_importances
    df_sorted_feats = df_sorted_feats.sort_values(by=['importances'])
    return df_sorted_feats


if __name__ == "__main__":
    # Create lists:
    dont_dummy = ['public_meeting', 'permit', 'status_group']
    one_hot_features = ['amount_tsh_zero_tsh', 'basin_Lake Victoria', 'district_code_district 1', 'extraction_type_class_gravity', 'management_group_user-group', 'payment_never pay', 'population_unknown', 'region_Dar es Salaam','quantity_enough', 'source_class_groundwater', 'waterpoint_type_group_communal standpipe', 'waterpoint_age_26_years']
    include = ['quantity_seasonal', 'waterpoint_age_over_26_years', 'waterpoint_age_11_to_25_years', 'payment_pay monthly', 'payment_other', 'extraction_type_class_submersible', 'population_101_to_200', 'population_201_to_380', 'population_0_to_100s', 'public_meeting', 'population_over_380', 'waterpoint_age_10_years_or_less', 'district_code_district 3', 'district_code_district 2', 'source_class_surface', 'permit', 'extraction_type_class_other', 'waterpoint_type_group_other', 'quantity_insufficient', 'quantity_dry']
    gis = ['gps_height', 'longitude', 'latitude']

    # Read in and clean the data; write a csv file for separate GIS analysis:
    data = pd.read_csv('../data/water_tz_data.csv')
    df_gis = cleaned.clean(data)
    # df_gis.to_csv('../data/water_data_gis.csv')
    # df_gis = pd.read_csv('../data/water_data_gis.csv')
    '''
    # Cleaning is time consuming, so this data can be used to run script multiple times
    df_gis = pd.read_csv('../data/water_data_gis.csv')
    # drop superfluous index column created from to_csv function:
    df_gis.drop(df_gis.columns[0], axis=1, inplace=True) #
    '''

    # Convert string target classes to numeric categorical labels:
    classes = df_gis.status_group.value_counts().values
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    df_gis['status_group'] = le.fit_transform(df_gis['status_group'])

    # Create dataframe without GPS coordinate columns:
    df = df_gis.copy()
    df.drop(gis, axis=1, inplace=True)

    # eda:
    '''
    # These do not seem to be working in the terminal, though they work in
    Jupyter Notebook.
    cleaned.dataframe_profile(data)
    cleaned.dataframe_profile(df)
    '''

    # Dummy, feature eliminate, split, and upsample the data for analysis:
    dummied = cleaned.dummify(df, dont_dummy, one_hot_features)
    dummied_elim = cleaned.feat_elim(dummied, include)
    X_train, X_holdout, y_train, y_holdout = split(dummied_elim)
    cols = X_train.columns
    X_smote, y_smote = upsample(X_train, y_train)

    # Run models:
    estimator, forest = rf.select_rand_forest(X_smote, y_smote)
    forest = rf.rand_forest(X_smote, y_smote, X_holdout, y_holdout, forest, estimator, classes)
    logreg = mnlr.log_regress(X_smote, y_smote, X_holdout, y_holdout, cols)
    bernoulli_nb = nb.naive_bayes(X_smote, y_smote, X_holdout, y_holdout)

    # Plot results:
    df_sorted_feats = feat_imps(forest, X_train, y_train)
    viz.feat_imps_barplot(df_sorted_feats, X_train.columns)
