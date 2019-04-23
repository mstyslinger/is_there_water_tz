import pandas as pd
import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas.plotting import autocorrelation_plot
from quilt.data.ResidentMario import missingno_data
import missingno as msno
import pandas_profiling


def clean(data):
    """
    Clens the dataset, dealing with missing vaues, converting strings to
    numerical values. Also does feature engneering and selection, converting
    years to ages, grouping continuous values into categorical buckets,
    clatifying with amiguous values, and dropping features that will be excluded
    from analysis.

    Parameters:
    data (Pandas Dataframe): Raw data set converted from CSV to Pandas DF

    Returns:
    Dataframe. cleaned and featurized for analysis. It includes GIS features for
    mapping purposes, which are subsequently dropped before model fitting.
    """
    df = data.copy()
    target = df.pop('status_group')
    # convert continuous water pressure at tap feature to categorical ranges
    df['amount_tsh'].replace(df['amount_tsh'].where(df['amount_tsh'] == 0), 0, inplace=True)
    df['amount_tsh'].replace(df['amount_tsh'].where((df['amount_tsh'] > 0) & (df['amount_tsh'] < 51)), 50, inplace=True)
    df['amount_tsh'].replace(df['amount_tsh'].where((df['amount_tsh'] > 50) & (df['amount_tsh'] < 251)), 250, inplace=True)
    df['amount_tsh'].replace(df['amount_tsh'].where((df['amount_tsh'] > 250) & (df['amount_tsh'] < 1001)), 1000, inplace=True)
    df['amount_tsh'].replace(df['amount_tsh'].where(df['amount_tsh'] > 1000), 1001, inplace=True)
    tsh_dict = {0: 'zero_tsh', 50: '1_to_50_tsh', 250: '51_to_250_tsh', 1000: '251_to_1000_tsh', 1001: 'over_1000_tsh'}
    df['amount_tsh'].replace(tsh_dict, inplace=True)
    # convert string dates to datetime values:
    for idx, date in enumerate(df['date_recorded']):
        yy = date[-2:]
        year = int(yy)
        if year > 19:
            df['date_recorded'][idx] = 1900 + int(date[-2:])
        elif year < 19:
            df['date_recorded'][idx] = 2000 + int(date[-2:])
    df['waterpoint_age'] = df['date_recorded']-df['construction_year'] # calculate water point age feature
    # impute values that had zero for construction_year with median age (26). After imputation, 36% of values are 26:
    df['waterpoint_age'].replace(df['waterpoint_age'].where(df['waterpoint_age'] > 1000), 26, inplace=True)
    # convert continuous 'waterpoint_age' to categorical ranges:
    df['waterpoint_age'].replace(df['waterpoint_age'].where(df['waterpoint_age'] <= 10), 10, inplace=True)
    # twenty_five = (df['waterpoint_age'] > 10) and (df['waterpoint_age'] < 26)
    df['waterpoint_age'].replace(df['waterpoint_age'].where((df['waterpoint_age'] > 10) & (df['waterpoint_age'] < 26)), 25, inplace=True)
    df['waterpoint_age'].replace(df['waterpoint_age'].where(df['waterpoint_age'] == 26), 26, inplace=True)
    df['waterpoint_age'].replace(df['waterpoint_age'].where(df['waterpoint_age'] > 26), 27, inplace=True)
    age_dict = {10: '10_years_or_less', 25: '11_to_25_years', 26: '26_years', 27: 'over_26_years'}
    df['waterpoint_age'].replace(age_dict, inplace=True)
    codes = df['district_code'].value_counts()
    for dist in codes.index: # Turn integer values into categorical districts
        df['district_code'].replace(dist, "district {}".format(dist), inplace=True)
    # convert continuous population to categorical ranges:
    df['population'].replace(df['population'].where(df['population'] <= 1), 1, inplace=True) # 0s and 1s (probably inaccurate) to unknown category - 48% of the data
    df['population'].replace(df['population'].where((df['population'] > 1) & (df['population'] <= 100)), 100, inplace=True)
    df['population'].replace(df['population'].where((df['population'] > 100) & (df['population'] <= 200)), 200, inplace=True)
    df['population'].replace(df['population'].where((df['population'] > 200) & (df['population'] <= 380)), 380, inplace=True)
    df['population'].replace(df['population'].where(df['population'] > 380), 381, inplace=True)
    pop_dict = {1: 'unknown', 100: '0_to_100s', 200: '101_to_200', 380: '201_to_380', 381: 'over_380'}
    df['population'].replace(pop_dict, inplace=True)
    df['permit'].fillna('False', inplace=True) # replace NaNs with 'False'
    bools = {'True': True, 'False': False}
    df['permit'].replace(bools, inplace=True) # Turn string True/False values into boolean
    df['public_meeting'].fillna('False', inplace=True) # replace NaNs with 'False'
    df['public_meeting'].replace(bools, inplace=True) # convert strings to boolean
    df['payment'].replace("unknown", "other", inplace=True) # join 'unknown' and 'other' under one value
    df['management_group'].replace("unknown", "other", inplace=True)
    df['management_group'].replace("parastatal", "state-owned entity", inplace=True) # rename value for clarity
    drop_lst = ['id','lga','funder', 'installer', 'wpt_name', 'subvillage', 'ward', 'recorded_by', 'scheme_management', 'scheme_name', 'construction_year', 'date_recorded', 'num_private', 'quality_group', 'quantity_group', 'region_code', 'waterpoint_type', 'source', 'source_type', 'water_quality', 'payment_type', 'extraction_type_group', 'extraction_type', 'management']
    df.drop(drop_lst, axis=1, inplace=True) # drop features from analysis
    df['status_group']=target # move target column to the end
    return df

def dataframe_profile(df):
    """
    Outputs a missing value matrix and Pandas Profile Report for EDA purposes

    Parameters(dataframe):
    Can be raw or cleaned Pandas Dataframe

    Returns:
    MSNO matrix and Pandas profile objects
    """
    msno_matrix = msno.matrix(df.sample(5000))
    pd_profile = pandas_profiling.ProfileReport(df)
    return msno_matrix, pd_profile

def dummify(df, dont_dummy, one_hot_features):
    cols = [col for col in df.columns if col not in dont_dummy]
    dummied = pd.get_dummies(data=df, columns=cols)
    drop_cols = one_hot_features
    dummied.drop(drop_cols, axis=1, inplace=True)
    return dummied

def feat_elim(dummied, include):
    """
    This function eliminates features by excluding any columns that are not in
    the 'include' list from the data frame for analysis. This can be a choice
    made after genrating feature importances.

    Parameters:
    dummy: dummied dataframe before spliting into train-test datasets.
    include: list of features to include in the resulting dataframe.

    Returns:
    A dummied dataframe with features culled.
    """
    cols = [col for col in dummied.columns if col in include] + ['status_group']
    dummied_elim = dummied[cols]
    return dummied_elim
