############################################################################################################################

########################################
## Author: Carlos Henrique Fantecelle ##
########################################

# This module contains useful functions
# I have built or gathered along. When
# the latter is the case, I indicated
# the author in the description of said
# function.

##################
## Dependencies ##                                                                                                          
##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

############################################################################################################################

def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, 
    millions and thousands) such as 4500 into 
    4.5K and also appropriately turns 4000 
    into 4K (no zero after the decimal).

    Author: https://dfrieds.com/
    """

    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format


def uniqueValuesPerColumn(df):
    """
    Takes a dataframe and returns unique values in each
    categorical column of the dataset.
    """
    
    variables = df.select_dtypes('object').columns.to_list()

    unique_values = []

    for variable in variables:
        unique_values.append(', '.join(df[variable].unique()))
    
    unique_df = pd.DataFrame({'Variable': variables,
                              'Unique values': unique_values})
    
    return(unique_df)


def classifyColumns(df):
    """
    Takes a dataframe and returns which columns
    are numeric, binary categories or multiple
    categories, in this order.
    """

    num_cols = df.select_dtypes(['int64', 'float64', 'int32', 'float32']).columns.to_list()
    cat_cols = df.select_dtypes('object').columns.to_list()
    bcat_cols = []
    mcat_cols = []

    for col in cat_cols:
        if len(df[col].unique()) == 2:
            bcat_cols.append(col)
        elif len(df[col].unique()) > 2:
            mcat_cols.append(col)

    return num_cols, bcat_cols, mcat_cols


def crossValClassModels(X, y, scaler, metric):
    """
    Makes cross validation data using multiple
    classification models for a baseline.

    Takes:
        X = Dataframe of independend variables.
        y = Series of the target variable.
        scaler = Scaling method used to 
                 transform/normalize the data
        metric = Metric chosen to evaluate models.

    Returns a dataframe with 'metric' results for
    each predefined model.
    """

    # List to build results df later
    model_name = []
    metric_result = []

    # Converting to arrays
    X = np.array(X)
    y = np.array(y)

    # Initiating models
    lg = LogisticRegression()
    dtc = DecisionTreeClassifier() 
    rf = RandomForestClassifier()
    svm = SVC() # Support Vector Machines
    sgd = SGDClassifier() # Stochastic Gradient Descent
    lgbm = LGBMClassifier() # LightGBM
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss") #XGBoost, parameters to supress warnings

    # Creating list of results
    models = [lg, dtc, rf, svm, sgd, lgbm, xgb]

    # Looping and evaluating each model
    for model in models:
        pipeline = make_pipeline(scaler, model)
        scores = cross_val_score(pipeline, X, y, scoring = metric)
        model_name.append(model.__class__.__name__)
        metric_result.append("{:.4f} (+/- {:.4f})".format(scores.mean(), scores.std()))


    metric_name = metric.capitalize()
    results_df = pd.DataFrame({'Model': model_name,
                                metric_name: metric_result})

    return(results_df)


def calcChurnKPIs(y_test, y_pred, X_test, value_column):
    """
    Function to calculate Churn Rate and Gross MMR Churn Rate.
    Takes as arguments the y_test and y_pred from final model,
    the X_test (w/ column names! be careful when using scaler!)
    and value_column as the name of column with MRR variable.
    """

    # Creating dict
    predictions = {'Index': y_test.index.values,
                   'Amount': X_test[value_column],
                   'Class': y_test.to_numpy(),
                   'Model Prediction': y_pred,
                   'Catch': ""
                   }

    # Converting to df
    predictions_df = pd.DataFrame(predictions).reset_index()

    # Looping to identify churners
    for i in range(predictions_df.shape[0]):
        if predictions_df.loc[i, 'Class'] == 1:
            if predictions_df.loc[i, 'Class'] == predictions_df.loc[i, 'Model Prediction']:
                predictions_df.loc[i,'Catch'] = 'Detected Churn'
            else:
                predictions_df.loc[i,'Catch'] = 'Undetected Churn'
        elif predictions_df.loc[i, 'Class'] == 0 and predictions_df.loc[i, 'Model Prediction'] == 1:
            predictions_df.loc[i,'Catch'] = 'Detected, not Churn'
        else:
            predictions_df.loc[i,'Catch'] = 'Not churn'

    # Extracting information
    n_customer = len(y_test)
    n_churn = predictions_df.Catch.value_counts()['Detected Churn'] + predictions_df.Catch.value_counts()['Undetected Churn']
    n_churn_detected = predictions_df.Catch.value_counts()['Detected Churn']
    total_mrr = predictions_df.Amount.sum()
    churn_mrr = predictions_df.loc[predictions_df['Catch'] == 'Detected Churn'].Amount.sum() + predictions_df.loc[predictions_df['Catch'] == 'Undetected Churn'].Amount.sum()
    churn_mrr_detected = predictions_df.loc[predictions_df['Catch'] == 'Detected Churn'].Amount.sum()

    # Creating kpi dictionary
    kpis_dict = {'KPI': ['Churn Rate', 'Gross MRR Churn Rate', 'Gross MRR Loss'],
                 'Real': ["{:.2%}".format(n_churn/n_customer),
                          "{:.2%}".format(churn_mrr/total_mrr),
                          "${:,.2f}".format(churn_mrr)],
                 'Detected': ["{:.2%}".format(n_churn_detected/n_customer),
                              "{:.2%}".format(churn_mrr_detected/total_mrr),
                              "${:,.2f}".format(churn_mrr_detected)],
                 'Difference': ["{:.2%}".format((n_churn/n_customer)-(n_churn_detected/n_customer)),
                                "{:.2%}".format((churn_mrr/total_mrr)-(churn_mrr_detected/total_mrr)),
                                "${:,.2f}".format(churn_mrr-churn_mrr_detected)]}

    # Converting to dataframe
    kpis_df = pd.DataFrame(kpis_dict)

    print("Gross MRR for the period analysed: ${:,.2f}".format(total_mrr))
    return(kpis_df)

# Creating function to calculate classification results
def calcClassResults(results_df, target_col, pred_col):
    """
    Function to calculate classification rates
    after prediction using a machine learning
    model.
    """

    # Creating dict
    predictions = {'Index': results_df.index.values,
                   'Class': results_df[target_col],
                   'Model Prediction': results_df[pred_col],
                   'Catch': ""
                   }

    # Converting to df
    predictions_df = pd.DataFrame(predictions).reset_index()

    # Looping to identify churners
    for i in range(predictions_df.shape[0]):
        if predictions_df.loc[i, 'Class'] != 0:
            if predictions_df.loc[i, 'Class'] == predictions_df.loc[i, 'Model Prediction']:
                predictions_df.loc[i,'Catch'] = 'True Positive'
            elif predictions_df.loc[i, 'Model Prediction'] == 0:
                predictions_df.loc[i,'Catch'] = 'False Negative'
            else:
                predictions_df.loc[i,'Catch'] = 'Wrong Positive'
        elif predictions_df.loc[i, 'Class'] == 0 and predictions_df.loc[i, 'Model Prediction'] != 0:
            predictions_df.loc[i,'Catch'] = 'False Positive'
        else:
            predictions_df.loc[i,'Catch'] = 'True Negative'

    size = predictions_df.shape[0]
    tp_rate = predictions_df.Catch.value_counts()['True Positive']/size
    wp_rate = predictions_df.Catch.value_counts()['Wrong Positive']/size
    fp_rate = predictions_df.Catch.value_counts()['False Positive']/size
    fn_rate = predictions_df.Catch.value_counts()['False Negative']/size
    tn_rate = predictions_df.Catch.value_counts()['True Negative']/size

    realp_rate = (predictions_df.Class.value_counts()[1] + predictions_df.Class.value_counts()[2])/size
    realn_rate = predictions_df.Class.value_counts()[0]/size

    # Creating df dictionary
    df_dict = {' ': ['Healthy', 'Suspected + Unhealthy'],
                 'Real': ["{:.2%}".format(realn_rate),
                          "{:.2%}".format(realp_rate)],
                 'Correctly Detected': ["{:.2%}".format(tn_rate),
                              "{:.2%}".format(tp_rate)],
                 'Difference': ["{:.2%}".format(realn_rate - tn_rate),
                                "{:.2%}".format(realp_rate - tp_rate)],
                 'Wrong prediction': ["{:.2%}".format(fp_rate + fn_rate),
                                      "{:.2%}".format(wp_rate)]}

    # Converting to dataframe
    summary_df = pd.DataFrame(df_dict)
    return(summary_df)



