#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import gc
import datetime

import numpy as np # linear algebra
np.seterr(divide='ignore', invalid='ignore')
from scipy import stats
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
tqdm.pandas(desc="progress monit")
import multiprocessing

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from matplotlib import pyplot as plt
import seaborn as sns

# ======== utility related function
def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes = workers)
    result = pool.map( _apply_df, [(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers))])
    pool.close()
    result = sorted(result,key=lambda x:x[0])
    return pd.concat([i[1] for i in result])

# ======== metrics functions
def smape(y_true, y_pred):
    '''
        calculate the symmetric mean absolute percentage error
        see: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

        y_true: array of numbers , y_pred: array of numbers -> score: float
    '''
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def smape_lightgbm(y_pred, train_data):
    '''
        eval for lightgbm
        y_pred: array of predicted values , train_data: lgb dataset from which y_pred is generated -> 'score name', score, is higher value better?
    '''
    y_true = train_data.get_label()
    return 'smape', smape(y_true, y_pred), False

def smape_lightgbm_inv_log(y_pred, train_data):
    '''
        eval for lightgbm when the input data is already went through np.log1p transformation
        y_pred: array of predicted values , train_data: lgb dataset from which y_pred is generated -> 'score name', score, is higher value better?
    '''
    y_true = train_data.get_label()
    return 'smape', smape(np.exp(y_true) - 1, np.exp(y_pred) - 1), False

def make_smape_lightgbm_inv_log_use_data(Y_vals, NUM_DAYS):
    '''
        higher order function to create evaluation function
    '''
    def smape_lightgbm_inv_log_use_data(y_pred, train_data):
        '''
            the function do evluation on actual data rather than the median target
        '''
        y_true = Y_vals.flatten()
        y_pred = np.array([np.repeat(i, NUM_DAYS) for i in y_pred]).flatten()
        return 'smape', smape(np.exp(y_true) - 1, np.exp(y_pred) - 1), False
    return smape_lightgbm_inv_log_use_data

"""
#problematic
def mapeobj(y_pred, train_data, group = None):
    y_true = train_data.get_label().values
    '''
    grad = np.sign(y_pred - y_true) * 1.0 / y_true
    hess = 0 * np.ones(len(y_pred))

    grad = -100((y_true - y_pred)/y_true)
    hess = 100/(y_true)
    '''
    import autograd
    mape = lambda y_true , y_pred: autograd.numpy.sum(autograd.numpy.abs(y_pred - y_true)*1.0/(y_true))*1.0/len(y_true)
    gradient = autograd.grad(mape)
    hessian = autograd.hessian(mape)
    return gradient(y_true, y_pred), hessian(y_true, y_pred)
"""

# ========= data IO function
'''
# rename the unzipped key_2.csv to key_2_original and run the following to fix the page column
# no need anymore, data is officially fixed
with open('../input/key_2.csv', 'w', encoding='utf-8') as outFile, open('../input/key_2_original.csv', 'r', encoding='utf-8') as inFile:
    for line in inFile:
        Fields = line.strip().replace('"','""').split(sep=',')
        Page = ','.join(Fields[:-1])
        newLine = '"%s","%s"\n' % (Page, Fields[-1])
        outFile.write(newLine)
'''

def read_data(log_transform = False, set_number = 1):
    '''
        read in dataset
        set_number 1 is for phase 1
        set_number 2 is for phase 2
        log_transform: whether to do np.log1p transform on input data or not -> train, test dataframes
        usage:
            data, test = read_data(log_transform = True, set_number = 2)
    '''
    train = pd.read_csv("../input/train_{}.csv".format(set_number))
    train = train.fillna(0)
    if log_transform:
        train.iloc[:, 1:] = np.log1p(train.iloc[:,1:])

    test = pd.read_csv("../input/key_{}.csv".format(set_number))
    test['Page'] = test.Page.apply(lambda x: x[:-11])
    return train, test

# ========== feature related functions
def get_reg_slope(y):
    '''
        y: pandas Series of view counts -> slope of linear regression fit (ie correlation coefficient)
    '''
    return stats.linregress(np.arange(y.shape[0]) + 1, y)[0]

def get_spearman_r(y):
    '''
        y: pandas Series of view counts -> Spearman rank-order correlation coefficient
    '''
    return stats.spearmanr(np.arange(y.shape[0]) + 1, y)[0]

def find_first_none_zero_dss(views):
    '''
        views: pandas Series of page view counts -> location where first non zero page view happens
    '''
    dss = views.nonzero()[0]
    if len(dss) == 0:
        return views.shape[0]
    else:
        return dss[0]

def extract_y(data, start_idx, NUM_DAYS):
    '''
        extract target for training
        if the start_idx is less than 2 NUM_DAYS towards the end of date, return all zeros as Y (np.array)
        else return actual values of start_idx + 2 * NUM_DAYS - 1 as Y

        enssentially this is to use NUM_DAYS period feature to predict the last day value in the adjacent next NUM_DAYS period
    '''
    assert start_idx + NUM_DAYS <= data.shape[1], "start_idx, NUM_DAYS combine is outside data range"
    if start_idx + 2 * NUM_DAYS > data.shape[1]:
        print("Y is all zero")
        return np.zeros(data.shape[0])
    elif start_idx + 2 * NUM_DAYS == data.shape[1]:
        print("Y is the values from date {}".format(data.columns[-1]))
        return data.iloc[:, -1].values
    print("Y is the values from date {}".format(data.columns[start_idx + 2 * NUM_DAYS - 1]))
    return data.iloc[:, start_idx + 2 * NUM_DAYS - 1].values

def extract_median_y(data, start_idx, NUM_DAYS):
    '''
        extract target for training
        if the start_idx is less than one NUM_DAYS towards the end of date, return all zeros as Y and Y_vals (np.array)
        else return median of the NUM_DAYS span from start_idx + NUM_DAYS towards right as Y and actual values of that span as Y_vals

        this target is upper bounded to mid thirty-ish smape score
    '''
    assert start_idx + NUM_DAYS <= data.shape[1], "start_idx, NUM_DAYS combine is outside data range"
    if start_idx + 2 * NUM_DAYS > data.shape[1]:
        print("Y and Y_vals are all zeros")
        return np.zeros(data.shape[0]), np.zeros([data.shape[0], NUM_DAYS])
    elif start_idx + 2 * NUM_DAYS == data.shape[1]:
        data_slice = data.iloc[:, start_idx + NUM_DAYS:]
        print("Y is the median of {} and {}, Y_val is the actual values".format(data_slice.columns[0], data_slice.columns[-1]))
        return data_slice.median(axis = 1).values, data_slice.values
    data_slice = data.iloc[:, start_idx + NUM_DAYS : start_idx + 2 * NUM_DAYS]
    print("Y is the median of {} and {}, Y_val is the actual values".format(data_slice.columns[0], data_slice.columns[-1]))
    return data_slice.median(axis = 1).values, data_slice.values

def extract_x(data, start_idx, NUM_DAYS, verbose = 0):
    assert start_idx + NUM_DAYS <= data.shape[1], "start_idx, NUM_DAYS combine is outside data range"
    date_to_predict = pd.to_datetime(
            pd.to_datetime(data.columns.values[start_idx]).to_pydatetime() + \
            datetime.timedelta(days = NUM_DAYS * 2)
        )
    # copy some raw view count data
    X = data.iloc[:, start_idx: start_idx + NUM_DAYS].copy()
    # date that we want to predict
    X['__month_to_predict']         = date_to_predict.month
    X['__day_to_predict']           = date_to_predict.day
    X['__day_of_week_to_predict']   = date_to_predict.dayofweek
    X['__day_of_year_to_predict']   = date_to_predict.dayofyear
    X['__week_of_year_to_predict']  = date_to_predict.weekofyear
    # type of device that accessed the page
    X['__source']                   = data.Page.map(lambda page: page.split('.org')[-1])
    X['__source']                   = X['__source'].map(X['__source'].value_counts().to_dict())
    # language of the page
    X['__page_type']                = data.Page.map(lambda x: x.split('.wiki')[0].split('_')[-1])
    X['__page_type']                = X['__page_type'].map(X['__page_type'].value_counts().to_dict())
    # the current month
    X['__mean_month']               = data.iloc[:, start_idx: start_idx + NUM_DAYS].columns.map(lambda x: x.split('-')[1]).values.astype(int).mean()
    X['__median_month']             = np.median(data.iloc[:, start_idx: start_idx + NUM_DAYS].columns.map(lambda x: x.split('-')[1]).values.astype(int))
    # the first day since the starting timepoint that we see a first nonzero view for the page
    X['__first_day_none_zero']      = apply_by_multiprocessing(data.iloc[:, 1:], find_first_none_zero_dss, axis = 1, workers = 4)
                                     # data.iloc[:, 1:].progress_apply(find_first_none_zero_dss, axis = 1)
    # the number of days between the first nonzero view day to the starting day that we want to predict views
    X['__days_since_none_zero']     = data.shape[1] - X['__first_day_none_zero']

    # generateing summary feature based on different time windows size backward
    for time_step in [7, 14, 28, 49, 77, 126, 203, 329]:

        start = start_idx + NUM_DAYS - time_step
        end   = start_idx + NUM_DAYS
        section = data.iloc[:, start : end]
        dates = section.columns.values
        feature_prefix = '__last_{}_day'.format(time_step)

        if verbose > 0:
            print("Creating summary features from last {} days ".format(time_step))
            print("time now is :{}".format(datetime.datetime.now()))
            print("using data range from {} to {}".format(dates[0], dates[-1]))

        # median page view during window
        X[feature_prefix + '_median']               = section.median(axis = 1)
        # mean page view during window
        X[feature_prefix + '_mean']                 = section.mean(axis = 1)
        # min page view druing window
        X[feature_prefix + '_min']                  = section.min(axis = 1)
        # max page view during window
        X[feature_prefix + '_max']                  = section.max(axis = 1)
        # half the difference between max page view and min page view during window
        X[feature_prefix + '_amplitude']            = X[feature_prefix + '_max'] - \
                                                                    X[feature_prefix + '_min'] / 2
        # the maximum distance from median page view to max and min page view during window
        X[feature_prefix + '_median_abs_dev']       = np.max(
                np.array([
                    X[feature_prefix + '_max'] - X[feature_prefix + '_median'],
                    X[feature_prefix + '_median'] - X[feature_prefix + '_min']]
                ).T,
                axis = 1
            )
        # the ratio between median abs dev and amplitude during the window
        X[feature_prefix + '_percent_amplitude']    = X[feature_prefix + '_median_abs_dev'] /\
                                                                    X[feature_prefix + '_amplitude']
        # stdev of page view during the window
        X[feature_prefix + '_std']                  = section.std(axis = 1)
        # skew during the window
        X[feature_prefix + '_skew']                 = section.skew(axis = 1)
        # kurtosis during the window
        X[feature_prefix + '_kurtosis']             = section.kurtosis(axis = 1)
        # number of day since the first non zero day to the start of the window
        X[feature_prefix + '_days_since_none_zero'] = data.shape[1] - time_step - X['__first_day_none_zero']
        # the 1st order linear regression coef of page view to time during the window
        X[feature_prefix + '_trend']                = apply_by_multiprocessing(section, get_reg_slope, axis = 1, workers = 4)
                                                                    # section.progress_apply(get_reg_slope , axis = 1) # old code
        # the spearman rank coef of page view to time during the window
        X[feature_prefix + '_spearmanr']            = apply_by_multiprocessing(section, get_spearman_r, axis = 1, workers = 4)
                                                                    # section.progress_apply(get_spearman_r , axis = 1)
        # some diff feature
        X[feature_prefix + '_diff_mean']            = section.diff(axis = 1).mean(axis = 1)
        X[feature_prefix + '_diff_median']          = section.diff(axis = 1).median(axis = 1)
        X[feature_prefix + '_diff_max']             = section.diff(axis = 1).max(axis = 1)
        X[feature_prefix + '_diff_min']             = section.diff(axis = 1).min(axis = 1)
        X[feature_prefix + '_diff_std']             = section.diff(axis = 1).std(axis = 1)
        X[feature_prefix + '_2nd_diff_mean']        = section.diff(axis = 1).diff(axis = 1).mean(axis = 1)
        X[feature_prefix + '_2nd_diff_median']      = section.diff(axis = 1).diff(axis = 1).median(axis = 1)
        X[feature_prefix + '_2nd_diff_max']         = section.diff(axis = 1).diff(axis = 1).max(axis = 1)
        X[feature_prefix + '_2nd_diff_min']         = section.diff(axis = 1).diff(axis = 1).min(axis = 1)
        X[feature_prefix + '_2nd_diff_std']         = section.diff(axis = 1).diff(axis = 1).std(axis = 1)
        X[feature_prefix + '_7th_diff_mean']        = section.diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).mean(axis = 1)
        X[feature_prefix + '_7th_diff_median']      = section.diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).median(axis = 1)
        X[feature_prefix + '_7th_diff_max']         = section.diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).max(axis = 1)
        X[feature_prefix + '_7th_diff_min']         = section.diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).min(axis = 1)
        X[feature_prefix + '_7th_diff_std']         = section.diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).diff(axis = 1).std(axis = 1)

        X = X.fillna(0)

    return X

def feature_gen(data, start_idx, NUM_DAYS, predict_median = False):
    '''
        generate X, Y (and optional Y_vals) for two types of training
    '''
    assert start_idx + NUM_DAYS <= data.shape[1], "start_idx, NUM_DAYS combine is outside data range"
    X = extract_x(data, start_idx, NUM_DAYS)
    if not predict_median:
        Y = extract_y(data, start_idx, NUM_DAYS)
        return X, Y, Y
    Y, Y_vals = extract_median_y(data, start_idx, NUM_DAYS)
    return X, Y, Y_vals

# ======== modeling related
def gbm_cv_median_target(NUM_DAYS = None):
    '''
        do lightgbm cv locally with median as target
    '''
    if not NUM_DAYS:
        NUM_DAYS = 62

    data, test = read_data(log_transform = True, set_number = 2) # do the page fix first

    X_train  , Y_train , Y_vals_train = feature_gen(data, start_idx = data.shape[1] - 3 * NUM_DAYS, NUM_DAYS = NUM_DAYS, predict_median = True)
    X_valid  , Y_valid , Y_vals_valid = feature_gen(data, start_idx = data.shape[1] - 2 * NUM_DAYS, NUM_DAYS = NUM_DAYS, predict_median = True)
    X_test   , _       , _            = feature_gen(data, start_idx = data.shape[1] - 1 * NUM_DAYS, NUM_DAYS = NUM_DAYS, predict_median = True)

    steps = 31
    # augument the dataset
    for shift in range(1, 3): # approx 13 mins per loop
        # shift the data preparation by step forward and append the created dataset to the bottom of the training dataframe
        print("creating shift {} data".format(shift))
        X_temp, Y_temp, Y_vals_temp = feature_gen(data, start_idx = data.shape[1] - 3 * NUM_DAYS - shift * steps, NUM_DAYS = NUM_DAYS, predict_median = True)
        X_temp.columns = X_train.columns
        X_train = X_train.append(X_temp)
        Y_train = np.append(Y_train, Y_temp)
        Y_vals_train = np.append(Y_vals_train, Y_vals_temp, axis = 0)

        del [X_temp, Y_temp, Y_vals_temp]
        gc.collect()

        X_temp, Y_temp, Y_vals_temp = feature_gen(data, start_idx = data.shape[1] - 2 * NUM_DAYS - shift * steps, NUM_DAYS = NUM_DAYS, predict_median = True)
        X_temp.columns = X_valid.columns
        X_valid = X_valid.append(X_temp)
        Y_valid = np.append(Y_valid, Y_temp)
        Y_vals_valid = np.append(Y_vals_valid, Y_vals_temp, axis = 0)

        del [X_temp, Y_temp, Y_vals_temp]
        gc.collect()

    # lowest training smape score
    print("Lowest trainning SMAPE possible is {}".format(
        smape(
            y_true = np.exp(Y_vals_train.flatten()) - 1 ,
            y_pred = np.exp(np.array([np.repeat(i, NUM_DAYS) for i in Y_train]).flatten()) -1
        )))
    # lowest validation smape score
    print("Lowest validation SMAPE possible is {}".format(
        smape(
            y_true = np.exp(Y_vals_valid.flatten()) - 1 ,
            y_pred = np.exp(np.array([np.repeat(i, NUM_DAYS) for i in Y_valid]).flatten()) -1
        )))

    # create lgb datasets
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval  = lgb.Dataset(X_valid, Y_valid, reference = lgb_train)

    # create custom eval function
    feval = make_smape_lightgbm_inv_log_use_data(Y_vals_valid, NUM_DAYS)  # creating the feval function for this validation dataset....

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'l1',
            'max_bin': 511,
            'num_leaves': 255,
            'num_threads': 4,
            #'max_depth': 9,
            'learning_rate': 0.02,
            'feature_fraction': 1,
            'bagging_fraction': 0.8,
            'verbose' : 0
        }

    gbm = lgb.train(
            params = params,
            train_set = lgb_train,
            valid_sets = lgb_eval,
            num_boost_round = 1024,
            #fobj = mapeobj, #does not work
            feval = feval,
            early_stopping_rounds = 20,
            verbose_eval = 20
        )

    feature_importance = pd.DataFrame({
            'feature_name'       : gbm.feature_name(),
            'feature_importance' : gbm.feature_importance()
        })

    feature_importance.sort_values(by = 'feature_importance', inplace = True)
    feature_importance.to_csv('feature_importance_fold_gap_cv_steps_lower_learning_rate.csv', index = False)

    pred = np.round(np.exp(gbm.predict(X_valid)) - 1)
    pred = np.array([np.repeat(i, NUM_DAYS) for i in pred]).flatten()
    smape(np.exp(Y_vals_valid.flatten()) -1, pred)

    final_gbm = lgb.train(
            params = params,
            train_set = lgb_eval,
            num_boost_round = gbm.best_iteration
        )

    predictions = np.round(np.exp(final_gbm.predict(X_test)) - 1)

    predictions = pd.DataFrame({
            'Page'   : data.Page,
            'Visits' : predictions
        })

    sub = test.merge(predictions, on='Page', how='left')
    sub['Visits'] = sub['Visits'].astype(int)
    sub[['Id','Visits']].to_csv('stage2_sub_ver2_xgb.csv', index=False)

def median_of_medians_validation(data, NUM_DAYS):
    # better than my code
    Windows = [11, 18, 30, 48, 78, 126, 203, 329]
    train = data.iloc[:, : data.shape[0] - NUM_DAYS]
    n = train.shape[1] - 1 #  550
    Visits = np.zeros(train.shape[0])
    for i, row in tqdm(train.iterrows()):
        M = []
        start = row[1:].nonzero()[0]
        if len(start) == 0:
            continue
        if n - start[0] < Windows[0]:
            Visits[i] = row.iloc[start[0]+1:].median()
            continue
        for W in Windows:
            if W > n-start[0]:
                break
            M.append(row.iloc[-W:].median())
        Visits[i] = np.median(M)

    Visits[np.where(Visits < 1)] = 0.
    y_pred = np.exp(np.array([np.repeat(i, NUM_DAYS) for i in Visits]).flatten()) -1
    y_true = data.iloc[:, data.shape[1] - NUM_DAYS:].values.flatten()
    smape(np.exp(y_true) -1, y_pred)


    train['Visits'] = np.round(Visits).astype('int')
    test = pd.read_csv("../input/key_2.csv")
    test['Page'] = test.Page.apply(lambda x: x[:-11])

    test = test.merge(train[['Page','Visits']], on='Page', how='left')
    test[['Id','Visits']].to_csv('median_of_medians_with_out_11th_data.csv', index=False) # score 44.8 on plb in first phase
