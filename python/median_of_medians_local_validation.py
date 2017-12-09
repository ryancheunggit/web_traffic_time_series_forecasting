#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

# Windows = [11, 18, 30, 48, 78, 126, 203, 329] the windows from the original kaggle kernel

def smape(y_true, y_pred):
    '''
        calculate the symmetric mean absolute percentage error

        see: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    '''
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def read_data():
    # read in data
    '''
        -> train, test dataframes
    '''
    train = pd.read_csv("../input/train_1.csv")
    train = train.fillna(0)
    test = pd.read_csv("../input/key_1.csv")
    test['Page'] = test.Page.apply(lambda x: x[:-11])
    return train, test

def generate_predictions(
        data,
        params = {
                'RATIO' : 1.61803398875,
                'NUM_WINDOWS' : 9,
                'HANDULING': np.round,
                'TRAINING_STEPS': 365
            }
        ):
    '''
        generate predictions using the parameters

        Parameters
        ----------
            data : data frame, each row is a timeserie, the first column is the page name, the rest columns are page view of the day

        Returns
        -------
            a data frame with two columns, page name and median pageview of the medians from rolling windows
    '''
    print('shape of data passed in generate_predictions function is {}'.format(data.shape))
    # params
    TRAINING_STEPS = params['TRAINING_STEPS']
    RATIO = params['RATIO']
    NUM_WINDOWS = params['NUM_WINDOWS']
    step_size = np.floor(TRAINING_STEPS / (RATIO ** NUM_WINDOWS)).astype(int) # calculated from above two params
    windows = np.round(RATIO ** np.arange(1, NUM_WINDOWS + 1) * step_size).astype(int)
    # generate predictions
    predictions = np.zeros(data.shape[0])
    for idx, row in tqdm(data.iterrows()):
        medians = []
        nonzero_viewcount_idx = row.nonzero()[0]  # nonzero return a tuple, the first element is the np array of indices
        if len(nonzero_viewcount_idx) == 0:
            # the series is all zero, return 0
            medians = [0]
        else:
            first_nonzero_idx = nonzero_viewcount_idx[0]
            # the series has value
            if TRAINING_STEPS < windows[0] + first_nonzero_idx:
                # if the serie is shorter than the smallest window size, return median on nonzero values
                medians = [np.median(row.iloc[first_nonzero_idx: ])]
            else:
                # we can get medians from windows
                for window_size in windows:
                    if TRAINING_STEPS < window_size + first_nonzero_idx:
                        break
                    medians.append(np.median(row.iloc[-window_size:]))

        predictions[idx] = np.median(medians)
    predictions = params['HANDULING'](predictions).astype(int)
    return predictions

def local_validation(data, params, num_folds = 5, num_predictions = 60, date_shift = 10):
    '''
        create rolling median estimates on @TRAINING_STEPS days of value
        calculate score on the @num_predictions days since the last training date
        do that for @num_folds times with training start date shiftted by @date_shift
        on the given paramster set
    '''
    total_steps = data.shape[1]
    assert total_steps > params['TRAINING_STEPS'] + num_predictions + num_folds * date_shift, 'not long enough data to cover these folds'
    scores = []

    step_size = np.floor(params['TRAINING_STEPS'] / (params['RATIO'] ** params['NUM_WINDOWS'])).astype(int) # calculated from above two params
    windows = np.round(params['RATIO'] ** np.arange(1, params['NUM_WINDOWS'] + 1) * step_size).astype(int)

    print("==================================")
    print("RATIO : {}".format(params['RATIO']))
    print("NUM_WINDOWS : {}".format(params['NUM_WINDOWS']))
    print("TRAINING_STEPS : {}".format(params['TRAINING_STEPS']))
    print("step_size : {}".format(step_size))
    print("windows : {}".format(windows))

    for fold in range(num_folds):
        print("doing fold number {}".format(fold + 1))
        print("train - val splitting")
        first_prediction_date = total_steps - num_predictions - date_shift * fold
        first_training_date = first_prediction_date - params['TRAINING_STEPS']
        train = data.iloc[:, first_training_date : first_prediction_date]
        print("training set shape: {}".format(train.shape))
        print("training data range from {} to {}".format(train.columns[0], train.columns[-1]))
        test = data.iloc[:, first_prediction_date: first_prediction_date + num_predictions]
        print("test set shape: {}".format(test.shape))
        print("test data range from {} to {}".format(test.columns[0], test.columns[-1]))
        print("generating median of rolling medians")
        if params:
            predictions = generate_predictions(train, params = params)
        else:
            predictions = generate_predictions(train)
        print("calculating the SMAPE score")
        y_true = test.values.flatten()
        y_pred = np.array([np.repeat(pred, num_predictions) for pred in predictions]).flatten()
        score = smape(y_true, y_pred)
        print("the score is: {}".format(score))
        scores.append(score)
    return scores

def do_local_validation():
    param_grid = {
        'RATIO': [1.61803398875],
        'NUM_WINDOWS' : list(range(8, 12)),
        'HANDULING': [
            np.round
            #, np.ceil
            #, np.floor
            ],
        'TRAINING_STEPS': [365, 425, 305]
    }

    print("read in data")
    train, test = read_data()

    cv_scores = []
    for params in list(ParameterGrid(param_grid)):
        print("cv on params : {}".format(params))
        scores = local_validation(
                data = train.drop('Page', axis = 1),
                params = params,
                num_folds = 5,
                num_predictions = 60,
                date_shift = 10
            )
        print(scores)
        cv_scores.append((params, scores))
    return cv_scores

def create_submission():
    train, test = read_data()
    predictions = pd.DataFrame({
        'Page': train.Page,
        'Visits': generate_predictions(
            data = train,
            params = {
                    'RATIO' : 1.61803398875,
                    'NUM_WINDOWS' : 6,
                    'HANDULING': np.round,
                    'TRAINING_STEPS': 365
                }
            )
        }
    )
    sub = test.merge(predictions, on='Page', how='left')
    sub[['Id','Visits']].to_csv('sub_round_to_int_365_day_6_windows.csv', index=False)

def main():
    train, test = read_data()
    cv_scores = do_local_validation(cv_scores)
    d = pd.DataFrame(
            [[str(params['HANDULING']), params['NUM_WINDOWS'], params['RATIO'], score] for (params, scores) in cv_scores for score in scores],
            columns = ['func', 'num_windows', 'ratio', 'score']
        )
    sns.factorplot(x="num_windows", y="score", hue="func", data=d)
    plt.show()
