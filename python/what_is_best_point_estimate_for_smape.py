#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

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

def experiment():
    train, test = read_data()
    mean_smapes = []
    median_smapes = []
    m1 = []
    for idx, row in tqdm(train.iterrows()):
        mean_smapes.append(smape(y_true = row[-60:].tolist(), y_pred = np.repeat(int(np.mean(row[-60:])), 60)))
        median_smapes.append(smape(row[-60:].tolist(), np.repeat(np.median(row[-60:]), 60)))
        m1.append(smape(row[-60:].tolist(), np.repeat(int(np.mean([np.median(row[-60:]), np.mean(row[-60:])])), 60)))

    scores = pd.DataFrame({
        'mean_smapes': mean_smapes,
        'median_smapes': median_smapes,
        'm1': m1
        })

    scores.describe()

if __name__ == '__main__':
    experiment() # conclution : median is best point estimates for smape purpose. 
