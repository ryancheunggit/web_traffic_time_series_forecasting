#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

# read in data
train = pd.read_csv("../input/train_3.csv")
train = train.fillna(0)

r = 1.61803398875
#Windows = np.round(r**np.arange(1,10) * 6).astype(int)
Windows = [11, 18, 30, 48, 78, 126, 203, 329]

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
train['Visits'] = np.round(Visits).astype('int')


test = pd.read_csv("../input/key_2.csv")
test['Page'] = test.Page.apply(lambda x: x[:-11])

test = test.merge(train[['Page','Visits']], on='Page', how='left')
test[['Id','Visits']].to_csv('median_of_medians_with_11th_data.csv', index=False)
# score 44.8 on plb in first phase
