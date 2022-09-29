# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:15:35 2022

@author: davince
"""

from flaml import AutoML
from sklearn.metrics import r2_score
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler



data = pd.read_csv('D:\\科研文件\\20220623 谱相容余震地震动生成\\database\\inputtrain.csv')

x = data.iloc[:, :4]
y = data.iloc[:, 4]
x = x.values
y = y.values


# scaler0 = MinMaxScaler() 
# scaler0.fit(x)
# x = scaler0.transform(x)

# y = (y-np.min(y))/(np.max(y)-np.min(y))

automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 100,  # in seconds
    "metric": 'r2',
    "task": 'regression',
    "log_file_name": "california.log",
}

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

automl.fit(X_train=x, y_train=y, **automl_settings)
reg = automl.model.estimator

yp = reg.predict(X_test)
r2sc = r2_score(y_test,yp)

joblib.dump(reg, 'D:\\科研文件\\20220623 谱相容余震地震动生成\\database\\SaY.pkl')
