#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午6:50
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : xgboost.py
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

dataset = loadtxt('/Users/wizardholy/Documents/pywork/datasets/pima-indians-diabetes/pima-indians-diabetes.data.txt', delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]
# print(X)
# print(Y)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# model = XGBClassifier()
# model.fit(X_train, y_train)
model = XGBClassifier()
#
# eval_set = [(X_test, y_test)]
# model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
params = {'max_depth':[2,3,4,5,6],'n_estimators':[100,300,500,700,900,1100],'learning_rate':[0.05,0.1,0.25,0.5, 0.1]}

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, params, scoring="neg_log_loss", n_jobs=-1, cv=5)
grid_result = grid_search.fit(X, Y)
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# plot_importance(model)
# pyplot.show()