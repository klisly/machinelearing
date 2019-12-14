#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-12-09 22:54
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : sample.deepfm.py
# -- deepfm 520的auc  16，17，18 -> 19 0.7734
# -- m1,bid,adid,adspaceid,adspacetype,adtype,nt,appid,adname,mo,osv,h_of_d,d_of_w,pdt,flag,p_city,install_pkgs,click_adids

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.preprocessing.sequence import pad_sequences
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_feature_names
from sklearn.metrics import roc_auc_score
def split_pkgs(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_pkgs:
            key2index_pkgs[key] = len(key2index_pkgs) + 1
    return list(map(lambda x: key2index_pkgs[x], key_ans))

def split_adis(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index_adids:
            key2index_adids[key] = len(key2index_adids) + 1
    return list(map(lambda x: key2index_adids[x], key_ans))

data = pd.read_csv("/data10t/john/adspace520train/corpus_520_1016_1019.csv")
sparse_features = ["adid", "adspaceid", "adspacetype", "adtype", "nt", "appid", "adname", "mo", "d_of_w", "h_of_d", "osv", "p_city"]
target = "flag"
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat].astype(str))

key2index_pkgs = {}
pkgs_list = list(map(split_pkgs, data['install_pkgs'].values))
pkgs_length = np.array(list(map(len, pkgs_list)))
max_pkgs_len = max(pkgs_length)
pkgs_list = pad_sequences(pkgs_list, maxlen=max_pkgs_len, padding='post', )

key2index_adids = {}
adids_list = list(map(split_adis, data['click_adids'].values))
adids_length = np.array(list(map(len, adids_list)))
max_adids_len = max(adids_length)
adids_list = pad_sequences(adids_list, maxlen=max_adids_len, padding='post', )

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
varlen_feature_columns_pkgs = [VarLenSparseFeat('install_pkgs', len(key2index_pkgs) + 1, max_pkgs_len, 'mean')] #
varlen_feature_columns_adids = [VarLenSparseFeat('click_adids', len(key2index_adids) + 1, max_adids_len, 'mean')] #

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns_pkgs + varlen_feature_columns_adids
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns_pkgs + varlen_feature_columns_adids
feature_names = get_feature_names(linear_feature_columns+dnn_feature_columns)

train_input = data[data["pdt"] != 20191019]
test_input = data[data["pdt"] == 20191019]

train_model_input = {name:train_input[name] for name in feature_names}#
train_pkgs = [pkgs_list[i] for i in train_input.index.values]
train_model_input["install_pkgs"] = np.array(train_pkgs)
train_adids = [adids_list[i] for i in train_input.index.values]
train_model_input["click_adids"] = np.array(train_adids)

test_model_input = {name:test_input[name] for name in feature_names}#
test_pkgs = [pkgs_list[i] for i in test_input.index.values]
test_model_input["install_pkgs"] = np.array(test_pkgs)
test_adids = [adids_list[i] for i in test_input.index.values]
test_model_input["click_adids"] = np.array(test_adids)

model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(train_model_input, train_input[target].values, batch_size=2000, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=2000)
print("test AUC", round(roc_auc_score(test_input[target].values, pred_ans), 4))
history = model.fit(train_model_input, train_input[target].values, batch_size=2000, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=2000)
print("test AUC", round(roc_auc_score(test_input[target].values, pred_ans), 4))