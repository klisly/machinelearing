#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 上午10:18
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : xgboostcli.py
import xgboost as xgb
# import csv
import jieba
import random
# jieba.load_userdict("wordDict.txt")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 读取训练集
def readtrain(file):
    with open(file, "r") as readf:
        content_train = []
        opinion_train = []
        lines = readf.readlines()
        random.shuffle(lines)
        for line in lines:
            ds = line.strip().split("\t")
            content_train.append(ds[0])
            opinion_train.append(ds[1])
    print("训练集有 %s 条句子" % len(content_train))
    train = [content_train, opinion_train]
    return train


# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode("utf8"))
    return a


# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c

clk = {"热映电影": 1, "火车票": 2, "景点": 3, "查周边": 4, "航班": 5, "在线视频": 6, "人物": 7, "查路况": 8, "外卖": 9, "音乐": 10, "餐馆": 11, "影院": 12, "酒店": 13, "股票": 14, "导航": 15, "城市": 16}
# 类别用数字表示：pos:2,neu:1,neg:0
def transLabel(labels):
    print("labels:"+str(labels))
    for i in range(len(labels)):
        labels[i] = clk[labels[i]]
    return labels


train = readtrain("/Users/wizardholy/Documents/GitHub/nlp/datas/corpus/train.txt")
content = segmentWord(train[0])
opinion = transLabel(train[1])  # 需要用数字表示类别
print(opinion)
# opinion = np.array(opinion)     # 需要numpy格式
#
#
train_content = content[:]
train_opinion = opinion[:]
test_content = content[3000:]
test_opinion = opinion[3000:]

vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))
weight = tfidf.toarray()
print(tfidf.shape)
test_tfidf = tfidftransformer.transform(vectorizer.transform(test_content))
test_weight = test_tfidf.toarray()
print(test_weight.shape)

dtrain = xgb.DMatrix(weight, label=train_opinion)
dtest = xgb.DMatrix(test_weight, label=test_opinion)  # label可以不要，此处需要是为了测试效果
param = {"max_depth":6, "eta":0.5, "eval_metric":"merror", "silent":1, "objective":"multi:softmax", "num_class":len(clk.keys()) + 1}  # 参数
evallist  = [(dtrain,"train"), (dtest,"test")]  # 这步可以不要，用于测试效果
num_round = 50  # 循环次数
bst = xgb.train(param, dtrain, num_round, evallist)
preds = bst.predict(dtest)
