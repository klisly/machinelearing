#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/26 下午5:28
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : fasttexttest.py
# _*_coding:utf-8 _*_
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
#训练模型
classifier = fasttext.supervised("/Users/wizardholy/Downloads/news_fasttext_train.txt","/Users/wizardholy/Downloads/news_fasttext.model",label_prefix="__label__")

#load训练好的模型
#classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')

#测试模型
result = classifier.test("/Users/wizardholy/Downloads/news_fasttext_test.txt")
print(result.precision)
print(result.recall)
