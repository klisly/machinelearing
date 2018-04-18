#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/9 下午2:25
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : WordFreq.py
from nltk.util import ngrams
import jieba

parts = list()
holders = set()
with open("/Users/wizardholy/Documents/GitHub/nlp/datas/case_template/final.query.txt") as f:
    for line in f:
        line = line.strip()
        if line.count("<") > 0:
            bindex = 0
            start = line.index("<", 0)
            try:
                while start >= 0:
                    end = line.index(">", start + 1)
                    holders.add(line[start:end + 1])
                    start = line.index("<", end + 1)
            except:
                start = -1

with open("/Users/wizardholy/Documents/GitHub/nlp/datas/case_template/final.query.txt") as f:
    for line in f:
        line = line.strip()
        for holder in holders:
            line = line.replace(holder, ' ')
        ds = line.split(' ')
        for d in ds:
            if len(d) > 0:
                parts.append(d)
count_dict = dict()
for part in parts:
    for word in jieba.cut(part):
        if word in count_dict.keys():
            count_dict[word] = count_dict[word] + 1
        else:
            count_dict[word] = 1
    # for size in range(2, 4):
    #     for d in list(ngrams(part, size)):
    #         word = ''.join(d)

    # 按照词频从高到低排列
count_list = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

in_metas = set()
with open("/Users/wizardholy/Documents/GitHub/nlp/datas/simwords.txt") as f:
    count = 0
    for line in f:
        count += 1
        if count % 2 == 0:
            ds = line.strip().split('|')
            for d in ds:
                if len(d) > 0:
                    in_metas.add(d)

str = ""
for key, value in count_list:
    if key not in in_metas:
        print(key, value)
        str += key+"|"
print(str)