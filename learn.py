#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/18 下午6:27
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : learn.py
# -*-coding:utf-8-*-
# def distance(vector1, vector2):
#     d = 0
#     for a, b in zip(vector1, vector2):
#         d += (a - b) ** 2
#     return d ** 0.5
#
#
# v1 = (1, 1, 1, 1)
# v2 = (1, 1, 1, 1)
# print distance(v1, v2)
#
# v3 = (1, 1,1)
# v4 = (1,0,0)
# print distance(v3, v4)


for i in range(0,10):
    for j in range(0,10):
        print("mv ~/project/crawler/pages/wx/"+str(i)+str(j)+"* ~/data/wx")