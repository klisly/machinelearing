#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-26 14:47
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : script.py
import numpy as np
from scipy.spatial.distance import pdist

# x = np.random.random(10)
# y = np.random.random(10)
#
# # solution1
# dist1 = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
#
# # solution2
# dist2 = pdist(np.vstack([x, y]), 'cosine')
#
# print('x', x)
# print('y', y)
# print('dist1', dist1)
# print('dist2', dist2)
import time
min_day = int(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))  # 去除3天之前的数据
print(min_day)