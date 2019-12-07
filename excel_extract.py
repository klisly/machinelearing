#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/11 10:05 PM
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : excel.py
import pandas as pd

datas = pd.read_excel('/Users/wizardholy/Documents/baice/up/用户画像 规划 (1).xlsx')
print(datas)

import numpy as np

from scipy import spatial