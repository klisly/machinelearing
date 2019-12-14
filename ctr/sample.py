#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-12-11 17:11
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : sample.py
import math


def walson_ctr(num_click, num_pv):
    if num_pv * num_click == 0 or num_pv < num_click:
        return 0
    z = 1.96
    n = num_pv
    p = 1.0 * num_click / num_pv
    score = (p + z * z / (2 * n) - z * math.sqrt((p * (1.0 - p) + z * z / (4. * n)) / n)) / (1. + z * z / n)
    return score


print(walson_ctr(5, 10))
print(walson_ctr(50, 100))
print(walson_ctr(500, 10000))
