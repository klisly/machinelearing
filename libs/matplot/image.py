#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/13 上午11:50
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : image.py
import matplotlib.pyplot as plt

# 读取一张小白狗的照片并显示
plt.figure('A Little White Dog')
little_dog_img = plt.imread('../../datas/dog.png')
plt.imshow(little_dog_img)

# Z是上小节生成的随机图案，img0就是Z，img1是Z做了个简单的变换
img0 = little_dog_img
img1 = 3 * little_dog_img + 4

# cmap指定为'gray'用来显示灰度图
fig = plt.figure('Auto Normalized Visualization')
ax0 = fig.add_subplot(121)
ax0.imshow(img0, cmap='gray')

ax1 = fig.add_subplot(122)
ax1.imshow(img1, cmap='gray')

plt.show()
