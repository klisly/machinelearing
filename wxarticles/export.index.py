#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 上午10:17
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : export.index.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 上午1:08
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : cate.process.py
import os
import json
import pymongo
import time

MONGODB_CONFIG = {
    'host': '127.0.0.1',
    'port': 27017,
    'db_name': 'weixin',
}


class MongoConn(object):
    def __init__(self):
        try:
            self.conn = pymongo.MongoClient(MONGODB_CONFIG['host'], MONGODB_CONFIG['port'])
            self.db = self.conn[MONGODB_CONFIG['db_name']]  # connect db
            if self.username and self.password:
                self.connected = self.db.authenticate(self.username, self.password)
            else:
                self.connected = True
        except Exception as e:
            print(e)


my_conn = MongoConn()
with open('data.index/cates.txt', mode='w') as f:
    lines = list()
    datas = my_conn.db['cate'].find({})
    for data in datas:
        print(data['name'])
        lines.append(data['name'] + "\n")
    f.writelines(lines)

with open('data.index/id2cate.txt', mode='w') as f:
    lines = list()
    datas = my_conn.db['article'].find({})
    for data in datas:
        ds = data['url'][1:] + "\t" + data['cate'] + "\t" + data['org']
        lines.append(ds + "\n")
    f.writelines(lines)
