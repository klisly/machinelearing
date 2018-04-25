#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 上午11:40
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : data.prepare.py
import os
import json
import pymongo
import time
import os
MONGODB_CONFIG = {
    'host': '127.0.0.1',
    'port': 27017,
    'db_name': 'second',
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

datas = my_conn.db['wxarticles'].find({})
for data in datas:
    title = data['title']
    href = data['href']
    account = data['account']
    tag = data['tag']
    print(title + "\t" + href + "\t" + tag)

