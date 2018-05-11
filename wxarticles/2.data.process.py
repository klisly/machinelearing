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
import shutil
import nltk
from lxml import html
from bs4 import BeautifulSoup

MONGODB_CONFIG = {
    'host': '127.0.0.1',
    'port': 27017,
    'db_name': 'second',
}


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


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
corpus_dir = "./data/corpus"
if os.path.exists(corpus_dir):
    shutil.rmtree(corpus_dir)

datas = my_conn.db['wxarticles'].find({})
for data in datas:
    title = data['title']
    href = data['href']
    account = data['account']
    tag = data['tag']
    if len(tag) > 0:
        srcFile = '/Users/wizardholy/data/wx' + "/" + href
        if os.path.exists(srcFile):
            targetFilePath = corpus_dir + "/" + tag + "/" + href
            ensure_dir(targetFilePath)
            with open(srcFile, encoding='utf-8') as f:
                data = html.fromstring(f.read())
                text = ''.join(data.xpath("//div[@class='rich_media_content ']/descendant-or-self::*/text()"))
                with open(targetFilePath, mode='w', encoding='utf-8') as tf:
                    tf.write(text)
        else:
            print("not exist file:" + href)
