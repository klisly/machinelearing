#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/7 下午3:21
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : KeyWord.py

# coding: utf-8

# In[1]:


import jieba
import numpy as np

def load_stopwords(file):
    stopwords = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line.__len__() > 0:
                stopwords.append(line)
    return stopwords

stopwords = load_stopwords("/Users/wizardholy/Documents/GitHub/eyespage-nlp/datas/dics/stopwords.txt")
print(stopwords)

def seg(file):
    docs = []
    odocs = []
    with open(file) as f:
        for line in f:
            if line.__len__() > 0:
                line = line.strip().replace('\u200b','')
                odocs.append(line)
                ds = list(jieba.cut(line))
                nds = []
                for d in ds:
                    if d not in stopwords:
                        nds.append(d)
                docs.append(' '.join(nds))
    return docs, odocs

docs, odocs = seg("/Users/wizardholy/Downloads/datas/joke.text.txt")
print(docs)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?',]

X = vectorizer.fit_transform(docs)


# In[10]:


feature_names = vectorizer.get_feature_names()


# In[11]:


doc = 0
for doc in range(100):
    feature_index = X[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [X[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        print(w, s)
    print (docs[doc])
    print(odocs[doc])


# In[12]:


print (docs[0])


# In[13]:


print(vectorizer.get_stop_words())


# In[14]:


vectorizer


# In[15]:


vectorizer.get_feature_names()


# In[16]:


indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 20
top_features = [features[i] for i in indices[:top_n]]
print(top_features)


# In[17]:


for i in indices:
    print(i)


# In[18]:


vectorizer.idf_[13203]

