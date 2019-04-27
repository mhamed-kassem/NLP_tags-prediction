# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 05:03:10 2019

@author: dell
"""

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('stack-overflow-data.csv')
df = pd.DataFrame(data)

y = df['tags']
y = y[:2000]
X = df['post']
X = X[:2000]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)
#print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, test_size=0.33, random_state=42)


clf = MultinomialNB().fit(X_train, y_train)
y_predict = clf.predict(X_test)
score = clf.score(X_test,y_test)

print('MultinomialNB SCORE : ',score)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test, y_predict))


'''

y = df['tags']
y = y[:2000]
X = df['post']
X = X[:2000]


tfidf = TfidfVectorizer(min_df=1,max_df=5,ngram_range=(1,2))
features = tfidf.fit_transform(X)
features = pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())



X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_predict = gnb.predict(X_test)
score = gnb.score(X_test,y_test)
print('GaussianNB SCORE : ',score)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test, y_predict))


'''