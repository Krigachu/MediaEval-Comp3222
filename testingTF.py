import codecs
import sys

import matplotlib.pyplot as plt
import pandas as pd
import re
import sklearn

print(sys.path)
# sys.path.append("C:/Program Files/Anaconda/envs/Coursework")
sys.path.append("C:/Program Files/Anaconda/envs/Coursework/Lib/site-packages")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from string import punctuation

import numpy as np
from langdetect import detect, DetectorFactory
# from langdetect import detect_langs
from textblob import TextBlob
import emoji
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DetectorFactory.seed = 0

def featureGeneration2(trainingSet):
    for index, row in trainingSet.iterrows():
        if row[6] == "real":
            target.append(1)
        else:
            target.append(0)

fullSet = pd.read_csv("training_set.csv", encoding="utf8", delimiter="x0x")

target = []

featureGeneration2(fullSet)
fullSet["target"] = target

trainingSet = fullSet.iloc[:10000,:]
trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/tfidfTesting.csv",encoding="utf8")

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainingSet.iloc[:,1])
print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, trainingSet.target)


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(trainingSet.iloc[:,1], trainingSet.target)

testingSplit = fullSet.iloc[10000:,:]
testingData = testingSplit.iloc[:,1]
testingTargets = testingSplit.iloc[:,6]

predict = text_clf.predict(testingData)
correct = 0
incorrect = 0
for index, row in enumerate(predict):
    if (row == testingTargets.iloc[index]):
        correct += 1
    else:
        incorrect += 1

accuracy = (correct / (correct + incorrect)) * 100
print("accuracy = ", accuracy)