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
from sklearn.model_selection import GridSearchCV
DetectorFactory.seed = 0



englishTrainingSet = pd.read_csv("D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/englishTrainingSet.csv",encoding="utf8")
print(englishTrainingSet["label"].value_counts())
realRecords = englishTrainingSet[englishTrainingSet["label"] == "real"]
fakeRecords = englishTrainingSet[englishTrainingSet["label"] == "fake"]
humourRecords = englishTrainingSet[englishTrainingSet["label"] == "humor"]

INDICESREAL = 4000
INDICESFAKE = 3000
INDICESHUMOUR = 1000

realRecordsData = realRecords.iloc[:INDICESREAL, 10:21]
fakeRecordsData = fakeRecords.iloc[:INDICESFAKE, 10:21]
humourRecordsData = humourRecords.iloc[:INDICESHUMOUR, 10:21]

realTargets = realRecords.iloc[:INDICESREAL, 21]
fakeTargets = fakeRecords.iloc[:INDICESFAKE, 21]
humourTargets = humourRecords.iloc[:INDICESHUMOUR, 21]

subTrainingSet = pd.concat([realRecordsData, fakeRecordsData, humourRecordsData])
subTargetSet = pd.concat([realTargets, fakeTargets, humourTargets])

# testing split
realTestRecordsData = realRecords.iloc[INDICESREAL:, 10:21]
fakeTestRecordsData = fakeRecords.iloc[INDICESFAKE:, 10:21]
humourTestRecordsData = humourRecords.iloc[INDICESHUMOUR:, 10:21]

realTestTargets = realRecords.iloc[INDICESREAL:, 21]
fakeTestTargets = fakeRecords.iloc[INDICESFAKE:, 21]
humourTestTargets = humourRecords.iloc[INDICESHUMOUR:, 21]

subTestTrainingSet = pd.concat([realTestRecordsData, fakeTestRecordsData, humourTestRecordsData])
subTestTargetSet = pd.concat([realTestTargets, fakeTestTargets, humourTestTargets])
print(subTargetSet.shape)
print(subTargetSet.shape)

# 10956 records with english text according to detectLang

# showLabelComposition(englishTrainingSet)
# Fake   47.09424322598303 % aka ~ 62.68%
# Real   37.32323693093696 %
# Humor   15.58251984308001 %

# 263004702072004608	one of these are dupes.
# 263030750415319040

# showLabelComposition()


# dumb training and testing dataset
# subTrainingSet = trainingSet.iloc[:99,:6]
# subTestingSet = trainingSet.iloc[100:199,:6]
# print(subTrainingSet)
# print(subTestingSet)

# subTrainingSetTarget = trainingSet.iloc[:99,6]
# subTestingSetTarget = trainingSet.iloc[100:199,6]


# making svc -> support vector classifier
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svm = svm.SVC()
classifier = GridSearchCV(svm, parameters)

classifier.fit(subTrainingSet, subTargetSet)
predictions = classifier.predict(subTestTrainingSet)
correct = 0
incorrect = 0
for index, row in enumerate(predictions):
    if (row == subTestTargetSet.iloc[index]):
        correct += 1
    else:
        incorrect += 1

accuracy = (correct / (correct + incorrect)) * 100
print("accuracy = ", accuracy)
print('RMSE = ', np.sqrt(mean_squared_error(subTestTargetSet, predictions)))
print(classifier.cv_results_)

