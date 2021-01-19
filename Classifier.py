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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


DetectorFactory.seed = 0

#englishTrainingSet = pd.read_csv("D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/englishTrainingSet.csv",encoding="utf8")
#trainingSet = pd.read_csv("allLanguagesDataset.csv",encoding="utf8")
trainingSet = pd.read_csv("trainingSetAllFeatures.csv",encoding="utf8")



#print(englishTrainingSet["label"].value_counts())
#realRecords = englishTrainingSet[englishTrainingSet["label"] == "real"]
#fakeRecords = englishTrainingSet[englishTrainingSet["label"] == "fake"]
#humourRecords = englishTrainingSet[englishTrainingSet["label"] == "humor"]

print(trainingSet["label"].value_counts())
realRecords = trainingSet[trainingSet["label"] == "real"]
fakeRecords = trainingSet[trainingSet["label"] == "fake"]
humourRecords = trainingSet[trainingSet["label"] == "humor"]

INDICESREAL = 5000
INDICESFAKE = 2500
INDICESHUMOUR = 2500
TARGETINDEX = 31

#print(englishTrainingSet.iloc[0,0])

realRecordsData = realRecords.iloc[:INDICESREAL, 11:TARGETINDEX]
fakeRecordsData = fakeRecords.iloc[:INDICESFAKE, 11:TARGETINDEX]
humourRecordsData = humourRecords.iloc[:INDICESHUMOUR, 11:TARGETINDEX]

realTargets = realRecords.iloc[:INDICESREAL, TARGETINDEX]
fakeTargets = fakeRecords.iloc[:INDICESFAKE, TARGETINDEX]
humourTargets = humourRecords.iloc[:INDICESHUMOUR, TARGETINDEX]

subTrainingSet = pd.concat([realRecordsData, fakeRecordsData, humourRecordsData])
subTargetSet = pd.concat([realTargets, fakeTargets, humourTargets])

# testing split
realTestRecordsData = realRecords.iloc[INDICESREAL:, 11:TARGETINDEX]
fakeTestRecordsData = fakeRecords.iloc[INDICESFAKE:, 11:TARGETINDEX]
humourTestRecordsData = humourRecords.iloc[INDICESHUMOUR:, 11:TARGETINDEX]

realTestTargets = realRecords.iloc[INDICESREAL:, TARGETINDEX]
fakeTestTargets = fakeRecords.iloc[INDICESFAKE:, TARGETINDEX]
humourTestTargets = humourRecords.iloc[INDICESHUMOUR:, TARGETINDEX]

subTestTrainingSet = pd.concat([realTestRecordsData, fakeTestRecordsData, humourTestRecordsData])
subTestTargetSet = pd.concat([realTestTargets, fakeTestTargets, humourTestTargets])


# 10956 records with english text according to detectLang

#actual test
testSet = pd.read_csv("testSetAllFeatures.csv",encoding="utf8")
testSetData = testSet.iloc[:,11:TARGETINDEX]
testSetTarget = testSet.iloc[:,TARGETINDEX]

parameters = {'kernel':('linear', 'rbf','poly'), 'C':[1,5,10]}
svm = svm.SVC()

#bayes = MultinomialNB()

#bayes.fit(subTrainingSet.iloc[:,3:],subTargetSet)
#predictions = bayes.predict(testSetData.iloc[:,3:])

#classifier = GridSearchCV(svm, parameters)
#classifier.fit(subTrainingSet, subTargetSet)
#predictions = classifier.predict(subTestTrainingSet)

svm.fit(subTrainingSet,subTargetSet)
predictions = svm.predict(subTestTrainingSet)
#predictions = svm.predict(testSetData)

correct = 0
incorrect = 0
for index, row in enumerate(predictions):
    if (row == subTestTargetSet.iloc[index]):
    #if (row == testSetTarget.iloc[index]):
        correct += 1
        #print("Actual ", testSetTarget.iloc[index], ", guess was: " ,row, ", CORRECT")
        print("Actual ", subTestTargetSet.iloc[index], ", guess was: " ,row, ", CORRECT")
    else:
        incorrect += 1
        #print("Actual ", testSetTarget.iloc[index], ", guess was: " ,row, ", INCORRECT")
        print("Actual ", subTestTargetSet.iloc[index], ", guess was: " ,row, ", INCORRECT")

accuracy = (correct / (correct + incorrect)) * 100
print("accuracy = ", accuracy)
#print('RMSE = ', np.sqrt(mean_squared_error(subTestTargetSet, predictions)))
#print('RMSE = ', np.sqrt(mean_squared_error(testSetTarget, predictions)))
#print(classifier.cv_results_)
print(metrics.f1_score(subTestTargetSet,predictions,average="micro"))
print(metrics.recall_score(subTestTargetSet,predictions))
print(metrics.precision_score(subTestTargetSet,predictions))


