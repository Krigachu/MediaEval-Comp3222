import codecs
import sys

import matplotlib.pyplot as plt
import pandas as pd

print(sys.path)
sys.path.append("C:/Program Files/Anaconda/envs/Coursework/Lib/site-packages") # Replace with own path to libraries


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

def createTrainingCsv(textFile="mediaeval-2015-trainingset.txt", csvFile="training_set.csv"):
    with codecs.open(textFile, "r", encoding="utf8") as txtFileToCsv:
        with codecs.open(csvFile, 'w', encoding="utf8") as newCsvFile:
            input_txt = txtFileToCsv.readlines()
            for line in input_txt:
                record = line.replace("\t", "x0x")
                newCsvFile.write(record)
    print("Finished converting ", textFile, " into a csv")

def createTestingCsv(textFile="mediaeval-2015-testset.txt", csvFile="test_set.csv"):
    with codecs.open(textFile, "r", encoding="utf8") as txtFileToCsv:
        with codecs.open(csvFile, 'w', encoding="utf8") as newCsvFile:
            input_txt = txtFileToCsv.readlines()
            for line in input_txt:
                record = line.replace("\t", "x0x")
                newCsvFile.write(record)
    print("Finished converting ", textFile, " into a csv")

def labelGeneration(trainingSet):
    target = []
    for index, row in trainingSet.iterrows():
        if row[6] == "real":
            target.append(0)
        else:
            target.append(1)

    trainingSet["target"] = target

createTrainingCsv()
createTestingCsv()

fullSet = pd.read_csv("training_set.csv", encoding="utf8", delimiter="x0x")
fullTestSet = pd.read_csv("test_set.csv", encoding="utf8", delimiter="x0x")

labelGeneration(fullSet)
labelGeneration(fullTestSet)

fullSet.to_csv(path_or_buf="labelledTrainingSet.csv",encoding="utf8")
fullTestSet.to_csv("labelledTestSet.csv",encoding="utf8")

trainingSet = pd.read_csv("labelledTrainingSet.csv",encoding="utf8")

realRecords = trainingSet[trainingSet["label"] == "real"]
fakeRecords = trainingSet[trainingSet["label"] == "fake"]
humourRecords = trainingSet[trainingSet["label"] == "humor"]

INDICESREAL = 4500 #5009 MAX
INDICESFAKE = 2250  #6481 MAX
INDICESHUMOUR = 2250 #2633 MAX
TARGETINDEX = 8
MAXFEATURES = 5000

#training split
realRecordsData = realRecords.iloc[:INDICESREAL, :]
fakeRecordsData = fakeRecords.iloc[:INDICESFAKE, :]
humourRecordsData = humourRecords.iloc[:INDICESHUMOUR, :]

realTargets = realRecords.iloc[:INDICESREAL, TARGETINDEX]
fakeTargets = fakeRecords.iloc[:INDICESFAKE, TARGETINDEX]
humourTargets = humourRecords.iloc[:INDICESHUMOUR, TARGETINDEX]

subTrainingSet = pd.concat([realRecordsData, fakeRecordsData, humourRecordsData]).reset_index()
subTargetSet = pd.concat([realTargets, fakeTargets, humourTargets]).reset_index()

# testing split
realTestRecordsData = realRecords.iloc[INDICESREAL:, :]
fakeTestRecordsData = fakeRecords.iloc[INDICESFAKE:, :]
humourTestRecordsData = humourRecords.iloc[INDICESHUMOUR:, :]

realTestTargets = realRecords.iloc[INDICESREAL:, TARGETINDEX]
fakeTestTargets = fakeRecords.iloc[INDICESFAKE:, TARGETINDEX]
humourTestTargets = humourRecords.iloc[INDICESHUMOUR:, TARGETINDEX]

subTestTrainingSet = pd.concat([realTestRecordsData, fakeTestRecordsData, humourTestRecordsData]).reset_index()
subTestTargetSet = pd.concat([realTestTargets, fakeTestTargets, humourTestTargets]).reset_index()


tfidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, stop_words="english" ,max_features=MAXFEATURES)
trainTFIDF = tfidf.fit_transform(subTrainingSet.tweetText)
#print("n_samples: %d, n_features: %d" % trainTFIDF.shape)

trueTestSet = pd.read_csv("labelledTestSet.csv",encoding="utf8")
print("Label breakdown: ", trueTestSet["label"].value_counts())

testTFIDF = tfidf.transform(subTestTrainingSet.tweetText)
trueTestTFIDF = tfidf.transform(trueTestSet.tweetText)


bayes = MultinomialNB()
bayes.fit(trainTFIDF, subTrainingSet.target)
#predictionsBayes = bayes.predict(testTFIDF)
parameters = {"alpha":[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
#gridSearchCV = GridSearchCV(bayes,parameters)
#gridSearchCV.fit(trainTFIDF,subTrainingSet.target)
#predictionsBayes = gridSearchCV.predict(testTFIDF)
predictionsBayes = bayes.predict(trueTestTFIDF)


def accuracyOfAlgo(predictions):
    correct = 0
    incorrect = 0
    for index, row in enumerate(predictions):
        if (row == trueTestSet.target[index]):
        #if (row == subTestTargetSet.target[index]):
            correct += 1
            print("Actual ", trueTestSet.target[index], " guess was: " ,row, " CORRECT")
            #print("Actual ", subTestTargetSet.target[index], " guess was: ", row, " CORRECT")
        else:
            incorrect += 1
            print("Actual ", trueTestSet.target[index], " guess was: " ,row, " INCORRECT")
            #print("Actual ", subTestTargetSet.target[index], " guess was: ", row, " INCORRECT")
    accuracy = (correct / (correct + incorrect)) * 100
    print("accuracy = ", accuracy)


#accuracyOfAlgo(predictions)
accuracyOfAlgo(predictionsBayes)
#accuracyOfAlgo(predictionsLR)



#print(metrics.f1_score(subTestTargetSet.target,predictions,average="micro"))
#print(metrics.f1_score(subTestTargetSet.target,predictionsBayes,average="micro"))
print(metrics.f1_score(trueTestSet.target,predictionsBayes,average="micro"))
#print(metrics.f1_score(subTestTargetSet.target,predictionsLR,average="micro"))

#print(metrics.recall_score(subTestTargetSet.target,predictions))
#print(metrics.recall_score(subTestTargetSet.target,predictionsBayes))
print(metrics.recall_score(trueTestSet.target,predictionsBayes))
#print(metrics.recall_score(subTestTargetSet.target,predictionsLR))

#print(metrics.precision_score(subTestTargetSet.target,predictions))
#print(metrics.precision_score(subTestTargetSet.target,predictionsBayes))
print(metrics.precision_score(trueTestSet.target,predictionsBayes))
#print(metrics.precision_score(subTestTargetSet.target,predictionsLR))
#print('RMSE = ', np.sqrt(mean_squared_error(subTestTargetSet, predictions)))


def showTrueF1Scores():
    realData = metrics.f1_score(trueTestSet.target,predictionsBayes,average="micro")
    fakeData = metrics.recall_score(trueTestSet.target,predictionsBayes)
    humourData = metrics.precision_score(trueTestSet.target,predictionsBayes)


    plt.figure()
    labelTypes = ["F1 score", "Recall score", "Precision score"]
    y_pos = np.arange(len(labelTypes))
    data = [realData, fakeData, humourData]

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labelTypes)
    plt.ylabel('Score')
    plt.title('Final scores of Multinomial Naive Bayes')

    plt.savefig(fname="Data Visualisation/final F1 scores.png")

    plt.show()

def showF1Scores():
    realData = metrics.f1_score(subTestTargetSet.target,predictions,average="micro")
    fakeData = metrics.f1_score(subTestTargetSet.target,predictionsBayes,average="micro")
    humourData = metrics.f1_score(subTestTargetSet.target,predictionsLR,average="micro")


    plt.figure()
    labelTypes = ["SVC", "Bayes", "Logistic Reg."]
    y_pos = np.arange(len(labelTypes))
    data = [realData, fakeData, humourData]

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labelTypes)
    plt.ylabel('Score')
    plt.title('F1 scores of ML algorithms')

    plt.savefig(fname="Data Visualisation/F1 scores training.png")

    plt.show()


def showRecallScores():
    realData = metrics.recall_score(subTestTargetSet.target,predictions)
    fakeData = metrics.recall_score(subTestTargetSet.target,predictionsBayes)
    humourData = metrics.recall_score(subTestTargetSet.target,predictionsLR)

    plt.figure()
    labelTypes = ["SVC", "Bayes", "Logistic Reg."]
    y_pos = np.arange(len(labelTypes))
    data = [realData, fakeData, humourData]

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labelTypes)
    plt.ylabel('Score')
    plt.title('Recall scores of ML algorithms')

    plt.savefig(fname="Data Visualisation/Recall scores training.png")

    plt.show()


def showPrecisionScores():
    realData = metrics.precision_score(subTestTargetSet.target,predictions)
    fakeData = metrics.precision_score(subTestTargetSet.target,predictionsBayes)
    humourData = metrics.precision_score(subTestTargetSet.target,predictionsLR)

    plt.figure()
    labelTypes = ["SVC", "Bayes", "Logistic Reg."]
    y_pos = np.arange(len(labelTypes))
    data = [realData, fakeData, humourData]

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labelTypes)
    plt.ylabel('Score')
    plt.title('Precision scores of ML algorithms')

    plt.savefig(fname="Data Visualisation/Precision scores training.png")

    plt.show()

#showF1Scores()
#showRecallScores()
#showPrecisionScores()
#print(gridSearchCV.best_params_)
showTrueF1Scores()
