import codecs
import sys

import matplotlib.pyplot as plt
import pandas as pd
import sklearn

print(sys.path)
#sys.path.append("C:/Program Files/Anaconda/envs/Coursework")
sys.path.append("C:/Program Files/Anaconda/envs/Coursework/Lib/site-packages")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import numpy as np
from langdetect import detect
from langdetect import detect_langs

#nltk.download("stopwords") # downloading stopwords

# converts a tab delimited txt file into csv file.
# @param txt_file = string name of tab delimited txt file to convert into csv
# @param csv_file = string name of new csv file


def createTrainingCsv(textFile="mediaeval-2015-trainingset.txt", csvFile="training_set.csv"):
    with codecs.open(textFile, "r", encoding="utf8") as txtFileToCsv:
        with codecs.open(csvFile, 'w', encoding="utf8") as newCsvFile:
            input_txt = textFile.readlines(txtFileToCsv)
            for line in input_txt:
                # record = line.replace('\t',',')
                # csvFile.write(record)
                newCsvFile.write(line)
    print("Finished converting ", textFile, " into a csv")


# Shows composition of labels in training data set
def showLabelComposition():
    labelComposition = trainingSet["label"].value_counts()
    totalLabels = trainingSet["label"].count()
    groundTruthTypes = ["Fake", "Real", "Humor"]
    # print(labelComposition)
    for counter, label in enumerate(groundTruthTypes):
        print(label, " ", str((labelComposition[counter] / totalLabels) * 100), "%")

    plt.figure(1)
    plt.pie(labelComposition, labels=groundTruthTypes)
    plt.title("The label composition of Training dataset")
    plt.savefig(fname="Data Visualisation/Label composition.png")
    plt.show()

def showLanguageComposition():
    languageComposition = trainingSet["language"].value_counts()
    print(languageComposition)
    totalLanguageTypes = trainingSet["language"].count()
    #languageTypes = trainingSet["language"].unique().sort
    languageTypes = trainingSet["language"].value_counts().index
    #print(languageTypes)

    for counter, label in enumerate(languageTypes):
        print(label, " ", str((languageComposition[counter] / totalLanguageTypes) * 100), "%")

    plt.figure(1)
    plt.pie(languageComposition,
            labels=languageTypes
             )
    plt.title("The label composition of Training dataset")
    plt.savefig(fname="Data Visualisation/Language composition.png")
    plt.show()


# TO DO:
# CLEAN DATASET
# remove dupes
#
# PRE-PROCESSING DATASET
# scikit has count vectorizer
# need to decide between n-grams, bag of words, tf-idf, POS-tagging

print("Hello World")
print(sklearn.__version__)
# create_training_csv() don't even need this smh
trainingSet = pd.read_csv("training_set.csv", encoding="utf8", delimiter="\t")
print(len(trainingSet))
print(len(trainingSet.drop_duplicates()))
duplicatedSeries = trainingSet.duplicated()
print(duplicatedSeries.loc[lambda x : x == True])

#print(trainingSet["username"].value_counts()) #some users have multiple posts in the same dataset

#stop_words = set(stopwords.words('english') +list(punctuation)) # adds punctuation to filtering

#subTrainingSet = trainingSet[0:11]
stop_words = set(stopwords.words('english'))
tokenizedTweets = []
languageDetected = []


for index,row in trainingSet.iterrows():
    tweetTokens = word_tokenize(row[1])
    try:
        lang = detect(row.iloc[1])
        languageDetected.append(lang)
    except:
        lang = "error"
        print("This row throws an error" + str(row.iloc[0]))
        languageDetected.append(lang)

    for w in tweetTokens:
        if w not in stop_words:
            tokenizedTweets.append(w)

corpus = set(tokenizedTweets)
print(corpus)
print(len(corpus))
trainingSet["language"] = languageDetected

showLanguageComposition()


#263004702072004608	one of these are dupes.
#263030750415319040



# showLabelComposition()


# dumb training and testing dataset
# subTrainingSet = trainingSet.iloc[:99,:6]
# subTestingSet = trainingSet.iloc[100:199,:6]
# print(subTrainingSet)
# print(subTestingSet)

# subTrainingSetTarget = trainingSet.iloc[:99,6]
# subTestingSetTarget = trainingSet.iloc[100:199,6]


# making dumb linear regression machine
# linearRegression = LinearRegression()
# linearRegression.fit(subTrainingSet,subTrainingSetTarget)
# linearRegressionPredictions = linearRegression.predict(subTestingSet)
# print('RMSE = ', np.sqrt(mean_squared_error(subTestingSetTarget, linearRegressionPredictions)))
