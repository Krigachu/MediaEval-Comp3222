import codecs
import sys

import matplotlib.pyplot as plt
import pandas as pd
import sklearn

print(sys.path)
#sys.path.append("C:/Program Files/Anaconda/envs/Coursework")
#sys.path.append("C:/Program Files/Anaconda/envs/Coursework/Lib/site-packages")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import numpy as np

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

print(trainingSet["username"].value_counts()) #some users have multiple posts in the same dataset

example_sent = """This is a sample sentence, 
                  showing off the stop words filtration."""

#stop_words = set(stopwords.words('english') +list(punctuation)) # adds punctuation to filtering
stop_words = set(stopwords.words('english'))


word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

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
