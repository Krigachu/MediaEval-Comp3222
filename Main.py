import codecs
import sys

import matplotlib.pyplot as plt
import pandas as pd
import re
import sklearn

print(sys.path)
#sys.path.append("C:/Program Files/Anaconda/envs/Coursework")
sys.path.append("C:/Program Files/Anaconda/envs/Coursework/Lib/site-packages")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

import numpy as np
from langdetect import detect, DetectorFactory
#from langdetect import detect_langs
from textblob import TextBlob
DetectorFactory.seed = 0

#nltk.download("stopwords") # downloading stopwords

# converts a tab delimited txt file into csv file.
# @param txt_file = string name of tab delimited txt file to convert into csv
# @param csv_file = string name of new csv file


def createTrainingCsv(textFile="mediaeval-2015-trainingset.txt", csvFile="training_set.csv"):
    with codecs.open(textFile, "r", encoding="utf8") as txtFileToCsv:
        with codecs.open(csvFile, 'w', encoding="utf8") as newCsvFile:
            input_txt = txtFileToCsv.readlines()
            for line in input_txt:
                record = line.replace("\t","x0x")
                newCsvFile.write(record)
    print("Finished converting ", textFile, " into a csv")


# Shows composition of labels in training data set
def showLabelComposition(trainingSet):
    labelComposition = trainingSet["label"].value_counts()
    totalLabels = trainingSet["label"].count()
    groundTruthTypes = ["Fake", "Real", "Humor"]
    # print(labelComposition)
    for counter, label in enumerate(groundTruthTypes):
        print(label, " ", str((labelComposition[counter] / totalLabels) * 100), "%")

    plt.figure()
    plt.pie(labelComposition, labels=groundTruthTypes)
    plt.title("The label composition of Training dataset")
    plt.savefig(fname="Data Visualisation/Label composition.png")
    plt.show()

def showLanguageComposition(trainingSet):
    languageComposition = trainingSet["language"].value_counts()
    print(languageComposition)
    totalLanguageTypes = trainingSet["language"].count()
    #languageTypes = trainingSet["language"].unique().sort
    languageTypes = trainingSet["language"].value_counts().index
    #print(languageTypes)

    for counter, label in enumerate(languageTypes):
        print(label, " ", str((languageComposition[counter] / totalLanguageTypes) * 100), "%")

    plt.figure()
    plt.pie(languageComposition,
            labels=languageTypes
             )
    plt.title("The language composition of Training dataset")
    plt.savefig(fname="Data Visualisation/Language composition.png")
    plt.show()

def showPolarityComposition(trainingSet):
    polarityTypes = trainingSet["polarity"].value_counts().index # labels

    realRecords = trainingSet[trainingSet["label"] =="real"]
    fakeRecords = trainingSet[trainingSet["label"]=="fake"]
    humourRecords = trainingSet[trainingSet["label"]=="humor"]
    fakeRecords = pd.concat([fakeRecords,humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecordsPolarities = realRecords["polarity"].value_counts()
    fakeRecordsPolarities = fakeRecords["polarity"].value_counts()
    print(realRecordsPolarities)
    print(fakeRecordsPolarities)


    polarityComposition = trainingSet["polarity"].value_counts()
    #print(polarityComposition)
    #totalPolarityTypes = trainingSet["polarity"].count()

    #for counter, label in enumerate(polarityTypes):
    #   print(label, " ", str((polarityComposition[counter] / totalPolarityTypes) * 100), "%")

    width = 0.35
    x= np.arange(len(polarityTypes))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, realRecordsPolarities, width, label='Real')
    rects2 = ax.bar(x + width / 2, fakeRecordsPolarities, width, label='Fake')

    ax.set_ylabel("Occurences")
    ax.set_title("Breakdown of polarities in fake and real posts")
    ax.set_xticks(x)
    ax.set_xticklabels(polarityTypes)
    ax.legend()

    #autolabel(rects1)
    #autolabel(rects2)

    fig.tight_layout()
    plt.savefig(fname="Data Visualisation/Polarity composition of Fake and Real English training set.png")

    plt.show()


def detectTweetTextLanguage(row):
    try:
        lang = detect(row.iloc[1])
        languageDetected.append(lang)
    except:
        lang = "error"
        print("This row throws an error " + str(row.iloc[0]))
        #262974742716370944	Man sandy Foreal??  ⚡⚡⚡☔☔⚡🌊🌊☁🚣⛵💡🔌🚬🚬🚬🔫🔫🔒🔒🔐🔑🔒🚪🚪🚪🔨🔨🔨🏊🏊🏊🏊🎣🎣🎣😱😰😖😫😩😤💨💨💨💨💦💦💦💧💦💥💥💥👽💩🙌🙌🙌🙌🙌🏃🏃🏃🏃🏃👫👭💏👪👪👬👭💑🙇🌕🌕🌕🌎 http://t.co/vEWVXy10	183424929	sandyA_fake_29	vtintit	Mon Oct 29 17:50:39 +0000 2012	fake
        languageDetected.append(lang)

def detectPolarityOfTweet(row):
    try:
        blob = TextBlob(row.iloc[1])
        polarityOfBlob = blob.polarity

        if (polarityOfBlob == 0):
            polarityOfBlob ="neutral"
        elif(polarityOfBlob > 0):
            polarityOfBlob ="positive"
        elif(polarityOfBlob < 0):
            polarityOfBlob ="negative"
        else:
            polarityOfBlob ="error"

        polarityTweet.append(polarityOfBlob)
    except:
        blob = "error"
        print("This row throws an error " + str(row.iloc[0]))
        polarityTweet.append(blob)

# TO DO:

# PRE-PROCESSING DATASET:
# remove dupes - DONE
# scikit has count vectorizer
# need to decide between n-grams, bag of words, tf-idf, POS-tagging

# MAKING FEATURES:
# language detection - DONE
# has URL?
# number of URLS
# retweet?

# ML ALGO:
#


def featureGeneration(trainingSet):
    for index, row in trainingSet.iterrows():
        tweetTokens = word_tokenize(row[1])
        detectTweetTextLanguage(row)
        detectPolarityOfTweet(row)
        detectLengthOfTweet(row)

        for w in tweetTokens:
            if w not in stopWords:
                tokenizedTweets.append(w)


def detectLengthOfTweet(row):
    numberOfCharacters.append(len(row[1]))
    tweetTokens = word_tokenize(row[1])
    URLs = re.findall(twitterLinkRegex,row[1])
    hashtags = re.findall(hashtagRegex,row[1])
    mentions = re.findall(mentionRegex,row[1])
    for w in tweetTokens:
        if w not in stopWordsPunctuation:
            tokenizedTweets.append(w)

    if(len(URLs)==None):
        URLs = 0
    if (len(hashtags)==None):
        hashtags = 0
    if (len(mentions)==None):
        hashtags = 0

    numberOfWords.append(len(tweetTokens))
    numberOfUrls.append(len(URLs))
    numberOfHashtags.append(len(hashtags))
    numberOfMentions.append(len(mentions))


def printFirst7Records():
    with codecs.open("training_set.csv", "r", encoding="utf8") as txtFileToCsv:
        c = 0;
        for line in txtFileToCsv.readlines():
            record = line.split(",")
            print(record)
            print(len(record))
            if c == 8:
                break
            c += 1



createTrainingCsv() # don't even need this smh
# #printFirst7Records()
trainingSet = pd.read_csv("training_set.csv", encoding="utf8",delimiter="x0x")
#englishTrainingSet = pd.read_csv("englishTrainingSet.csv",encoding="utf8")
print("This is the size of the data ", trainingSet.shape)
#print("Size of englishTrainingSet.csv", englishTrainingSet.shape)
#print(trainingSet["username"].value_counts()) #some users have multiple posts in the same dataset

#subTrainingSet = trainingSet[0:11]
stopWords = set(stopwords.words("english"))
stopWordsPunctuation = set(stopwords.words("english") + list(punctuation))
twitterLinkRegex = "http:.*"
#twitterLinkRegex = "http: \*/\*/t.co/* | http://t.co/*"
hashtagRegex = "#([0-9]*[a-zA-Z]*)*"
mentionRegex = "@([0-9]*[a-zA-Z]*)*"



tokenizedTweets = []
languageDetected = []
polarityTweet = []
numberOfCharacters = []
numberOfWords = []
numberOfUrls = []
numberOfHashtags = []
numberOfMentions = []

featureGeneration(trainingSet)

corpus = set(tokenizedTweets)
print(corpus)
print(len(corpus))

#trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/testing1.csv",encoding="utf8")
trainingSet["language"] = languageDetected
trainingSet["polarity"] = polarityTweet
trainingSet["character length"] = numberOfCharacters
trainingSet["word length"] = numberOfWords
trainingSet["number of URLS"] = numberOfUrls
trainingSet["number of Hashtags"] = numberOfHashtags
trainingSet["number of mentions"] = numberOfMentions
#trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/testing2.csv",encoding="utf8")

showLanguageComposition(trainingSet) #11142
#en   76.93157494994131 % english
#es   9.024373403300421 % spanish
#tl   2.2440102188773046 % tagalog
#fr   1.5397362424911967 % french
#id   1.2221224884347166 % indonesian

languageFilter = trainingSet["language"] == "en"

englishTrainingSet = trainingSet[languageFilter].reset_index(drop=True)
englishTrainingSet = englishTrainingSet.drop_duplicates(ignore_index=False) #11141
trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/testing2.csv",encoding="utf8")
showLabelComposition(englishTrainingSet)

#print("the shape of english records " ,str(englishTrainingSet.shape)) #10956
#print("The shape of original records " ,str(trainingSet.shape)) #10955
#englishTrainingSet = englishTrainingSet.drop_duplicates(ignore_index=True)
#print("shape of english records with dupes removed ",str(t))
#duplicatedSeries = englishTrainingSet.duplicated(keep=False)
#print(duplicatedSeries.loc[lambda x : x == True])
#print("does english training set have same size ", str(englishTrainingSet.shape))

#showLanguageComposition(englishTrainingSet)

showPolarityComposition(englishTrainingSet)
#englishTrainingSet
print(englishTrainingSet["number of URLS"].value_counts())
englishTrainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/englishTrainingSet.csv",encoding="utf8")

#10956 records with english text according to detectLang

#showLabelComposition(englishTrainingSet)
#Fake   47.09424322598303 % aka ~ 62.68%
#Real   37.32323693093696 %
#Humor   15.58251984308001 %

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
