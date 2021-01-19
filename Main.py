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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

DetectorFactory.seed = 0


# nltk.download()

# nltk.download("stopwords") # downloading stopwords

# converts a tab delimited txt file into csv file.
# @param txt_file = string name of tab delimited txt file to convert into csv
# @param csv_file = string name of new csv file


def createTrainingCsv(textFile="mediaeval-2015-trainingset.txt", csvFile="training_set.csv"):
    with codecs.open(textFile, "r", encoding="utf8") as txtFileToCsv:
        with codecs.open(csvFile, 'w', encoding="utf8") as newCsvFile:
            input_txt = txtFileToCsv.readlines()
            for line in input_txt:
                record = line.replace("\t", "x0x")
                newCsvFile.write(record)
    print("Finished converting ", textFile, " into a csv")


# Shows composition of labels in training data set
def showLabelComposition(trainingSet):
    labelComposition = trainingSet["label"].value_counts()
    totalLabels = trainingSet["label"].count()
    groundTruthTypes = ["Fake", "Real", "Humor"]
    print(labelComposition)
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
    # languageTypes = trainingSet["language"].unique().sort
    languageTypes = trainingSet["language"].value_counts().index
    # print(languageTypes)

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
    polarityTypes = trainingSet["polarity"].value_counts().index  # labels

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecordsPolarities = realRecords["polarity"].value_counts()
    fakeRecordsPolarities = fakeRecords["polarity"].value_counts()
    humourRecordsPolarites = humourRecords["polarity"].value_counts()
    print(realRecordsPolarities)
    print(fakeRecordsPolarities)

    polarityComposition = trainingSet["polarity"].value_counts()
    # print(polarityComposition)
    # totalPolarityTypes = trainingSet["polarity"].count()

    # for counter, label in enumerate(polarityTypes):
    #   print(label, " ", str((polarityComposition[counter] / totalPolarityTypes) * 100), "%")

    width = 0.25
    x = np.arange(len(polarityTypes))
    # new_x = x * 2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, realRecordsPolarities, width, label='Real')
    rects2 = ax.bar(x, fakeRecordsPolarities, width, label='Fake')
    rects3 = ax.bar(x + width, humourRecordsPolarites, width, label='Humor')

    ax.set_ylabel("Occurence")
    ax.set_title("Polarity breakdown of dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(polarityTypes)
    ax.legend()

    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.savefig(fname="Data Visualisation/Polarity breakdown.png")

    plt.show()


def showSubjectivityComposition(trainingSet):
    polarityTypes = trainingSet["subjectivity"].value_counts().index  # labels

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecordsPolarities = realRecords["subjectivity"].value_counts()
    fakeRecordsPolarities = fakeRecords["subjectivity"].value_counts()
    humourRecordsPolarites = humourRecords["subjectivity"].value_counts()
    print(realRecordsPolarities)
    print(fakeRecordsPolarities)

    width = 0.25
    x = np.arange(len(polarityTypes))
    # new_x = x * 2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, realRecordsPolarities, width, label='Real')
    rects2 = ax.bar(x, fakeRecordsPolarities, width, label='Fake')
    rects3 = ax.bar(x + width, humourRecordsPolarites, width, label='Humor')

    ax.set_ylabel("Occurence")
    ax.set_title("Subjectivity breakdown of dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(polarityTypes)
    ax.legend()

    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.savefig(fname="Data Visualisation/Subjectivity breakdown.png")

    plt.show()


def showEmojiComposition(trainingSet):
    # emojiTypes = trainingSet["number of emojis"].value_counts().index # labels

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    fakeRecords = pd.concat([fakeRecords, humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecords = realRecords["number of emojis"].mean()
    fakeRecords = fakeRecords["number of emojis"].mean()
    # realRecords = realRecords["number of emojis"].sum()
    # fakeRecords = fakeRecords["number of emojis"].sum()
    print(realRecords)
    print(fakeRecords)

    emojiComposition = trainingSet["number of emojis"].value_counts()
    # print(emojiComposition)
    # totalemojiTypes = trainingSet["emoji"].count()

    # for counter, label in enumerate(emojiTypes):
    #   print(label, " ", str((emojiComposition[counter] / totalemojiTypes) * 100), "%")

    plt.figure()
    labels = ["Real", "Fake"]
    y_pos = np.arange(len(labels))
    heightData = [realRecords, fakeRecords]

    plt.bar(y_pos, heightData, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Usage')
    plt.title('Mean Emoji usages in Fake and Real posts in the english training dataset')
    plt.savefig(fname="Data Visualisation/Mean Emoji usages in Fake and Real posts in the english training dataset.png")

    plt.show()


def showMentionComposition(trainingSet):
    # emojiTypes = trainingSet["number of emojis"].value_counts().index # labels

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    fakeRecords = pd.concat([fakeRecords, humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecords = realRecords["number of mentions"].mean()
    fakeRecords = fakeRecords["number of mentions"].mean()
    # realRecords = realRecords["number of mentions"].sum()
    # fakeRecords = fakeRecords["number of mentions"].sum()
    print(realRecords)
    print(fakeRecords)

    plt.figure()
    labels = ["Real", "Fake"]
    y_pos = np.arange(len(labels))
    heightData = [realRecords, fakeRecords]

    plt.bar(y_pos, heightData, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Usage')
    plt.title('Mean @mentions usages in Fake and Real posts in the english training dataset')
    plt.savefig(
        fname="Data Visualisation/Mean @mention usages in Fake and Real posts in the english training dataset.png")

    plt.show()


def showURLComposition(trainingSet):
    # emojiTypes = trainingSet["number of emojis"].value_counts().index # labels

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    fakeRecords = pd.concat([fakeRecords, humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    # realRecords = realRecords["number of URLS"].mean()
    # fakeRecords = fakeRecords["number of URLS"].mean()
    realRecords = realRecords["number of URLS"].sum()
    fakeRecords = fakeRecords["number of URLS"].sum()
    print(realRecords)
    print(fakeRecords)

    plt.figure()
    labels = ["Real", "Fake"]
    y_pos = np.arange(len(labels))
    heightData = [realRecords, fakeRecords]

    plt.bar(y_pos, heightData, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Usage')
    plt.title('Total URL usages in Fake and Real posts in the english training dataset')
    plt.savefig(fname="Data Visualisation/Total URL usages in Fake and Real posts in the english training dataset.png")

    plt.show()


def showHashtagsComposition(trainingSet):
    # emojiTypes = trainingSet["number of emojis"].value_counts().index # labels

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    fakeRecords = pd.concat([fakeRecords, humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    # realRecords = realRecords["number of URLS"].mean()
    # fakeRecords = fakeRecords["number of URLS"].mean()
    realRecords = realRecords["number of hashtags"].sum()
    fakeRecords = fakeRecords["number of hashtags"].sum()
    print(realRecords)
    print(fakeRecords)

    plt.figure()
    labels = ["Real", "Fake"]
    y_pos = np.arange(len(labels))
    heightData = [realRecords, fakeRecords]

    plt.bar(y_pos, heightData, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Usage')
    plt.title('Total Hashtag usages in Fake and Real posts in the english training dataset')
    plt.savefig(
        fname="Data Visualisation/Total Hashtag usages in Fake and Real posts in the english training dataset.png")

    plt.show()


def showTweetComposition(trainingSet):
    # polarityTypes = trainingSet["polarity"].value_counts().index # labels
    labelTypes = ["no. exclamations", "no. questions", "no. ellipsis", "no. locations","no. disaster words","no. emojis", "no. URLS", "no. Hashtags",
                  "no. mentions"]
    print(labelTypes)

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    # fakeRecords = pd.concat([fakeRecords,humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    #realRecordsNumberOfCharacters = realRecords["character length"].mean()
    #fakeRecordsNumberOfCharacters = fakeRecords["character length"].mean()
    #humourRecordsNumberOfCharacters = humourRecords["character length"].mean()

    realRecordsNumberOfExclamations = realRecords["number of exclamations"].mean()
    fakeRecordsNumberOfExclamations = fakeRecords["number of exclamations"].mean()
    humourRecordsNumberOfExclamations = fakeRecords["number of exclamations"].mean()

    realRecordsNumberOfQuestions = realRecords["number of questions"].mean()
    fakeRecordsNumberOfQuestions = fakeRecords["number of questions"].mean()
    humourRecordsNumberOfQuestions = fakeRecords["number of questions"].mean()

    realRecordsNumberOfEllipsis = realRecords["number of ellipsis"].mean()
    fakeRecordsNumberOfEllipsis = fakeRecords["number of ellipsis"].mean()
    humourRecordsNumberOfEllipsis = fakeRecords["number of ellipsis"].mean()

    #realRecordsNumberOfWords = realRecords["word length"].mean()
    #fakeRecordsNumberOfWords = fakeRecords["word length"].mean()
    #humourRecordsNumberOfWords = fakeRecords["word length"].mean()

    realRecordsNumberOfLocations = realRecords["number of locations"].mean()
    fakeRecordsNumberOfLocations = fakeRecords["number of locations"].mean()
    humourRecordsNumberOfLocations = fakeRecords["number of locations"].mean()

    realRecordsNumberOfDisasterWords = realRecords["number of disaster words"].mean()
    fakeRecordsNumberOfDisasterWords = fakeRecords["number of disaster words"].mean()
    humourRecordsNumberOfDisasterWords = fakeRecords["number of disaster words"].mean()

    realRecordsNumberOfEmojis = realRecords["number of emojis"].mean()
    fakeRecordsNumberOfEmojis = fakeRecords["number of emojis"].mean()
    humourRecordsNumberOfEmojis = fakeRecords["number of emojis"].mean()

    realRecordsNumberOfURLS = realRecords["number of URLS"].mean()
    fakeRecordsNumberOfURLS = fakeRecords["number of URLS"].mean()
    humourRecordsNumberOfURLS = fakeRecords["number of URLS"].mean()

    realRecordsNumberOfHashtags = realRecords["number of Hashtags"].mean()
    fakeRecordsNumberOfHashtags = fakeRecords["number of Hashtags"].mean()
    humourRecordsNumberOfHashtags = fakeRecords["number of Hashtags"].mean()

    realRecordsNumberOfMentions = realRecords["number of mentions"].mean()
    fakeRecordsNumberOfMentions = fakeRecords["number of mentions"].mean()
    humourRecordsNumberOfMentions = fakeRecords["number of mentions"].mean()

    realData = [realRecordsNumberOfExclamations, realRecordsNumberOfQuestions, realRecordsNumberOfEllipsis,realRecordsNumberOfLocations,realRecordsNumberOfDisasterWords,
                realRecordsNumberOfEmojis, realRecordsNumberOfURLS, realRecordsNumberOfHashtags,
                realRecordsNumberOfMentions]
    fakeData = [fakeRecordsNumberOfExclamations, fakeRecordsNumberOfQuestions, fakeRecordsNumberOfEllipsis,fakeRecordsNumberOfLocations,fakeRecordsNumberOfDisasterWords,
                fakeRecordsNumberOfEmojis, fakeRecordsNumberOfURLS, fakeRecordsNumberOfHashtags,
                fakeRecordsNumberOfMentions]
    humourData = [humourRecordsNumberOfExclamations, humourRecordsNumberOfQuestions, humourRecordsNumberOfEllipsis,humourRecordsNumberOfLocations,humourRecordsNumberOfDisasterWords,
                  humourRecordsNumberOfEmojis, humourRecordsNumberOfURLS, humourRecordsNumberOfHashtags,
                  humourRecordsNumberOfMentions]

    # polarityComposition = trainingSet["polarity"].value_counts()
    # print(polarityComposition)
    # totalPolarityTypes = trainingSet["polarity"].count()

    # for counter, label in enumerate(polarityTypes):
    #   print(label, " ", str((polarityComposition[counter] / totalPolarityTypes) * 100), "%")

    width = 0.25
    x = np.arange(len(labelTypes))
    # new_x = x * 2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, realData, width, label='Real')
    rects2 = ax.bar(x, fakeData, width, label='Fake')
    rects3 = ax.bar(x + width, humourData, width, label='Humor')

    ax.set_ylabel("Mean Occurences")
    ax.set_title("Tweet breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labelTypes)
    ax.legend()

    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.savefig(fname="Data Visualisation/Tweet breakdown of English training set.png")

    plt.show()


def showPOSTagsComposition(trainingSet):
    # polarityTypes = trainingSet["polarity"].value_counts().index # labels
    labelTypes = ["verb count","noun count","adjective count","adverb count","pronoun count"]
    print(labelTypes)

    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    # fakeRecords = pd.concat([fakeRecords,humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    #realRecordsNumberOfCharacters = realRecords["character length"].mean()
    #fakeRecordsNumberOfCharacters = fakeRecords["character length"].mean()
    #humourRecordsNumberOfCharacters = humourRecords["character length"].mean()

    realRecordsVerbCount = realRecords["verb count"].mean()
    fakeRecordsVerbCount = fakeRecords["verb count"].mean()
    humourRecordsVerbCount = fakeRecords["verb count"].mean()

    realRecordsNounCount = realRecords["noun count"].mean()
    fakeRecordsNounCount = fakeRecords["noun count"].mean()
    humourRecordsNounCount = fakeRecords["noun count"].mean()

    realRecordsAdjectiveCount = realRecords["adjective count"].mean()
    fakeRecordsAdjectiveCount = fakeRecords["adjective count"].mean()
    humourRecordsAdjectiveCount = fakeRecords["adjective count"].mean()

    #realRecordsNumberOfWords = realRecords["word length"].mean()
    #fakeRecordsNumberOfWords = fakeRecords["word length"].mean()
    #humourRecordsNumberOfWords = fakeRecords["word length"].mean()

    realRecordsAdverbCount = realRecords["adverb count"].mean()
    fakeRecordsAdverbCount = fakeRecords["adverb count"].mean()
    humourRecordsAdverbCount = fakeRecords["adverb count"].mean()

    realRecordsPronounCount = realRecords["pronoun count"].mean()
    fakeRecordsPronounCount = fakeRecords["pronoun count"].mean()
    humourRecordsPronounCount = fakeRecords["pronoun count"].mean()


    realData = [realRecordsVerbCount,realRecordsNounCount,realRecordsAdjectiveCount,realRecordsAdverbCount,realRecordsPronounCount]
    fakeData = [fakeRecordsVerbCount,fakeRecordsNounCount,fakeRecordsAdjectiveCount,fakeRecordsAdverbCount,fakeRecordsPronounCount]
    humourData = [humourRecordsVerbCount,humourRecordsNounCount,humourRecordsAdjectiveCount,humourRecordsAdverbCount,humourRecordsPronounCount]


    # polarityComposition = trainingSet["polarity"].value_counts()
    # print(polarityComposition)
    # totalPolarityTypes = trainingSet["polarity"].count()

    # for counter, label in enumerate(polarityTypes):
    #   print(label, " ", str((polarityComposition[counter] / totalPolarityTypes) * 100), "%")

    width = 0.25
    x = np.arange(len(labelTypes))
    # new_x = x * 2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, realData, width, label='Real')
    rects2 = ax.bar(x, fakeData, width, label='Fake')
    rects3 = ax.bar(x + width, humourData, width, label='Humor')

    ax.set_ylabel("Mean Occurences")
    ax.set_title("Mean occurences of POS tags in tweet content")
    ax.set_xticks(x)
    ax.set_xticklabels(labelTypes)
    ax.legend()

    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.savefig(fname="Data Visualisation/POS tag breakdown.png")

    plt.show()


def showNumberOfWordsUsage(trainingSet):
    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    # fakeRecords = pd.concat([fakeRecords,humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecordsNumberOfWords = realRecords["word count"].mean()
    fakeRecordsNumberOfWords = fakeRecords["word count"].mean()
    humourRecordsNumberOfWords = humourRecords["word count"].mean()

    realData = realRecordsNumberOfWords
    fakeData = fakeRecordsNumberOfWords
    humourData = humourRecordsNumberOfWords

    plt.figure()
    labelTypes = ["Real", "Fake", "Humour"]
    y_pos = np.arange(len(labelTypes))
    data = [realData, fakeData, humourData]

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labelTypes)
    plt.ylabel('Mean word count')
    plt.title('Mean word count in the training dataset')

    plt.savefig(fname="Data Visualisation/Mean words in training set.png")

    plt.show()


def showNumberOfCharactersUsage(trainingSet):
    realRecords = trainingSet[trainingSet["label"] == "real"]
    fakeRecords = trainingSet[trainingSet["label"] == "fake"]
    humourRecords = trainingSet[trainingSet["label"] == "humor"]
    # fakeRecords = pd.concat([fakeRecords,humourRecords])

    # print(realRecords.iloc[:,6:9].head(10))
    # print(fakeRecords.iloc[:,6:9].head(10))

    realRecordsNumberOfWords = realRecords["character count"].mean()
    fakeRecordsNumberOfWords = fakeRecords["character count"].mean()
    humourRecordsNumberOfWords = humourRecords["character count"].mean()

    realData = realRecordsNumberOfWords
    fakeData = fakeRecordsNumberOfWords
    humourData = humourRecordsNumberOfWords

    plt.figure()
    labelTypes = ["Real", "Fake", "Humour"]
    y_pos = np.arange(len(labelTypes))
    data = [realData, fakeData, humourData]

    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labelTypes)
    plt.ylabel('Mean character count')
    plt.title('Mean character count in the training dataset')

    plt.savefig(fname="Data Visualisation/Mean character count training set.png")

    plt.show()


def detectTweetTextLanguage(row):
    try:
        lang = detect(row.iloc[1])
        languageDetected.append(lang)
    except:
        lang = "error"
        print("This row throws an error " + str(row.iloc[0]))
        # 262974742716370944	Man sandy Foreal??  ⚡⚡⚡☔☔⚡🌊🌊☁🚣⛵💡🔌🚬🚬🚬🔫🔫🔒🔒🔐🔑🔒🚪🚪🚪🔨🔨🔨🏊🏊🏊🏊🎣🎣🎣😱😰😖😫😩😤💨💨💨💨💦💦💦💧💦💥💥💥👽💩🙌🙌🙌🙌🙌🏃🏃🏃🏃🏃👫👭💏👪👪👬👭💑🙇🌕🌕🌕🌎 http://t.co/vEWVXy10	183424929	sandyA_fake_29	vtintit	Mon Oct 29 17:50:39 +0000 2012	fake
        languageDetected.append(lang)


def detectPolarityOfTweet(row):
    try:
        blob = TextBlob(row.iloc[1])
        polarityOfBlob = blob.polarity
        polarityScores.append(polarityOfBlob)

        if (polarityOfBlob == 0):
            polarityOfBlob = "neutral"
        elif (polarityOfBlob > 0):
            polarityOfBlob = "positive"
        elif (polarityOfBlob < 0):
            polarityOfBlob = "negative"
        else:
            polarityOfBlob = "error"

        polarityTweet.append(polarityOfBlob)
    except:
        blob = "error"
        print("This row throws an error " + str(row.iloc[0]))
        polarityTweet.append(blob)


def detectSubjectivityOfTweet(row):
    try:
        blob = TextBlob(row.iloc[1])
        subjectivityOfBlob = blob.subjectivity
        subjectivityScores.append(subjectivityOfBlob)

        if (subjectivityOfBlob == 0.5):
            subjectivityOfBlob = "neutral"
        elif (subjectivityOfBlob < 0.5):
            subjectivityOfBlob = "objective"
        elif (subjectivityOfBlob > 0.5):
            subjectivityOfBlob = "subjective"
        else:
            subjectivityOfBlob = "error"

        subjectivityTweet.append(subjectivityOfBlob)
    except:
        blob = "error"
        print("This row throws an error " + str(row.iloc[0]))
        subjectivityTweet.append(blob)


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
        detectSubjectivityOfTweet(row)
        detectTweetFeatures(row)

    trainingSet["language"] = languageDetected
    trainingSet["polarity"] = polarityTweet
    trainingSet["subjectivity"] = subjectivityTweet
    trainingSet["polarity score"] = polarityScores
    trainingSet["subjectivity score"] = subjectivityScores
    trainingSet["character count"] = trainingSet.iloc[:, 1].apply(lambda x: len(x))  # taken
    trainingSet['punctuation count'] = trainingSet.iloc[:, 1].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))  # taken
    trainingSet["number of exclamations"] = trainingSet.iloc[:, 1].apply(lambda x: extractExclamations(x))
    trainingSet["number of questions"] = trainingSet.iloc[:, 1].apply(lambda x: extractQuestions(x))
    trainingSet["number of ellipsis"] = trainingSet.iloc[:, 1].apply(lambda x: extractEllipsis(x))
    trainingSet["word count"] = trainingSet.iloc[:, 1].apply(lambda x: len(x.split()))  # taken
    trainingSet['noun count'] = trainingSet.iloc[:, 1].apply(lambda x: checkPosTag(x, 'noun'))
    trainingSet['verb count'] = trainingSet.iloc[:, 1].apply(lambda x: checkPosTag(x, 'verb'))
    trainingSet['adjective count'] = trainingSet.iloc[:, 1].apply(lambda x: checkPosTag(x, 'adjective'))
    trainingSet['adverb count'] = trainingSet.iloc[:, 1].apply(lambda x: checkPosTag(x, 'adverb'))
    trainingSet['pronoun count'] = trainingSet.iloc[:, 1].apply(lambda x: checkPosTag(x, 'pronoun'))
    trainingSet["number of locations"] = numberOfLocations
    trainingSet["number of disaster words"] = numberOfDisasterWords
    trainingSet["number of emojis"] = trainingSet.iloc[:, 1].apply(lambda x: extractEmojis(x))
    trainingSet["number of URLS"] = trainingSet.iloc[:, 1].apply(lambda x: extractUrlCount(x))
    trainingSet["number of Hashtags"] = trainingSet.iloc[:, 1].apply(lambda x: extractHashtags(x))
    trainingSet["number of mentions"] = trainingSet.iloc[:, 1].apply(lambda x: extractMentions(x))
    trainingSet['word density'] = trainingSet['character count'] / (trainingSet['word count'] + 1)  # taken
    trainingSet["target"] = target


def featureGeneration2(trainingSet):
    for index, row in trainingSet.iterrows():
        if row[6] == "real":
            target.append(1)
        else:
            target.append(0)


def detectTweetFeatures(row):
    tweet = row[1]

    locations = 0
    disasterWords = 0
    tweetTokens = word_tokenize(tweet)

    if row[6] == "real":
        target.append(0)
    else:
        target.append(1)

    for w in tweetTokens:
        if w not in stopWordsPunctuation:
            tokenizedTweets.append(w)
            if ((cityCorpora == w) | (countryCorpora == w) | (iso3Corpora == w.upper())).any():
                locations =+ 1
            if (naturalDisasterWordsCopora == w).any():
                disasterWords =+ 1

    numberOfLocations.append(locations)
    numberOfDisasterWords.append(disasterWords)


def extractEmojis(tweet):
    emojis = emoji.demojize(tweet)
    emojis = re.findall(emojiRegex, emojis)
    #emojisTweet = [emoji.emojize(x) for x in emojis]
    #cnt = len(emojisTweet)
    cnt = len(emojis)
    return cnt


def extractEllipsis(tweet):
    ellipsisR = re.findall(ellipsisRegex, tweet)
    cnt = (len(ellipsisR))
    return cnt


def extractQuestions(tweet):
    questions = re.findall(questionRegex, tweet)
    cnt = (len(questions))
    return cnt


def extractExclamations(tweet):
    exclamations = re.findall(exclamationRegex, tweet)
    cnt = (len(exclamations))
    return cnt

def extractMentions(tweet):
    mentions = re.findall(mentionRegex, tweet)
    cnt = (len(mentions))
    return cnt

def extractHashtags(tweet):
    hashtags = re.findall(hashtagRegex, tweet)
    cnt = (len(hashtags))
    return cnt

def extractUrlCount(tweet):
    URLs = re.findall(twitterLinkRegex, tweet)
    cnt = (len(URLs))
    return cnt


def checkPosTag(tweetTokens, tag):
    cnt = 0
    try:
        POSTaggedTokens = nltk.pos_tag(tweetTokens)
        for tuple in POSTaggedTokens:
            POStag = tuple[1]
            if POStag in pos_tags[tag]:
                cnt += 1
    except:
        pass
    return cnt

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


createTrainingCsv()
trainingSet = pd.read_csv("training_set.csv", encoding="utf8", delimiter="x0x")
#trainingSet = pd.read_csv("testset.csv", encoding="utf8", delimiter="x0x")

#createTrainingCsv("mediaeval-2015-testset.txt","testset.csv")
#testSet = pd.read_csv("testset.csv", encoding="utf8", delimiter="x0x")

pos_tags = {
    "noun" :  ["NN","NNS","NNP","NNPS"],
    "pronoun" : ["PRP","PRP$","WP","WP$"],
    "verb" : ["VB","VBD","VBG","VBN","VBP","VBZ"],
    "adjective" : ["JJ","JJR","JJS"],
    "adverb" : ["RB","RBR","RBS","WRB"]
}

locationCorpora = pd.read_csv("Cities database/worldcities.csv", encoding="utf8")
cityCorpora = locationCorpora["city"]
countryCorpora = locationCorpora["country"].drop_duplicates()
iso3Corpora = locationCorpora["iso3"].drop_duplicates()

#borrowed and slightly edited.
naturalDisasterWordsCopora = pd.Series(["tsunami", "disaster", "volcano", "tornado", "avalanche", "earthquake", "blizzard", "drought", "fire", "tremor", "storm","magma","twister", "windstorm", "heat wave", "cyclone", "fire", "flood","hailstorm", "lava", "lightning", "high-pressure", "hail", "hurricane", "seismic", "erosion", "whirlpool", "whirlwind","thunderstorm", "barometer", "gale", "blackout", "gust", "force", "low-pressure", "volt", "snowstorm", "rainstorm", "storm", "nimbus", "violent", "sandstorm", "casualty", "fatal", "fatality", "cumulonimbus", "death", "lost", "destruction", "tension", "cataclysm", "damage", "uproot", "underground", "destroy", "arsonist", "arson", "rescue", "permafrost", "disaster", "fault", "scientist", "shelter"])

# englishTrainingSet = pd.read_csv("englishTrainingSet.csv",encoding="utf8")
print("This is the size of the data ", trainingSet.shape)
# print("Size of englishTrainingSet.csv", englishTrainingSet.shape)
# print(trainingSet["username"].value_counts()) #some users have multiple posts in the same dataset

# subTrainingSet = trainingSet[0:11]
stopWords = set(stopwords.words("english"))
stopWordsPunctuation = set(stopwords.words("english") + list(punctuation))
twitterLinkRegex = "http:.*"
# twitterLinkRegex = "http: \*/\*/t.co/* | http://t.co/*"
hashtagRegex = "#([0-9]*[a-zA-Z]*)+"
mentionRegex = "@([0-9]*[a-zA-Z]*)+"
exclamationRegex = "!"
questionRegex = "\?"
ellipsisRegex = "\.{3}"
emojiRegex = r"(:[^:]*:)"

# think about emoticon regex
# need to extract named entities
# locations
# respected news agents

MAXFEATURES = 1000;


tokenizedTweets = []
languageDetected = []
polarityTweet = []
subjectivityTweet = []
polarityScores = []
subjectivityScores = []
numberOfLocations = []
numberOfDisasterWords = []
target = []

#featureGeneration(trainingSet)
#featureGeneration(testSet)

#corpus = trainingSet.iloc[:,1]
#corpusCSV = pd.Series(corpus)
#corpusCSV.to_csv("corpusCSV.csv",encoding="utf8")

#corpus = pd.read_csv("corpusCSV.csv",encoding="utf8")
#vectorizer = CountVectorizer(stop_words="english")
#X = vectorizer.fit_transform(corpus) # 27468 words
#print(vectorizer.get_feature_names()) #token_pattern=r'\b\w+\b' -> whitespace word whitespace

#bigramVectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, stop_words="english" ,max_features=MAXFEATURES)
#X = bigramVectorizer.fit_transform(corpus) # 85822 words
#print(bigramVectorizer.get_feature_names())

#print(corpus)
#print(X.toarray())
#print(len(X.toarray()[0]))

#print(X.shape)
#print(len(corpus))

#transformer = TfidfTransformer()
#tfidf = transformer.fit_transform(X)
#print(tfidf.shape)
#print(tfidf.toarray())

#print(transformer.idf_)

#tfidfVectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, stop_words="english" ,max_features=MAXFEATURES,)
#Y = tfidfVectorizer.fit_transform(corpus)
#print(Y.shape)
#print(Y.toarray())

#print(tfidfVectorizer.idf_)
#print("n_samples: %d, n_features: %d" % Y.shape)

#trainingSet["Vectors"] = Y.toarray()


# trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/testing1.csv",encoding="utf8")


#trainingSet.to_csv("trainingSetAllFeatures.csv",encoding="utf8")
trainingSet = pd.read_csv("trainingSetAllFeatures.csv",encoding="utf8")

#showLabelComposition(trainingSet)
showPOSTagsComposition(trainingSet)
showPolarityComposition(trainingSet)
showSubjectivityComposition(trainingSet)
#showNumberOfCharactersUsage(trainingSet)
#showNumberOfWordsUsage(trainingSet)
#testSet["language"] = languageDetected
#testSet["polarity"] = polarityTweet
#testSet["subjectivity"] = subjectivityTweet
#testSet["polarity score"] = polarityScores
#testSet["subjectivity score"] = subjectivityScores
#testSet["character length"] = numberOfCharacters
#testSet["number of exclamations"] = numberOfExclamations
#testSet["number of questions"] = numberOfQuestions
#testSet["number of ellipsis"] = numberOfEllipsis
#testSet["word length"] = numberOfWords
#testSet["number of locations"] = numberOfLocations
#testSet["number of disaster words"] = numberOfDisasterWords
#testSet["number of emojis"] = numberOfEmojis
#testSet["number of URLS"] = numberOfUrls
#testSet["number of Hashtags"] = numberOfHashtags
#testSet["number of mentions"] = numberOfMentions
#testSet["target"] = target

#trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/tfidfAttempt.csv",encoding="utf8")
#testSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/allLanguagesDatasetTEST.csv",encoding="utf8")
#trainingSet = pd.read_csv("testing2.csv",encoding="utf8")
# showLanguageComposition(trainingSet) #11142
# en   76.93157494994131 % english
# es   9.024373403300421 % spanish
# tl   2.2440102188773046 % tagalog
# fr   1.5397362424911967 % french
# id   1.2221224884347166 % indonesian

#languageFilter = trainingSet["language"] == "en"
#languageFilter = testSet["language"] == "en"

#englishTrainingSet = trainingSet[languageFilter].reset_index(drop=True)
#englishTrainingSet = englishTrainingSet.drop_duplicates(ignore_index=False)  # 11141

#englishTestSet = testSet[languageFilter].reset_index(drop=True)
#englishTestSet = englishTestSet.drop_duplicates(ignore_index=False)  # 11141

#trainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/testing2.csv", encoding="utf8")

# print("the shape of english records " ,str(englishTrainingSet.shape)) #10956
# print("The shape of original records " ,str(trainingSet.shape)) #10955
# englishTrainingSet = englishTrainingSet.drop_duplicates(ignore_index=True)
# print("shape of english records with dupes removed ",str(t))
# duplicatedSeries = englishTrainingSet.duplicated(keep=False)
# print(duplicatedSeries.loc[lambda x : x == True])
# print("does english training set have same size ", str(englishTrainingSet.shape))

# showLanguageComposition(englishTrainingSet)

# showLabelComposition(englishTrainingSet)
# showPolarityComposition(englishTrainingSet)
# showEmojiComposition(englishTrainingSet)
# showMentionComposition(englishTrainingSet)
# showURLComposition(englishTrainingSet)
# showHashtagsComposition(englishTrainingSet)
# showTweetComposition(trainingSet)
# showNumberOfWordsUsage(englishTrainingSet)
# showNumberOfCharactersUsage(englishTrainingSet)


# englishTrainingSet
# print(englishTrainingSet["number of emojis"].value_counts())
# index = englishTrainingSet["number of emojis"] == 16
# print(englishTrainingSet[index])


#englishTrainingSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/tfidfAttemptEnglish.csv",encoding="utf8")
#englishTestSet.to_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/englishTrainingSet.csv",encoding="utf8")


#englishTrainingSet = pd.read_csv(path_or_buf="D:/Work/Uni work/Comp3222 - MLT/CW/comp3222-mediaeval/englishTrainingSet.csv",encoding="utf8")
#print(englishTrainingSet["label"].value_counts())
#realRecords = englishTrainingSet[englishTrainingSet["label"] == "real"]
#fakeRecords = englishTrainingSet[englishTrainingSet["label"] == "fake"]
#humourRecords = englishTrainingSet[englishTrainingSet["label"] == "humor"]

realRecords = trainingSet[trainingSet["label"] == "real"]
fakeRecords = trainingSet[trainingSet["label"] == "fake"]
humourRecords = trainingSet[trainingSet["label"] == "humor"]

INDICESREAL = 4000
INDICESFAKE = 3000
INDICESHUMOUR = 1000

realRecordsData = realRecords.iloc[:INDICESREAL, 10:23]
fakeRecordsData = fakeRecords.iloc[:INDICESFAKE, 10:23]
humourRecordsData = humourRecords.iloc[:INDICESHUMOUR, 10:23]

realTargets = realRecords.iloc[:INDICESREAL, 23]
fakeTargets = fakeRecords.iloc[:INDICESFAKE, 23]
humourTargets = humourRecords.iloc[:INDICESHUMOUR, 23]

subTrainingSet = pd.concat([realRecordsData, fakeRecordsData, humourRecordsData])
subTargetSet = pd.concat([realTargets, fakeTargets, humourTargets])

# testing split
realTestRecordsData = realRecords.iloc[INDICESREAL:, 10:23]
fakeTestRecordsData = fakeRecords.iloc[INDICESFAKE:, 10:23]
humourTestRecordsData = humourRecords.iloc[INDICESHUMOUR:, 10:23]

realTestTargets = realRecords.iloc[INDICESREAL:, 23]
fakeTestTargets = fakeRecords.iloc[INDICESFAKE:, 23]
humourTestTargets = humourRecords.iloc[INDICESHUMOUR:, 23]

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


# making dumb linear regression machine
svm = svm.SVC()
svm.fit(subTrainingSet, subTargetSet)
predictions = svm.predict(subTestTrainingSet)
correct = 0
incorrect = 0
for index, row in enumerate(predictions):
    if (row == subTestTargetSet.iloc[index]):
        correct += 1
    else:
        incorrect += 1

accuracy = (correct / (correct + incorrect)) * 100
print("accuracy = ", accuracy)
# linearRegressionPredictions = linearRegression.predict(subTestingSet)
# print('RMSE = ', np.sqrt(mean_squared_error(subTestingSetTarget, linearRegressionPredictions)))
