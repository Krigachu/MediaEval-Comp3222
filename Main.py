import sklearn
import csv
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# converts a tab delimited txt file into csv file.
# @param txt_file = string name of tab delimited txt file to convert into csv
# @param csv_file = string name of new csv file
from sklearn.metrics import mean_squared_error


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
    #print(labelComposition)
    for counter,label in enumerate(groundTruthTypes):
        print(label," ", str((labelComposition[counter] / totalLabels) * 100),"%" )

    plt.figure(1)
    plt.pie(labelComposition, labels=groundTruthTypes)
    plt.title("The label composition of Training dataset")
    plt.savefig(fname="Data Visualisation/Label composition.png")
    plt.show()



print("Hello World")
print(sklearn.__version__)
# create_training_csv() don't even need this smh
trainingSet = pd.read_csv("training_set.csv", encoding="utf8", delimiter="\t")
#showLabelComposition()

#dumb training and testing dataset
subTrainingSet = trainingSet.iloc[:99,:6]
subTestingSet = trainingSet.iloc[100:199,:6]
print(subTrainingSet)
print(subTestingSet)

subTrainingSetTarget = trainingSet.iloc[:99,6]
subTestingSetTarget = trainingSet.iloc[100:199,6]


#making dumb linear regression machine
linearRegression = LinearRegression()
linearRegression.fit(subTrainingSet,subTrainingSetTarget)
linearRegressionPredictions = linearRegression.predict(subTestingSet)
print('RMSE = ', np.sqrt(mean_squared_error(subTestingSetTarget, linearRegressionPredictions)))


