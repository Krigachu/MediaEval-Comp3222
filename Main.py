import sklearn
import csv
import codecs
import pandas as pd


# converts a tab delimited txt file into csv file.
# @param txt_file = string name of tab delimited txt file to convert into csv
# @param csv_file = string name of new csv file
def create_training_csv(txt_file="mediaeval-2015-trainingset.txt", csv_file="training_set.csv"):
    with codecs.open(txt_file,"r",encoding="utf8") as textFile:
        with codecs.open(csv_file, 'w',encoding="utf8") as csvFile:
            input_txt = textFile.readlines(textFile)
            for line in input_txt:
                #record = line.replace('\t',',')
                #csvFile.write(record)
                csvFile.write(line)
    print("Finished converting ", txt_file," into a csv")



print("Test")

print("Hello World")
print(sklearn.__version__)
#create_training_csv() don't even need this smh
trainingSet = pd.read_csv("training_set.csv",encoding="utf8",delimiter="\t")

#print(trainingSet["tweetText"])

#Shows composition of labels in training data set
print(trainingSet["label"].value_counts())

