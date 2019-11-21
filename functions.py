"""
We will re-train text blob NaiveBayesClassifier
to predict prop of political sentment on arabic 
.. i tried hard to make it easy as drinking water and usefull as i can so i hope this tiny thing can help making this world better one day..
"""
"""
STEP -1 
we will define training examples in json file i will attach in this repo
"""




from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
import csv
import json
import _pickle as cPickle
class Analyzer:
    # __ parse data files and put every line polarity and text in tuple then put it in a big list to feed our model with it
    def GetDataFromFiles(self, _list):
        files = [f for f in _list]
        self.data_list = []
        for f in files:
            with open(f, 'r') as f:
                csv_reader = csv.reader(f, delimiter=',')
                for row in csv_reader:
                    line_in_list = row[0].split('\t')
                    ready_tuple = (line_in_list[1], line_in_list[0])
                    self.data_list.append(ready_tuple)
        print(str(len(self.data_list))+'____lines')
        return self.data_list[0:10000]

    def ReadyModel(self, files_list):
        self.data = self.GetDataFromFiles(files_list)
        self.model = NaiveBayesClassifier(self.data, format="json")
        print('feeding_done')
        return self.model


model = Analyzer()
cl = model.ReadyModel(['train_Arabic_tweets_negative_20190413.csv',
                       'train_Arabic_tweets_positive_20190413.csv'])
test = [('شيئ سئ'), ('مصر حلوه')]
save_training = open('saved_training.pickle', 'wb')
cPickle.dump(cl, save_training)
save_training.close()

print(cl.classify("وحش جدا"))
