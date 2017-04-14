from csv import DictReader, DictWriter

import numpy as np
from numpy import array

import time
import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
import nltk
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle

#kfolds
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_predict

#tokenizer
nltk.download('punkt')

#Stemmer
nltk.download("stopwords")
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer

#Lemmatizer
from nltk.stem.wordnet import WordNetLemmatizer


kTARGET_FIELD = 'Label'
kTEXT_FIELD = ['Top1', 'Top2','Top3','Top4','Top5']


class Featurizer:
    def __init__(self):
    
        #self.vectorizer = CountVectorizer()
        #self.vectorizer = CountVectorizer(ngram_range=(1,2)) #.64 #original examples
        #self.vectorizer = TfidfVectorizer(ngram_range=(1,2)) #.659 #original examples #.70 submission
        #self.vectorizer = TfidfVectorizer() #.658 #stemmed words #.666 submission
        #self.vectorizer = TfidfVectorizer(tokenizer = tokenize_stem, ngram_range=(1,2)) #.667 #submission .50
        #self.vectorizer = FeatureUnion([("tfidf",TfidfVectorizer(ngram_range=(1,2), stop_words = 'english')), ("stem", TfidfVectorizer(tokenizer = tokenize_stem, stop_words = 'english'))])
        self.vectorizer = FeatureUnion([("tfidf",TfidfVectorizer(ngram_range=(1,2))), ("stem", TfidfVectorizer(tokenizer = tokenize_stem, stop_words = 'english'))]) #.70 #submission .49
        

    def train_feature(self, examples):
        #print ("train,", examples[0])
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))
    
               
stemmer = SnowballStemmer("english", ignore_stopwords=True)
lemmaer = WordNetLemmatizer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def lemma_tokens(tokens, stemmer):
    lemmed = []
    for item in tokens:
        lemmed.append(lemmaer.lemmatize(item))
    return lemmed

def tokenize_lemma(text):
    tokens = nltk.word_tokenize(text)
    lems = lemma_tokens(tokens, lemmaer)
    return lems

    
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()                 
    argparser.add_argument("--limit", help="Limit size of train data to utelize",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kfolds", help="Number of k-folds for cross validation",
                           type=int, default=0, required=False)
    argparser.add_argument("--save", help="will save predictions file",
                           dest='saveFile', action ='store_true', required=False)
    argparser.set_defaults(saveFile=False)
    args = argparser.parse_args()

    print("saveFile: {}".format(args.saveFile))

    # Cast to list to keep it all in memory
    #data = list(DictReader(open("../data/Combined_News_DJIA.csv", 'r')))
    
    data = pd.read_csv('../data/Combined_News_DJIA.csv')
    data.head()
    
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']

    print("len(train) = {}".format(len(train)))
    print("len(test) = {}".format(len(test)))

    feat = Featurizer()

    labels = []
    
    ######
    #create set of potential labels from data (useful if we get into multiclass or switch from 0/1 to true/false
    ######
    for item in train[kTARGET_FIELD]:
        if not item in labels:
            labels.append(item)

    print("Label set: {}".format(str(labels)))
    
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    
    ######
    #shuffle the training data
    ######
    #print("preshuffle: {}".format(train.head(5)))
    #random.shuffle(train) #errors out
    #train.sample(frac=1) #didn't shuffle #top suggestion from: http://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    train = shuffle(train) # also potential suggestion from same stackoverflow: 29576430
    #print("postshuffle: {}".format(train.head(5)))
    
    if args.limit < 0:
        args.limit = len(train)

    if args.kfolds > 0 :
        print ("folds: %i " % (args.kfolds))
        x_train_list = [x[kTEXT_FIELD] for x in train[:args.limit]]
        
        x_train = feat.train_feature(x_train_list)
        y_train = array(list(labels.index(x[kTARGET_FIELD]) for x in train[:args.limit]))

        scores = cross_val_score(lr, x_train, y_train, cv = args.kfolds, scoring = 'f1_macro' )
        print ("scores: %s" % (scores))
        print(np.average(scores))
        
        lr.fit(x_train, y_train)
        x_test = feat.test_feature([x[kTEXT_FIELD] for x in test])
        predictions = cross_val_predict(lr, x_train, y_train, cv = args.kfolds) #lr.predict(x_test)
    
    else:
        x_train = feat.train_feature([x[kTEXT_FIELD] for x in train[:args.limit]])
        y_train = array(list(labels.index(x[kTARGET_FIELD]) for x in train[:args.limit]))
        
        x_test = feat.test_feature([x[kTEXT_FIELD] for x in test])
        
        lr.fit(x_train, y_train)
        predictions = lr.predict(x_test)
    

    print("Len(train): %i    Len(y_train): %i " % (len(train), len(y_train)))
    print("Set(y_train): %s" % (set(y_train)))
    
    feat.show_top10(lr, labels)
   

    if args.saveFile:
        predDocName = "../results/predictions " + str(time.strftime("(%Y.%m.%d) %I;%M;%S")) + ".csv"
        
        o = DictWriter(open(predDocName, 'wb'), ["Id", "spoiler"])
        o.writeheader()
        for ii, pp in zip([x['Id'] for x in test], predictions):
            d = {'Id': ii, 'spoiler': labels[pp]}
            o.writerow(d)

    print("done")