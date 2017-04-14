# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../data"]).decode("utf8")) #msp commented out

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../data/Combined_News_DJIA.csv')
data.head()

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
trainin1 = [data['Top1'][i] + " " + data['Top2'][i] + " " + data['Top3'][i] + " " + data['Top4'][i] + " " + data['Top5'][i] for i in range(len(train))]
testin1 = [data['Top1'][i] + " " + data['Top2'][i] + " " + data['Top3'][i] + " " + data['Top4'][i] + " " + data['Top5'][i] for i in range(len(train), len(train)+len(test))]
djia = pd.read_csv('../data/DJIA_table.csv')
#percents = [100*(a - djia['Open'][i])/a for i, a in enumerate(djia['Close'])]
percents = list(map(int, data['Label']))

trainout = percents[:len(trainin1)]
testout = percents[:len(testin1)]
print("len(testout): {}".format(len(testout)))

count_vect = CountVectorizer()
trainvec = count_vect.fit_transform(trainin1+testin1)

trainin2 = [nltk.word_tokenize(a) for a in trainin1]
testin2 = [nltk.word_tokenize(a) for a in testin1]

trainin = []
testin = []
length = 0
t = []
for b in trainin2[0]:
    if count_vect.vocabulary_.get(b) == None:
        t.append(0)
    else:
        t.append(count_vect.vocabulary_.get(b))
length = len(t)
trainin.append(t)
for i, a in enumerate(trainin2):
    t = []
    for b in a:
        if count_vect.vocabulary_.get(b) == None:
            t.append(0)
        else:
            t.append(count_vect.vocabulary_.get(b))
    trainin.append(t)
    if length < len(t):
        length = len(t)
for i, a in enumerate(testin2):
    t = []
    for b in a:
        if count_vect.vocabulary_.get(b) == None:
            t.append(0)
        else:
            t.append(count_vect.vocabulary_.get(b))
    testin.append(t)
    if length < len(t):
        length = len(t)
for i, a in enumerate(trainin):
    if len(a) < length:
        trainin[i] += [0]*(length - len(a))
for i, a in enumerate(testin):
    if len(a) < length:
        testin[i] += [0]*(length - len(a))

    
trainin = np.array(trainin)[1:]
testin = np.array(testin)
print("trainin.shape: {}".format(trainin.shape))
print("len(trainout) = {}".format(len(trainout)))
print("testin.shape: {}".format(testin.shape))
print("len(testout) = {}".format(len(testout)))
'''
svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
svr_rbf.fit(trainin[:-1], trainout) # fitting the data points in the models
print("fit 1")
svr_lin.fit(trainin[:-1], trainout)
print("fit 2")
r = svr_rbf.predict(testin)
print(r)
l = svr_lin.predict(testin)
print(l)
print(r)
print(l)
print(p)
p1 = [a == testout[i] for i, a in enumerate(r)]
p2 = [a == testout[i] for i, a in enumerate(l)]
ps = [p1, p2]
print([float(a.find(True))/len(a) for a in ps])
'''
svm = MLPClassifier()
svm.fit(trainin, trainout)
r = svm.predict(testin)
results = [r[i] == a for i, a in enumerate(testout)]
t = len([a for a in results if a])
print("t: {}".format(t))
t2 = len([a for a in results if not a])
print("t2: {}".format(t2))
print("float(t)/float(t+t2) = {}".format(float(t)/float(t+t2)))