import os
import fnmatch
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import pos_tag,pos_tag_sents
import regex as re
import operator
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
from nltk.corpus import stopwords
import nltk
import numpy
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()


def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def pos(review_without_stopwords):
    return TextBlob(review_without_stopwords).tags


result = pd.read_csv("reviews.csv")

stop = stopwords.words('english')

result['text_without_stopwords'] = result['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

result['text_without_stopwords'] = result['text_without_stopwords'].apply(stem_sentences)

os = result.text.apply(pos)
os1 = pd.DataFrame(os)

os1['pos'] = os1['text'].map(lambda x:" ".join(["/".join(x) for x in x ]) )

result = pd.merge(result, os1,right_index=True,left_index = True)

review_train, review_test, label_train, label_test = train_test_split(result['pos'],result['deceptive'], test_size=0.3,random_state=13)

tf_vect = TfidfVectorizer(lowercase = True, use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(1,3))

X_train_tf = tf_vect.fit_transform(review_train)
X_test_tf = tf_vect.transform(review_test)


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print("##################################################################")
    print(grid_search.cv_results_)
    print("##################################################################")
    print(grid_search.cv)
    print("##################################################################")
    print(grid_search.return_train_score)
    print("##################################################################")
    print(grid_search.scoring)
    print("##################################################################")
    print(grid_search.error_score)
    return grid_search.cv_results_


#print(svc_param_selection(X_train_tf,label_train,5))

clf = svm.SVC(C=10,gamma=0.001,kernel='linear')
clf.fit(X_train_tf,label_train)
pred = clf.predict(X_test_tf)

with open('vectorizer.pickle', 'wb') as fin:
    pickle.dump(tf_vect, fin)

with open('mlmodel.pickle','wb') as f:
    pickle.dump(clf,f)

pkl = open('mlmodel.pickle', 'rb')
clf = pickle.load(pkl)
vec = open('vectorizer.pickle', 'rb')
tf_vect = pickle.load(vec)

X_test_tf = tf_vect.transform(review_test)
pred = clf.predict(X_test_tf)

print(metrics.accuracy_score(label_test, pred))

print (confusion_matrix(label_test, pred))

print (classification_report(label_test, pred))
