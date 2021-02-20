import pandas as pd
import numpy as np
import time
import re
import string
import nltk
import scipy.sparse as sp
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset


def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    clf_predict = clf.predict(xtest)
    # acc = accuracy_score(ytest,clf_predict)
    # ham = hamming_loss(ytest,clf_predict)
    # f1 = f1_score(ytest, clf_predict, average='weighted')
    # p = precision_score(y_test, clf_predict, average='weighted')
    # r = recall_score(y_test, clf_predict, average='weighted')
    # m = multilabel_confusion_matrix(ytest, clf_predict)
    r = classification_report(ytest, clf_predict)
    print(r)
    writeLog(r)

def lowerCase(text):
    text_low = text.str.lower()
    return text_low

def stemming(tokens):
    text = [PorterStemmer().stem(word) for word in tokens]
    return text

def lem(tokens):
     text =[nltk.WordNetLemmatizer().lemmatize(word) for word in tokens]
     return text

def removePunctuation(text):
    text = "".join([c for c in text if c not in string.punctuation])
    return text

def tokenizeText(text):
    tokenized_text = re.split('\W+', text)
    return tokenized_text

def removeStopwords(tokenized_text):
    text = [word for word in tokenized_text if word not in stopwords]
    return text

def writeLog(log):
    f = open(filename, "a")
    logN = str(log)
    f.write(logN + '\n') # Ensures linebreak for automated analysis
    f.close()


def calcMetics(classifier, category,x_train, x_test, y_train, y_test):
    results = []
    results.append(category)
    # results.append(classifier.score(x_test, y_test[category]))
    # results.append(recall_score(y_test[category], classifier.predict(x_test)))
    # results.append(precision_score(y_test[category], classifier.predict(x_test)))
    results.append(f1_score(y_test[category], classifier.predict(x_test)))
    crossF1 = cross_val_score(classifier, x_train, y_train[category], cv=10, scoring='f1_macro')
    crossF1 = ("%0.2f (+/- %0.2f)" % (crossF1.mean(), crossF1.std() * 2))
    results.append(crossF1)
    results.append('')
    writeLog(results)
    return results

def trainClassifier(classifier,category, x_train, x_test, y_train, y_test ):
    classifier.fit(x_train, y_train[category])
    result = calcMetics(classifier, category, x_train, x_test, y_train, y_test )
    return result

def randomForest(x_train, x_test, y_train, y_test, labelTitle):
    rf = RandomForestClassifier(n_estimators=100)
    writeLog("RandomForestClassifier Tree, acc, recall, precision, f1")
    for x in labelTitle:
        result = trainClassifier(rf, x, x_train, x_test, y_train, y_test)
        results.append(result)
#---------- End of functions--------

#Create Log File
path = "./logfiles/"
experimentID = "ID10_"
experiment = "SWR_ST_BOW"
timestamp = int(round(time.time() * 1000))
filename = path+experimentID+experiment +str(timestamp)+'.csv';
f = open(filename, "x")

writeLog("Preporcessing, SWR, Stemming")
writeLog("Feature Extraction, none")
writeLog("Metadata, none")
writeLog("Sentiment, no")

#Read Data
data = pd.read_csv('./SentimentData.csv' , index_col=0)
stopwords = nltk.corpus.stopwords.words('english')

#Prerocessing
tokenized_data = data['comment'].apply(lambda x: tokenizeText(x))
tokenized_data = tokenized_data.apply(lambda x: removeStopwords(x))
tokenized_data = tokenized_data.apply(lambda x: stemming(x)) #Stemming
detokenize_data = tokenized_data.apply(lambda x: TreebankWordDetokenizer().detokenize(x))

#feature extraction
bow = CountVectorizer().fit_transform(detokenize_data)
# ngram = CountVectorizer(ngram_range=(2,4)).fit_transform(detokenize_data)

print("feature extraction complete")

all_features = bow
# all_features = ngram

results = []
labelTitle =['NA','BUG', 'FUNC','NON_FUNC','FR']
all_lables = data[['NA','BUG', 'FUNC','NON_FUNC','FR']]
X_train, X_test, y_train, y_test = train_test_split(all_features, all_lables, test_size=0.2, random_state=42)
randomForest(X_train, X_test, y_train, y_test, labelTitle)
#
# #Multilabel
# all_lablesML = data[['NA','BUG', 'FUNC','NON_FUNC']]
# X_train, X_test, y_train, y_test = train_test_split(all_features, all_lablesML, test_size=0.2, random_state=42)
# clf_chain_model =build_model(RandomForestClassifier(), LabelPowerset, X_train, y_train, X_test, y_test)
#


