import pandas as pd
import time
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset


def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    clf_predict = clf.predict(xtest)
    r = classification_report(ytest, clf_predict)
    print(r)
    writeLog(r)

def lowerCase(text):
    text_low = text.str.lower()
    return text_low

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

#---------- End of functions--------

#Create Log File
path = "./logfiles/"
timestamp = int(round(time.time() * 1000))
filename = path+str(timestamp)+'.csv';
f = open(filename, "x")


#Read Data
data = pd.read_csv('../SentimentData63.csv' , index_col=0)
stopwords = nltk.corpus.stopwords.words('english')

#Prerocessing
corpus = data['comment'].apply(nfx.remove_stopwords)
corpus = corpus.apply(nfx.remove_punctuations)
corpus = corpus.apply(nfx.remove_urls)
corpus = corpus.apply(nfx.remove_hashtags)
corpus = corpus.apply(nfx.remove_emails)

#feature extraction
tfidf = TfidfVectorizer().fit_transform(corpus)
all_features = tfidf
plot = data[['NA','BUG', 'FUNC', 'NON_FUNC']]


#Multilabel
all_lablesML = data[['NA','BUG', 'FUNC', 'NON_FUNC']]
X_train, X_test, y_train, y_test = train_test_split(all_features, all_lablesML, test_size=0.2, random_state=42)
writeLog("DT")
clf_chain_model =build_model(DecisionTreeClassifier(), LabelPowerset, X_train, y_train, X_test, y_test)
writeLog("SVC")
clf_chain_model =build_model(SVC(), LabelPowerset, X_train, y_train, X_test, y_test)



