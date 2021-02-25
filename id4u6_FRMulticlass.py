import pandas as pd
import time
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scipy import sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

#-------- Define Functions

def lem(tokens):
     text =[nltk.WordNetLemmatizer().lemmatize(word) for word in tokens]
     return text

def tokenizeText(text):
    tokenized_text = re.split('\W+', text)
    return tokenized_text

def removeStopwords(tokenized_text):
    text = [word for word in tokenized_text if word not in stopwords]
    return text

def calcMetics(classifier, category,x_train, x_test, y_train, y_test ):
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

#Train Classifier
def trainClassifier(classifier,category, x_train, x_test, y_train, y_test ):
    classifier.fit(x_train, y_train[category])
    result = calcMetics(classifier, category, x_train, x_test, y_train, y_test )
    return result

def writeLog(log):
    f = open(filename, "a")
    logN = str(log)
    f.write(logN + '\n') # Ensures linebreak for automated analysis
    f.close()


def multiclassClassificationOvR(classifier, x_train, x_test, y_train, y_test):
    clf = OneVsRestClassifier(classifier).fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    results = classification_report(y_predict, y_test)
    writeLog(results)
    return results

#---------- End of functions--------

#Create Log File
path = "./logfiles/"
experimentID = "ID6_MC_FR"
experiment = "_SWR_LEM_BI_META_BoW"
timestamp = int(round(time.time() * 1000))
filename = path+experimentID+experiment +str(timestamp)+'.csv';
f = open(filename, "x")

#load data
data = pd.read_csv('./SentimentDataFR.csv' , index_col=0)
stopwords = nltk.corpus.stopwords.words('english')

#Execute Preporcessing
tokenized_data = data['comment'].apply(lambda x: tokenizeText(x))
tokenized_data = tokenized_data.apply(lambda x: removeStopwords(x))
tokenized_data = tokenized_data.apply(lambda x: lem(x))
detokenize_data = tokenized_data.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
print("Preprocessing complete")

# Feature Extraction
# Apply BoW Model
bow = CountVectorizer().fit_transform(detokenize_data)
ngram = CountVectorizer(ngram_range=(2,2)).fit_transform(detokenize_data)
# Sentiment
sentiment = data['sentiment']
sentiment_Array = sentiment.apply(lambda x: x+5)
sentiment_Array = sentiment_Array.to_numpy().reshape(-1,1)

#Rating
rating = data['stars']
rating = rating.to_numpy().reshape(-1,1)

#Length
length = data['comment'].apply(lambda x: len(x))
length = length.to_numpy().reshape(-1,1)


#Build feature matrix (4)
# all_features= bow
# joinedMatrix = sp.hstack((bow, sentiment_Array, rating, length), format='csr')

#Build feature matrix (4)
# joinedMatrix = sp.hstack((bow, ngram), format='csr')
joinedMatrix = sp.hstack((bow, ngram, sentiment_Array, rating, length), format='csr')

all_features = joinedMatrix


all_labels =data[['NA', 'BUG', 'FUNC', 'NON_FUNC', 'FR']]
categories= data['categoryFR']
labelTitle = ['NA','BUG', 'FR']

writeLog("Configurations:")
writeLog("Preprocessing: SWR, Lem, Metadata")
writeLog("Feature Extraction: BOW")


# Multiclass Classification
x_train, x_test, y_train, y_test = train_test_split(all_features, categories, test_size=0.2, random_state=42)
writeLog("MultiClass")

writeLog("NB")
result = multiclassClassificationOvR(MultinomialNB(),x_train, x_test, y_train, y_test)
print(result)

writeLog("DT")
result = multiclassClassificationOvR(tree.DecisionTreeClassifier(),x_train, x_test, y_train, y_test)
print(result)

writeLog("LR")
result = multiclassClassificationOvR(LogisticRegression(),x_train, x_test, y_train, y_test)
print(result)


