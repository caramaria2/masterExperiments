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

#Calculate Metrics for Classifier
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

def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    clf_predict = clf.predict(xtest)
    # acc = accuracy_score(ytest,clf_predict)
    # ham = hamming_loss(ytest,clf_predict)
    # f1 = f1_score(ytest, clf_predict, average='weighted')
    # p = precision_score(y_test, clf_predict, average='weighted')
    # r = recall_score(y_test, clf_predict, average='weighted')
    r = classification_report(ytest, clf_predict)
    print(r)
    writeLog(r)

def multiclassClassificationOvR(classifier, x_train, x_test, y_train, y_test):
    clf = OneVsRestClassifier(classifier).fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    results = classification_report(y_predict, y_test)
    writeLog(results)
    return results

#---------- End of functions--------

#Create Log File
path = "./logfiles/"
experimentID = "ID6_"
experiment = "SWR_LEM_Bigram_BOW_Meta"
timestamp = int(round(time.time() * 1000))
filename = path+experimentID+experiment +str(timestamp)+'.csv';
f = open(filename, "x")


#load data
data = pd.read_csv('./SentimentData.csv' , index_col=0)
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
#Build feature matrix
# joinedMatrix = sp.hstack((bow, ngram), format='csr')
joinedMatrix = sp.hstack((bow, ngram, sentiment_Array, rating, length), format='csr')

all_labels =data[['NA', 'BUG', 'FUNC', 'NON_FUNC', 'FR']]
categories= data['category']
labelTitle = ['NA','BUG', 'FUNC', 'NON_FUNC', 'FR']

writeLog("Configurations:")
writeLog("Preprocessing: SWR, Lem")
writeLog("Feature Extraction: BOW, NGRAM, META")

# Binary Class
x_train, x_test, y_train, y_test = train_test_split(joinedMatrix, all_labels, test_size=0.2, random_state=42)
writeLog("Category, Accuacy, Recall, Precision, F1")
writeLog("Binary Class")

writeLog("NB")
for x in labelTitle:
    trainClassifier(MultinomialNB(), x, x_train, x_test, y_train, y_test)

writeLog("DT")
for x in labelTitle:
    trainClassifier(tree.DecisionTreeClassifier(), x, x_train, x_test, y_train, y_test)

writeLog("LR")
for x in labelTitle:
    trainClassifier(LogisticRegression(), x, x_train, x_test, y_train, y_test)


# Multiclass Classification
# x_train, x_test, y_train, y_test = train_test_split(joinedMatrix, categories, test_size=0.3, random_state=42)
# writeLog("MultiClass")
#
# writeLog("NB")
# result = multiclassClassificationOvR(MultinomialNB(),x_train, x_test, y_train, y_test)
# print(result)
#
# writeLog("DT")
# result = multiclassClassificationOvR(tree.DecisionTreeClassifier(),x_train, x_test, y_train, y_test)
# print(result)
#
# writeLog("LR")
# result = multiclassClassificationOvR(LogisticRegression(),x_train, x_test, y_train, y_test)
# print(result)

# #Multilabel
# writeLog("MultiLabel")
# all_labelsML =data[['NA', 'BUG', 'FUNC', 'NON_FUNC']]
# X_train, X_test, y_train, y_test = train_test_split(joinedMatrix, all_labelsML, test_size=0.2, random_state=42)
# writeLog("NB")
# clf_chain_model = build_model(MultinomialNB(), LabelPowerset, X_train, y_train, X_test, y_test)
# writeLog("DT")
# clf_chain_modelSVM =build_model(tree.DecisionTreeClassifier(), LabelPowerset, X_train, y_train, X_test, y_test)
# writeLog("LR")
# clf_chain_modelSVM =build_model(LogisticRegression(), LabelPowerset, X_train, y_train, X_test, y_test)
