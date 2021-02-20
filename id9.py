import pandas as pd
import time
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scipy import sparse as sp
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

#-------- Define Functions
def stemming(tokens):
    text = [PorterStemmer().stem(word) for word in tokens]
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

def trainClassifier(classifier,category, x_train, x_test, y_train, y_test ):
    classifier.fit(x_train, y_train[category])
    result = calcMetics(classifier, category, x_train, x_test, y_train, y_test )
    return result

def gbr(x_train, x_test, y_train, y_test, labelTitle):
    gbr = GradientBoostingClassifier(random_state=0)
    writeLog("GBRegession Tree, acc, recall, precision, f1")
    for x in labelTitle:
        result = trainClassifier(gbr, x, x_train, x_test, y_train, y_test)
        results.append(result)


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

#---------- End of functions--------

#Create Log File
path = "./logfiles/"
experimentID = "ID9_"
experiment = "SWR_ST_ngram_tfidf"
timestamp = int(round(time.time() * 1000))
filename = path+experimentID+experiment +str(timestamp)+'.csv';
f = open(filename, "x")

writeLog("logfile,"+str(timestamp))
writeLog("Preporcessing, SWR, stemming")
writeLog("Feature Extraction, N-Gram(2,3), TFIDF")
writeLog("Sentiment, mo")


#Read Data
data = pd.read_csv('./SentimentData.csv' , index_col=0)
stopwords = nltk.corpus.stopwords.words('english')

#Prerocessing
tokenized_data = data['comment'].apply(lambda x: tokenizeText(x))
tokenized_data = tokenized_data.apply(lambda x: removeStopwords(x))
tokenized_data = tokenized_data.apply(lambda x: stemming(x)) #Stemming
detokenize_data = tokenized_data.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
print("preprocessing complete")


#Feature Extraction
ngram = CountVectorizer(ngram_range=(2,3)).fit_transform(detokenize_data)
tfidf = TfidfVectorizer().fit_transform(detokenize_data)

joinedMatrix = sp.hstack((ngram, tfidf), format='csr')

print("Feature extractionComplete")


# Train Machine Learning Classifier: GBRT
results = []
labelTitle =['NA','BUG', 'FUNC','NON_FUNC']
all_lables = data[[ 'NA','BUG', 'FUNC','NON_FUNC']]
X_train, X_test, y_train, y_test = train_test_split(joinedMatrix, all_lables, test_size=0.2, random_state=42)

writeLog("Feature Extraction: TFIDF Vector Array");
#Execute
gbr(X_train, X_test, y_train, y_test, labelTitle)

# #Multilabel
# X_train, X_test, y_train, y_test = train_test_split(joinedMatrix, all_lables, test_size=0.2, random_state=42)
# clf_chain_modelSVM =build_model(GradientBoostingClassifier(random_state=0), LabelPowerset, X_train, y_train, X_test, y_test)
