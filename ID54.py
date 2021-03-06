import pandas as pd
import time
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from scipy import sparse as sp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import neattext.functions as nfx

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

def stemming(tokens):
    text = [PorterStemmer().stem(word) for word in tokens]
    return text

def lem(tokens):
     text =[nltk.WordNetLemmatizer().lemmatize(word) for word in tokens]
     return text

def tokenizeText(text):
    tokenized_text = re.split('\W+', text)
    return tokenized_text

def removeStopwords(tokenized_text):
    text = [word for word in tokenized_text if word not in stopwords]
    return text

def removePunctuation(text):
    text = "".join([c for c in text if c not in string.punctuation])
    return text

def calcLenghth(text):
    length = len(text)
    return length

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
    result = calcMetics(classifier, category, x_train, x_test, y_train, y_test)
    return result

def naiveBayes(x_train, x_test, y_train, y_test, labelTitle):
    nb = MultinomialNB()
    writeLog("Naive Bayes, acc, recall, precision, f1, crossVAcc, crossVf1")
    for x in labelTitle:
        result = trainClassifier(nb, x, x_train, x_test, y_train, y_test)
        results.append(result)

def j48(x_train, x_test, y_train, y_test, labelTitle):
    j48clf = tree.DecisionTreeClassifier()
    writeLog("J48, acc, recall, precision, f1, crossVAcc, crossVf1")
    for x in labelTitle:
        result = trainClassifier(j48clf, x, x_train, x_test, y_train, y_test)
        results.append(result)

def svm(x_train, x_test, y_train, y_test, labelTitle):
    svmClass = SVC()
    writeLog("SVM, acc, recall, precision, f1, crossVAcc, crossVf1")
    for x in labelTitle:
        result = trainClassifier(svmClass, x, x_train, x_test, y_train, y_test)
        results.append(result)

def lr(x_train, x_test, y_train, y_test, labelTitle):
    lrclass =LogisticRegression()
    writeLog("SVM, acc, recall, precision, f1, crossVAcc, crossVf1")
    for x in labelTitle:
        result = trainClassifier(lrclass, x, x_train, x_test, y_train, y_test)

#---------- End of functions--------

#Create Log File
path = "./logfiles/"
experimentID = "ID54_"
experiment = "SVM_SWR_PR_Lengh"
timestamp = int(round(time.time() * 1000))
filename = path+experimentID+experiment +str(timestamp)+'.csv';
f = open(filename, "x")

#Read Data
data = pd.read_csv('./SentimentData.csv' , index_col=0)
stopwords = nltk.corpus.stopwords.words('english')
# Train Machine Learning Classifier: GBRT
results = []
labelTitle =['NA','BUG', 'FUNC','NON_FUNC']
all_lables = data[['NA','BUG', 'FUNC','NON_FUNC']]

#Prerocessing

corpus = data['comment']
# detokenize_data = corpus.apply(nfx.remove_punctuations)
tokenized_data = corpus.apply(lambda x: tokenizeText(x))
# tokenized_data = tokenized_data.apply(lambda x: removeStopwords(x))
tokenized_data = tokenized_data.apply(lambda x: lem(x)) #Lemmatization
detokenize_data = tokenized_data.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
print("preprocessing complete")


#Classification Techniques
#BOW
uni = CountVectorizer(ngram_range=(1,1)).fit_transform(detokenize_data)

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
joinedMatrix = sp.hstack((uni, sentiment_Array, rating, length), format='csr')
# joinedMatrix = sp.hstack((uni, length), format='csr')

print('Feature Extraction complete')

all_features= joinedMatrix

# #Execute
X_train, X_test, y_train, y_test = train_test_split(all_features, all_lables, test_size=0.2, random_state=42)
# naiveBayes(X_train, X_test, y_train, y_test, labelTitle)
# j48(X_train, X_test, y_train, y_test, labelTitle)
# lr(X_train, X_test, y_train, y_test, labelTitle)
svm(X_train, X_test, y_train, y_test, labelTitle)





# #Multilabel
# writeLog("MultiLabel")
# all_labelsML =data[['NA', 'BUG', 'FUNC', 'NON_FUNC']]
# X_train, X_test, y_train, y_test = train_test_split(all_features, all_labelsML, test_size=0.2, random_state=42)
# # writeLog("NB")
# clf_chain_model = build_model(MultinomialNB(), LabelPowerset, X_train, y_train, X_test, y_test)
# writeLog("DT")
# clf_chain_modelSVM =build_model(tree.DecisionTreeClassifier(), LabelPowerset, X_train, y_train, X_test, y_test)
# writeLog("LR")
# clf_chain_modelSVM =build_model(LogisticRegression(), LabelPowerset, X_train, y_train, X_test, y_test)
# writeLog("SVM")
# clf_chain_modelSVM =build_model(SVC(), LabelPowerset, X_train, y_train, X_test, y_test)