import time
import random
import numpy as np
from numpy import asarray
from numpy import zeros
import pandas as pd
import neattext.functions as nfx
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPool1D, Conv1D, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import CSVLogger
from nltk.tokenize import word_tokenize
import tensorflow as tf

from sklearn.model_selection import KFold


# make keras & tensorfow reproducable
seed_value= 0
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def writeLog(log):
    f = open(filename, "a")
    logN = str(log)
    f.write(logN + '\n') # Ensures linebreak for automated analysis
    f.close()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def buildModel(vocab_length, embedding_matrix, max_length):
    model = Sequential()
    emeddingLayer = Embedding(vocab_length, 300, weights=[embedding_matrix], input_length=max_length)
    model.add(emeddingLayer)
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=3, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=4, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=5, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPool1D())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m])
    return model

def executeANNClassificationForClassCross(className):
    sentiments = data[className]
    n_folds = 10
    cv_scores, model_history = list(), list()
    for _ in range(n_folds):
        X_train, X_val, y_train, y_val = train_test_split(padded_docs, sentiments, test_size=0.10,
                                                          random_state=np.random.randint(1, 1000, 1)[0])
        model = buildModel(vocab_length, embedding_matrix, max_length)
        history = model.fit(X_train, y_train, epochs=50, verbose=1)
        loss, acc, f1 = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(f1)

    writeLog(cv_scores)



# Logfile
#Create Log File
path = "./logfiles/"
timestamp = int(round(time.time() * 1000))
filename = path+str(timestamp)+'.csv';
csv_logger = CSVLogger(filename, append=True, separator=',')

data = pd.read_csv("./SentimentData.csv", index_col=0)
corpus = data['comment']


# preprocessing
corpus = corpus.apply(nfx.remove_stopwords)
corpus = corpus.apply(nfx.remove_punctuations)
corpus = corpus.apply(lambda x: x.lower())

# word statistics
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(corpus)
vocab_length = len(word_tokenizer.word_index) + 1

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpus, key=word_count)
max_length = len(word_tokenize(longest_sentence))

# encoding
encoded_docs = word_tokenizer.texts_to_sequences(corpus)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Embedding
embeddings_dictionary = dict()
glove_file = open('./GoogleNews-vectors-negative300.text', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_length, 300))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



# Deep Learning Classifier
executeANNClassificationForClassCross("NA")
executeANNClassificationForClassCross("BUG")
executeANNClassificationForClassCross("FUNC")
executeANNClassificationForClassCross("NON_FUNC")



