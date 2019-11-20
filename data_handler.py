import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 
import json
import random
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

lemmatizer = WordNetLemmatizer() 

NGRAM_RANGE = (1, 2)

TOP_K = 20000

TOKEN_MODE = 'word'

kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
}

vectorizer = TfidfVectorizer(**kwargs)


def preprocess_data():
    with open("intents.json") as file:
        data = json.load(file)

    labels = []
    patterns_text = []
    patterns_label = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            lemmatized = [ lemmatizer.lemmatize(w.lower()) for w in  nltk.word_tokenize(pattern) if w != "?"]
            patterns_text.append(' '.join(lemmatized))
            patterns_label.append(intent["tag"])
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    training = vectorizer.fit_transform(patterns_text).toarray()

    labels = sorted(labels)
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, patter in enumerate(patterns_text):
        output_row = out_empty[:]
        output_row[labels.index(patterns_label[x])] = 1

        output.append(output_row)

    return training, output, labels, data

def handle_predict_data(user_input): 
    lemmatized_user_input = [' '.join([ lemmatizer.lemmatize(w.lower()) for w in  nltk.word_tokenize(user_input)])]
    transformed = vectorizer.transform(lemmatized_user_input).toarray()
    return transformed, lemmatized_user_input