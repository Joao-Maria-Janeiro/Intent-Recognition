import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
from sklearn.feature_extraction.text import CountVectorizer


stemmer = LancasterStemmer()
cv = CountVectorizer()


with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
patterns_text = []
patterns_label = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        stemmed = [ stemmer.stem(w.lower()) for w in  nltk.word_tokenize(pattern)]
        words.extend(stemmed)
        patterns_text.append(pattern)
        patterns_label.append(intent["tag"])
    
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
    

words = sorted(list(set(words)))
training = cv.fit_transform(patterns_text).toarray()

labels = sorted(labels)


output = []
out_empty = [0 for _ in range(len(labels))]

for x, patter in enumerate(patterns_text):
    output_row = out_empty[:]
    output_row[labels.index(patterns_label[x])] = 1

    output.append(output_row)

training = np.array(training)
output = np.array(output)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(len(training[0]), len(training[1]))),
    keras.layers.Dense(146, activation="relu"),
    keras.layers.Dense(len(training[0]), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training, output, epochs=1000)