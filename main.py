import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 
import json
from sklearn.feature_extraction.text import CountVectorizer
import random

lemmatizer = WordNetLemmatizer() 
cv = CountVectorizer()


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
    keras.layers.Dense(len(output[1]), input_shape=(len(training[1]),)),
    keras.layers.Dense(146, activation='relu'),
    keras.layers.Dense(len(output[1]), activation="softmax")
])

model.compile(optimizer="adam", loss=tf.losses.softmax_cross_entropy, metrics=["accuracy"])

model.fit(training, output, epochs=1000)

tag = ""
while tag != "goodbye":
    user_input = input("User: ")

    results = model.predict(cv.transform(
        [' '.join([ lemmatizer.lemmatize(w.lower()) for w in  nltk.word_tokenize(user_input)])]
    ).toarray())
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            print("Robot: " + random.choice(intent['responses']))


